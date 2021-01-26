from typing import Any, Dict, Tuple, List, Literal, Optional, Union, Callable, Hashable
from typing import NamedTuple

import sys
import os
import json
import time
import math
import datetime
import io

import numpy as np
import pandas as pd
import xarray as xr
from scipy import interpolate
import cv2


import requests
import flask

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as table
import dash_daq as daq
import dash_leaflet as dl
import dash_leaflet.express as dlx
from dash.dependencies import Output, Input, State, ALL
from dash.exceptions import PreventUpdate

from shapely import wkb
from shapely.geometry.multipolygon import MultiPolygon
from psycopg2 import sql
import psycopg2.extensions
from queuepool.pool import Pool
from queuepool.psycopg2cm import ConnectionManagerExtended

import pyaconf


CONFIG = pyaconf.load(os.environ["CONFIG"])
CS = CONFIG["countries"]


# Functions and definitions


FuncInterp2d = Callable[[np.ndarray, np.ndarray], np.ndarray]


class DataArrayEntry(NamedTuple):
    name: str
    data_array: xr.DataArray
    interp2d: Optional[FuncInterp2d]
    min_val: Optional[float]
    max_val: Optional[float]
    colormap: Optional[np.ndarray]


class Extent(NamedTuple):
    dim: str
    left: float
    right: float
    point_width: float


class RGB(NamedTuple):
    red: int
    green: int
    blue: int


class RGBA(NamedTuple):
    red: int
    green: int
    blue: int
    alpha: int


def initDBPool(name, config):
    dbpoolConf = config[name]
    dbpool = Pool(
        name=dbpoolConf["name"],
        capacity=dbpoolConf["capacity"],
        maxIdleTime=dbpoolConf["max_idle_time"],
        maxOpenTime=dbpoolConf["max_open_time"],
        maxUsageCount=dbpoolConf["max_usage_count"],
        closeOnException=dbpoolConf["close_on_exception"],
    )
    for i in range(dbpool.capacity):
        dbpool.put(
            ConnectionManagerExtended(
                name=dbpool.name + "-" + str(i),
                autocommit=False,
                isolation_level=psycopg2.extensions.ISOLATION_LEVEL_SERIALIZABLE,
                dbname=dbpool.name,
                host=dbpoolConf["host"],
                port=dbpoolConf["port"],
                user=dbpoolConf["user"],
                password=dbpoolConf["password"],
            )
        )
    if dbpoolConf["recycle_interval"] is not None:
        dbpool.startRecycler(dbpoolConf["recycle_interval"])
    return dbpool


def obtain_geometry(point: Tuple[float, float], table: str, config) -> MultiPolygon:
    y, x = point
    with dbpool.take() as cm:
        conn = cm.resource
        with conn:  # transaction
            df = pd.read_sql(
                sql.SQL(
                    """
                    with a as(
                        select gid, the_geom,
                            ST_SetSRID(ST_MakePoint(%(x)s, %(y)s),4326) as pt,
                            adm0_name, adm1_name, adm2_name
                            from g2015_2014_2)
                    select gid, ST_AsBinary(the_geom) as the_geom, pt,
                        adm0_name, adm1_name, adm2_name
                        from a
                        where the_geom && pt and ST_Contains(the_geom, pt) and
                            adm0_name = %(adm0_name)s
                    """
                ).format(sql.Identifier(table)),
                conn,
                params=dict(x=x, y=y, adm0_name=config["adm0_name"]),
            )
    # print("bytes: ", sum(df.the_geom.apply(lambda x: len(x.tobytes()))), "x, y: ", x, y)
    df["the_geom"] = df["the_geom"].apply(lambda x: wkb.loads(x.tobytes()))
    if len(df.index) != 0:
        res = df["the_geom"].values[0]
        if not isinstance(res, MultiPolygon):
            # make a MultiPolygon out of a single polygon
            res = MultiPolygon([res])
        attrs = {k: vs[0] for k, vs in df.iteritems() if k not in ("the_geom", "pt")}
    else:
        res = None
        attrs = None
    return res, attrs


def from_months_since(x, year_since=1960):
    int_x = int(x)
    return datetime.date(
        int_x // 12 + year_since, int_x % 12 + 1, int(30 * (x - int_x) + 1)
    )


from_months_since_v = np.vectorize(from_months_since)


def to_months_since(d, year_since=1960):
    return (d.year - year_since) * 12 + (d.month - 1) + (d.day - 1) / 30.0


def extent(da: xr.DataArray, dim: str, default: Optional[Extent] = None) -> Extent:
    if default is None:
        default = Extent(dim, np.nan, np.nan, np.nan)
    coord = da[dim]
    n = len(coord.values)
    if n == 0:
        res = Extent(dim, np.nan, np.nan, default.point_width)
    elif n == 1:
        res = Extent(
            dim,
            coord.values[0] - default.point_width / 2,
            coord.values[0] + default.point_width / 2,
            default.point_width,
        )
    else:
        point_width = coord.values[1] - coord.values[0]
        res = Extent(
            dim,
            coord.values[0] - point_width / 2,
            coord.values[-1] + point_width / 2,
            point_width,
        )
    return res


def extents(
    da: xr.DataArray, dims: List[str] = None, defaults: Optional[List[Extent]] = None
) -> List[Extent]:
    if dims is None:
        dims = da.dims
    if defaults is None:
        defaults = [Extent(dim, np.nan, np.nan, np.nan) for dim in dims]
    return [extent(da, k, defaults[i]) for i, k in enumerate(dims)]


def g_lon(tx: int, tz: int) -> float:
    return tx * 360 / 2 ** tz - 180


def g_lat(tx: int, tz: int) -> float:
    return tx * 180 / 2 ** tz - 90


def g_lat_3857(tx: int, tz: int) -> float:
    a = math.pi - 2 * math.pi * tx / 2 ** tz
    return 180 / math.pi * math.atan(0.5 * (math.exp(a) - math.exp(-a)))


def tile_extents(g: Callable[[int, int], float], tx: int, tz: int, n: int = 1):
    assert n >= 1 and tz >= 0 and 0 <= tx < 2 ** tz
    a = g(tx, tz)
    for i in range(1, n + 1):
        b = g(tx + i / n, tz)
        yield a, b
        a = b


def create_interp2d(
    da: xr.DataArray, dims: Tuple[str, str]
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    x = da[dims[1]].values
    y = da[dims[0]].values
    z = da.values
    f = interpolate.interp2d(x, y, z, kind="linear", copy=False, bounds_error=False)
    return f


def produce_tile(
    interp2d: Callable[[np.ndarray, np.ndarray], np.ndarray],
    tx: int,
    ty: int,
    tz: int,
    tile_width: int = 256,
    tile_height: int = 256,
) -> np.ndarray:
    x = np.fromiter(
        (a + (b - a) / 2.0 for a, b in tile_extents(g_lon, tx, tz, tile_width)),
        np.double,
    )
    y = np.fromiter(
        (a + (b - a) / 2.0 for a, b in tile_extents(g_lat_3857, ty, tz, tile_height)),
        np.double,
    )
    # print("*** produce_tile:", tz, tx, ty, x[0], x[-1], y[0], y[-1])
    z = interp2d(x, y)
    return z


def produce_test_tile(w: int = 256, h: int = 256, text: str = "") -> np.ndarray:
    line_thickness = 1
    line_type = cv2.LINE_AA  # cv2.LINE_4 | cv2.LINE_8 | cv2.LINE_AA
    red_color = (0, 0, 255, 255)
    green_color = (0, 255, 0, 255)
    blue_color = (255, 0, 0, 255)

    layer1 = np.zeros((h, w, 4), np.uint8)
    layer2 = np.zeros((h, w, 4), np.uint8)
    layer3 = np.zeros((h, w, 4), np.uint8)

    cv2.ellipse(
        layer1,
        (w // 2, h // 2),
        (w // 3, h // 3),
        0,
        0,
        360,
        red_color,
        line_thickness,
        lineType=line_type,
    )

    cv2.putText(
        layer1,
        text,
        (w // 2, h // 2),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=red_color,
        thickness=1,
        lineType=line_type,
    )

    cv2.rectangle(
        layer2, (0, 0), (w - 1, h - 1), green_color, line_thickness, lineType=line_type
    )
    cv2.line(
        layer3, (0, 0), (w - 1, h - 1), blue_color, line_thickness, lineType=line_type
    )
    cv2.line(
        layer3, (0, h - 1), (w - 1, 0), blue_color, line_thickness, lineType=line_type
    )

    res = layer1[:]
    cnd = layer2[:, :, 3] > 0
    res[cnd] = layer2[cnd]
    cnd = layer3[:, :, 3] > 0
    res[cnd] = layer3[cnd]

    return res


# DataArrays


def correct_coord(da: xr.DataArray, coord_name: str) -> xr.DataArray:
    coord = da[coord_name]
    values = coord.values
    min_val = values[0]
    max_val = values[-1]
    n = len(values)
    point_width = (max_val - min_val) / n
    vs = np.fromiter(
        ((min_val + point_width / 2.0) + i * point_width for i in range(n)), np.double
    )
    return da.assign_coords({coord_name: vs})


def parse_color(s: str) -> RGBA:
    v = int(s)
    return RGBA(v >> 16 & 0xFF, v >> 8 & 0xFF, v >> 0 & 0xFF, 255)


def parse_color_item(vs: List[RGBA], s: str) -> List[RGBA]:
    if s == "null":
        rs = [RGBA(0, 0, 0, 0)]
    elif s[0] == "[":
        rs = [parse_color(s[1:])]
    elif s[-1] == "]":
        n = int(s[:-1])
        assert 1 < n <= 256 and len(vs) != 0
        rs = [vs[-1]] * (n - 1)
    else:
        rs = [parse_color(s)]
    return vs + rs


def parse_colormap(s: str) -> np.ndarray:
    vs = []
    for x in s[1:-1].split(" "):
        vs = parse_color_item(vs, x)
    # print("*** CM cm:", len(vs), [f"{v.red:02x}{v.green:02x}{v.blue:02x}" for v in vs])
    rs = np.array([vs[int(i / 256.0 * len(vs))] for i in range(0, 256)], np.uint8)
    return rs


def apply_colormap(im: np.ndarray, colormap: np.ndarray) -> np.ndarray:
    im = im.astype(np.uint8)
    im = cv2.merge(
        [
            cv2.LUT(im, colormap[:, 2]),
            cv2.LUT(im, colormap[:, 1]),
            cv2.LUT(im, colormap[:, 0]),
            cv2.LUT(im, colormap[:, 3]),
        ]
    )
    return im


def open_data_arrays():
    rs = {}
    bath = xr.open_dataset("bath432.nc", decode_times=False)["bath"].transpose("Y", "X")
    # bath = correct_coord(bath, "Y")
    # bath = correct_coord(bath, "X")

    bath = xr.where(bath < 0.0, 0.0, bath)
    rs["bath"] = DataArrayEntry(
        "bath", bath, create_interp2d(bath, bath.dims), 0.0, 7964.0, None
    )
    print(bath, extents(bath))

    rain = xr.open_dataset("rain-noaa.nc", decode_times=False)["prcp_est"].transpose(
        "Y", "X", ...
    )
    rs["rain"] = DataArrayEntry(
        "rain", rain, None, None, None, parse_colormap(rain.attrs["colormap"])
    )
    # print("*** colormap:", rs["rain"].colormap, rs["rain"].colormap.shape)
    print(rain, extents(rain, ["Y", "X"]))
    # print(from_months_since_v(rain["T"].values))

    pnep = xr.open_dataset("pnep-malawi.nc", decode_times=False)["prob"].transpose(
        "Y", "X", ...
    )
    pnep["T"] = pnep["S"] + pnep["L"]
    rs["pnep"] = DataArrayEntry("pnep", pnep, None, None, None, None)
    print(
        pnep,
        extents(pnep, ["Y", "X"]),
        pnep["S"].values.shape,
        pnep["L"].values.shape,
        pnep["T"].values.shape,
    )
    # print(from_months_since_v(pnep["S"].values))
    # print(from_months_since_v(pnep["T"].values))
    # print(pnep.sel(S=274, L=2.5).values)

    return rs


DATA_ARRAYS: Dict[str, DataArrayEntry] = open_data_arrays()

TABLE_COLUMNS = [
    "Year",
    "ENSO State",
    "Forecast, %",
    "Rain Rank",
    "Farmers' reported Bad Years",
]

SUMMARY_ROWS = [
    "Worthy-action",
    "Act-in-vain",
    "Fail-to-act",
    "Worthy-Inaction",
    "Rate",
]


def generate_table(config, year, issue_month, season, freq, positions):
    time.sleep(1)

    year_min, year_max = config["seasons"][season]["year_range"]

    df = pd.DataFrame({k: [] for k in TABLE_COLUMNS})
    df[df.columns[0]] = [x for x in range(year_max, year_min - 1, -1)]

    return df


# Server


dbpool = initDBPool("dbpool", CONFIG)

PFX = "/fbfmaproom"
server = flask.Flask(__name__)
app = dash.Dash(
    __name__,
    server=server,
    url_base_pathname=f"{PFX}/",
    meta_tags=[
        {"name": "description", "content": "content description 1234"},
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
    ],
)


# Layout


def map_layout():
    return dl.Map(
        [
            dl.LayersControl(
                [
                    dl.BaseLayer(
                        dl.TileLayer(
                            url="https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
                        ),
                        name="Street",
                        checked=True,
                    ),
                    dl.BaseLayer(
                        dl.TileLayer(
                            url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
                        ),
                        name="Topo",
                        checked=False,
                    ),
                    dl.Overlay(
                        dl.TileLayer(
                            url="/tiles/bath/{z}/{x}/{y}",
                            opacity=0.6,
                        ),
                        name="Rain",
                        checked=True,
                    ),
                ],
                position="topleft",
                id="layers_control",
            ),
            dl.LayerGroup(
                [
                    dl.Polygon(
                        positions=[(0, 0), (0, 0)],
                        color="rgb(49, 109, 150)",
                        fillColor="orange",
                        fillOpacity=0.1,
                        weight=1,
                        id="feature",
                    ),
                    dl.Marker(
                        [
                            dl.Popup(id="marker_popup"),
                        ],
                        position=(0, 0),
                        draggable=True,
                        id="marker",
                    ),
                ],
                id="pixel_layer",
            ),
            dl.ScaleControl(imperial=False),
        ],
        id="map",
        style={
            "width": "100%",
            "height": "100%",
            "position": "absolute",
        },
    )


def logo_layout():
    return html.Div(
        [html.H4("FBF—Maproom"), html.Img(id="logo")],
        id="logo_panel",
        className="info",
        style={
            "position": "absolute",
            "top": "10px",
            "width": "110px",
            "left": "90px",
            "z-index": "1000",
            "height": "75px",
            "pointer-events": "auto",
            "padding-left": "10px",
            "padding-right": "10px",
        },
    )


def command_layout():
    return html.Div(
        [
            html.Div(
                [
                    html.Label("Mode:"),
                    dcc.Dropdown(
                        id="mode",
                        options=[
                            dict(
                                label=v,
                                value=v,
                            )
                            for v in ["District", "Regional", "National", "Pixel"]
                        ],
                        value="District",
                        clearable=False,
                    ),
                ],
                style={
                    "width": "100px",
                    "display": "inline-block",
                    "padding": "10px",
                    "vertical-align": "top",
                },
            ),
            html.Div(
                [
                    html.Label("Issue month:"),
                    dcc.Dropdown(
                        id="issue_month",
                        clearable=False,
                    ),
                ],
                style={
                    "width": "100px",
                    "display": "inline-block",
                    "padding": "10px",
                    "vertical-align": "top",
                },
            ),
            html.Div(
                [
                    html.Label("Leads:"),
                    dcc.Dropdown(
                        id="season",
                        clearable=False,
                    ),
                ],
                style={
                    "width": "100px",
                    "display": "inline-block",
                    "padding": "10px",
                    "vertical-align": "top",
                },
            ),
            html.Div(
                [
                    html.Label("Year:"),
                    dcc.Input(
                        id="year",
                        type="number",
                        step=1,
                        style={
                            "height": "32px",
                            "width": "95%",
                            "border-color": "rgb(200, 200, 200)",
                            "border-width": "1px",
                            "border-style": "solid",
                            "border-radius": "4px",
                            "text-indent": "8px",
                        },
                    ),
                ],
                style={
                    "width": "100px",
                    "display": "inline-block",
                    "padding": "10px",
                    "vertical-align": "top",
                },
            ),
            html.Div(
                [
                    html.Label("Frequency of triggered forecasts:"),
                    dcc.RangeSlider(
                        id="freq",
                        min=5,
                        max=95,
                        step=10,
                        value=(15, 25),
                        marks={k: dict(label=f"{k}%") for k in range(5, 96, 10)},
                        pushable=5,
                        included=False,
                    ),
                ],
                style={
                    "width": "400px",
                    "display": "inline-block",
                    "padding": "10px",
                    "vertical-align": "top",
                },
            ),
        ],
        id="command_panel",
        className="info",
        style={
            "position": "absolute",
            "top": "10px",
            "right": "10px",
            "left": "230px",
            "z-index": "1000",
            "height": "75px",
            "pointer-events": "auto",
            "padding-left": "10px",
            "padding-right": "10px",
        },
    )


def table_layout():
    return html.Div(
        [
            html.Div(id="log"),
            dcc.Loading(
                [
                    table.DataTable(
                        id="table",
                        columns=[{"name": x, "id": x} for x in TABLE_COLUMNS],
                        page_action="none",
                        style_table={
                            "height": "600px",
                            "overflowY": "auto",
                            "border": "1px solid rgb(240, 240, 240)",
                        },
                        style_header={
                            "border": "1px solid rgb(240, 240, 240)",
                        },
                        style_cell={
                            "whiteSpace": "normal",
                            "height": "auto",
                            "textAlign": "center",
                            "border": "1px solid rgb(240, 240, 240)",
                        },
                        fixed_rows={
                            "headers": False,
                            "data": 0,
                        },
                    ),
                ],
                type="dot",
                parent_style={"height": "100%"},
                style={"opacity": 0.2},
            ),
        ],
        className="info",
        style={
            "position": "absolute",
            "top": "110px",
            "right": "10px",
            "z-index": "1000",
            "height": "80%",
            "width": "600px",
            "pointer-events": "auto",
            "padding-left": "10px",
            "padding-right": "10px",
        },
    )


def app_layout():
    return html.Div(
        [
            dcc.Location(id="location", refresh=True),
            map_layout(),
            logo_layout(),
            command_layout(),
            table_layout(),
        ]
    )


app.layout = app_layout()

# Callbacks


def calculate_bounds(pt, res):
    x, y = pt
    dx, dy = res
    return [[x // dx * dx, y // dy * dy], [x // dx * dx + dx, y // dy * dy + dy]]


def country(pathname: str) -> str:
    return pathname.split("/")[2]


@app.callback(
    Output("logo", "src"),
    Output("map", "center"),
    Output("map", "zoom"),
    Output("marker", "position"),
    Output("season", "options"),
    Output("season", "value"),
    Input("location", "pathname"),
)
def _(pathname):
    c = CS[country(pathname)]
    season_options = [
        dict(
            label=c["seasons"][k]["label"],
            value=k,
        )
        for k in sorted(c["seasons"].keys())
    ]
    season_value = min(c["seasons"].keys())

    return (
        f"{PFX}/assets/{c['logo']}",
        c["center"],
        c["zoom"],
        c["marker"],
        season_options,
        season_value,
    )


@app.callback(
    Output("year", "min"),
    Output("year", "max"),
    Output("year", "value"),
    Output("issue_month", "options"),
    Output("issue_month", "value"),
    Input("season", "value"),
    Input("location", "pathname"),
)
def _(season, pathname):
    c = CS[country(pathname)]["seasons"][season]
    year_min, year_max = c["year_range"]
    issue_month_options = [
        dict(
            label=pd.to_datetime(int(v) + 1, format="%m").month_name(),
            value=int(v) + 1,
        )
        for v in reversed(c["issue_months"])
    ]
    issue_month_value = int(c["issue_months"][-1]) + 1

    return (
        year_min,
        year_max,
        year_max,
        issue_month_options,
        issue_month_value,
    )


@app.callback(
    Output("log", "children"),
    Input("map", "click_lat_lng"),
)
def _(position):
    return str(position)


@app.callback(
    Output("feature", "positions"),
    Output("marker_popup", "children"),
    Input("location", "pathname"),
    Input("marker", "position"),
    Input("mode", "value"),
)
def _(pathname, position, mode):
    c = CS[country(pathname)]
    title = mode
    content = ""
    positions = None
    if mode == "Pixel":
        (x0, y0), (x1, y1) = calculate_bounds(position, c["resolution"])
        positions = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        title += " " + str((round((x0 + x1) / 2, 2), round((y0 + y1) / 2, 2)))
    else:
        geom, attrs = obtain_geometry(position, "g2015_2014_2", c)
        print("*** geom geom: ", attrs)
        if geom is not None:
            xs, ys = geom[-1].exterior.coords.xy
            positions = list(zip(ys, xs))
            title += " " + attrs["adm2_name"] + " " + str(len(geom))
            content = str(
                dict(marker=(round(position[1], 2), round(position[0], 2))) | attrs
            )

    if positions is None:
        raise PreventUpdate

    return positions, [html.H2(title), html.P(content)]


@app.callback(
    Output("table", "data"),
    Input("year", "value"),
    Input("issue_month", "value"),
    Input("freq", "value"),
    Input("feature", "positions"),
    Input("location", "pathname"),
    State("season", "value"),
)
def _(year, issue_month, freq, positions, pathname, season):
    print(
        "*** callback table:", year, issue_month, season, freq, len(positions), pathname
    )
    c = CS[country(pathname)]
    return generate_table(c, year, issue_month, season, freq, positions).to_dict(
        "records"
    )


# Endpoints


@server.route(f"/tiles/<data_array>/<int:tz>/<int:tx>/<int:ty>")
def tiles(data_array, tz, tx, ty):
    dae = DATA_ARRAYS[data_array]
    z = produce_tile(dae.interp2d, tx, ty, tz, 256, 256)
    im = cv2.flip((z - dae.min_val) * 255 / (dae.max_val - dae.min_val), 0)
    im2 = produce_test_tile(256, 256, f"{tx},{ty}x{tz}")
    # im += np.max(im2, axis=2)
    # cv2.imwrite(f"tiles/{tx},{ty}x{tz}.png", cv2.LUT(im.astype(np.uint8), np.fromiter(range(255, -1, -1), np.uint8)))
    im = apply_colormap(im, DATA_ARRAYS["rain"].colormap)
    cv2_imencode_success, buffer = cv2.imencode(".png", im)
    assert cv2_imencode_success
    io_buf = io.BytesIO(buffer)
    resp = flask.send_file(io_buf, mimetype="image/png")
    resp.headers["Cache-Control"] = "private, max-age=0, no-cache, no-store"
    return resp


if __name__ == "__main__":
    app.run_server()
