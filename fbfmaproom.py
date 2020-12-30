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
import dash_daq as daq
import dash_leaflet as dl
import dash_leaflet.express as dlx
from dash.dependencies import Output, Input, State, ALL

import pyaconf
import queuepool

# import cartopy
import cartopy.crs as crs

# import matplotlib
import matplotlib.pyplot as plt


# import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


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


def from_months_since(x, year_since=1960):
    int_x = int(x)
    return datetime.date(
        int_x // 12 + year_since, int_x % 12 + 1, int(30 * (x - int_x) + 1)
    )


from_months_since_v = np.vectorize(from_months_since)


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


def parse_colormap(s: str) -> np.ndarray:
    cm = [
        RGBA(0, 0, 0, 0)
        if x == "null"
        else RGBA(int(x) >> 16 & 0xFF, int(x) >> 8 & 0xFF, int(x) >> 0 & 0xFF, 255)
        for x in s[1:-1].split(" ")
    ]
    return np.array([cm[int(i / 256.0 * len(cm))] for i in range(0, 256)], np.uint8)


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

    rain = xr.open_dataset("rain-madagascar.nc", decode_times=False)[
        "prcp_est"
    ].transpose("Y", "X", ...)
    rs["rain"] = DataArrayEntry(
        "rain", rain, None, None, None, parse_colormap(rain.attrs["colormap"])
    )
    print("*** colormap:", rs["rain"].colormap, rs["rain"].colormap.shape)
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


# Server

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
                            dl.Popup([html.H2("Pixel"), html.P(id="marker_text")]),
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
                        options=[
                            dict(
                                label=pd.to_datetime(v, format="%m").month_name(),
                                value=v,
                            )
                            for v in range(8, 12)
                        ],
                        value=10,
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
                        options=[dict(label="DJF", value=v) for v in range(1, 2)],
                        value=1,
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
                        min=1982,
                        max=2020,
                        value=2020,
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


def generate_table(year):
    time.sleep(1)

    if year is None:
        return html.Div()

    return html.Table(
        [
            html.Thead(
                [
                    html.Tr(
                        [html.Th([f"{x}:"], style={"color": "#4166B2"})]
                        + [html.Th([str(year)]) for _ in range(4)]
                    )
                    for x in [
                        "Worthy-action",
                        "Act-in-vain",
                        "Fail-to-act",
                        "Worthy-Inaction",
                        "Rate",
                    ]
                ]
                + [
                    html.Tr(
                        [
                            html.Th(
                                [x],
                                style={"background-color": "rgb(240, 240, 240)"},
                            )
                            for x in [
                                "Year",
                                "ENSO State",
                                "Forecast, %",
                                "Rain Rank",
                                "Farmers' reported Bad Years",
                            ]
                        ]
                    )
                ]
            ),
            html.Tbody(
                [
                    html.Tr([html.Td([y])] + [html.Td() for _ in range(4)])
                    for y in range(2020, 1999, -1)
                ]
            ),
        ],
    )


def table_layout():
    return html.Div(
        [
            dcc.Loading(
                [],
                id="table_panel",
                type="dot",
                parent_style={"height": "100%"},
                style={"opacity": 0.2},
            )
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


def calculate_polygon(mode, pt, res):
    if mode == "Pixel":
        (x0, y0), (x1, y1) = calculate_bounds(pt, res)
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    else:
        (x0, y0), (x1, y1) = calculate_bounds(pt, res)
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y0)]


def country(pathname: str) -> str:
    return pathname.split("/")[2]


@app.callback(
    Output("logo", "src"),
    Output("map", "center"),
    Output("map", "zoom"),
    Output("marker", "position"),
    Input("location", "pathname"),
)
def _(pathname):
    c = CS[country(pathname)]
    return f"{PFX}/assets/{c['logo']}", c["center"], c["zoom"], c["marker"]


@app.callback(
    Output("feature", "positions"),
    Output("marker_text", "children"),
    Input("location", "pathname"),
    Input("marker", "position"),
    Input("mode", "value"),
)
def _(pathname, position, mode):
    c = CS[country(pathname)]
    positions = calculate_polygon(mode, position, c["resolution"])
    return positions, str(positions)


@app.callback(
    Output("table_panel", "children"),
    Input("year", "value"),
    Input("issue_month", "value"),
    Input("season", "value"),
    Input("freq", "value"),
    Input("feature", "positions"),
)
def _(year, issue_month, season, freq, positions):
    print("*** callback year:", year, issue_month, season, freq, positions)
    return [generate_table(year)]


# Endpoints


@server.route(f"/tiles/<data_array>/<int:tz>/<int:tx>/<int:ty>")
def tiles(data_array, tz, tx, ty):
    dae = DATA_ARRAYS[data_array]
    z = produce_tile(dae.interp2d, tx, ty, tz, 256, 256)
    im = cv2.flip((z - dae.min_val) * 255 / (dae.max_val - dae.min_val), 0)
    im2 = produce_test_tile(256, 256, f"{tx},{ty}x{tz}")
    # im += np.max(im2, axis=2)
    # cv2.imwrite(f"tiles/{tx},{ty}x{tz}.png", cv2.LUT(im.astype(np.uint8), np.fromiter(range(255, -1, -1), np.uint8)))

    """
    print(
        "*** im:",
        im.shape,
        DATA_ARRAYS["rain"].colormap.shape,
        np.max(im),
        np.min(im),
    )
    """
    im = apply_colormap(im, DATA_ARRAYS["rain"].colormap)
    cv2_imencode_success, buffer = cv2.imencode(".png", im)
    assert cv2_imencode_success
    io_buf = io.BytesIO(buffer)
    resp = flask.send_file(io_buf, mimetype="image/png")
    resp.headers["Cache-Control"] = "private, max-age=0, no-cache, no-store"
    return resp


if __name__ == "__main__":
    app.run_server()
