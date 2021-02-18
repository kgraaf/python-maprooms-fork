from typing import Any, Dict, Tuple, List, Literal, Optional, Union, Callable, Hashable
import os
import time
import io
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import cv2
import flask
import dash
import dash_html_components as html
from dash.dependencies import Output, Input, State, ALL
from dash.exceptions import PreventUpdate
from shapely import wkb
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry import Polygon
from psycopg2 import sql
import pyaconf
import pingrid
from pingrid import BGRA
import fbflayout


CONFIG = pyaconf.load(os.environ["CONFIG"])
CS = CONFIG["countries"]

DBPOOL = pingrid.init_dbpool("dbpool", CONFIG)

TABLE_COLUMNS = [
    dict(id="year_label", name="Year"),
    dict(id="enso_state", name="ENSO State"),
    dict(id="forecast", name="Forecast, %"),
    dict(id="rain_rank", name="Rain Rank"),
    dict(id="bad_year", name="Farmers' reported Bad Years"),
]

PFX = "/fbfmaproom"
SERVER = flask.Flask(__name__)
APP = dash.Dash(
    __name__,
    server=SERVER,
    url_base_pathname=f"{PFX}/",
    meta_tags=[
        {"name": "description", "content": "content description 1234"},
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
    ],
)

APP.layout = fbflayout.app_layout(TABLE_COLUMNS)


def open_data_arrays():
    rs = {}
    bath = xr.open_dataset("bath432.nc", decode_times=False)["bath"].transpose("Y", "X")
    # bath = pingrid.correct_coord(bath, "Y")
    # bath = pingrid.correct_coord(bath, "X")

    bath = xr.where(bath < 0.0, 0.0, bath)
    rs["bath"] = pingrid.DataArrayEntry(
        "bath", bath, {(): pingrid.create_interp2d(bath, bath.dims)}, 0.0, 7964.0, None
    )
    print(bath, pingrid.extents(bath))

    rain = (
        xr.open_dataset("rain-noaa.nc", decode_times=False)["prcp_est"]
        .pipe(pingrid.roll_to, "X", 180)
        .transpose("Y", "X", ...)
    )
    rain_min = 0.0
    rain_max = rain.max()
    rs["rain"] = pingrid.DataArrayEntry(
        "rain",
        rain,
        {},
        rain_min,
        rain_max,
        pingrid.parse_colormap(rain.attrs["colormap"]),
    )
    # print("*** colormap:", rs["rain"].colormap, rs["rain"].colormap.shape)
    print(rain, pingrid.extents(rain, ["Y", "X"]))
    # print(pingrid.from_months_since_v(rain["T"].values))

    pnep = xr.open_dataset("pnep-malawi.nc", decode_times=False)["prob"].transpose(
        "Y", "X", ...
    )
    # pnep["T"] = pnep["S"] + pnep["L"]
    pnep_min = 0.0
    pnep_max = 100.0
    pnep_colormap = pingrid.parse_colormap(
        "[null 12632256 8388608 [16711680 14] [16760576 13] [15453831 14] 13959039 "
        "[13959039 13] 64636 [65535 25] [36095 26] [255 51] [128 51] [128 51] 2763429]"
    )
    rs["pnep"] = pingrid.DataArrayEntry(
        "pnep", pnep, {}, pnep_min, pnep_max, pnep_colormap
    )
    print(
        pnep,
        pingrid.extents(pnep, ["Y", "X"]),
        pnep["S"].values.shape,
        pnep["L"].values.shape,
        # pnep["T"].values.shape,
    )
    return rs


SHAPE_CONF = dict(
    District=dict(table="g2015_2014_2", label="adm2_name", geom="the_geom"),
    Regional=dict(table="g2015_2014_1", label="adm1_name", geom="the_geom"),
    National=dict(table="g2015_2012_0", label="adm0_name", geom="the_geom"),
)


def retrieve_geometry(
    dbpool, point: Tuple[float, float], mode: str, adm0_name: str
) -> MultiPolygon:
    y, x = point
    sc = SHAPE_CONF[mode]
    with dbpool.take() as cm:
        conn = cm.resource
        with conn:  # transaction
            df = pd.read_sql(
                sql.SQL(
                    """
                    with a as(
                        select gid, {geom} as the_geom,
                            ST_SetSRID(ST_MakePoint(%(x)s, %(y)s),4326) as pt,
                            adm0_name, {label} as label
                            from {table})
                    select gid, ST_AsBinary(the_geom) as the_geom, pt,
                        adm0_name, label
                        from a
                        where the_geom && pt and ST_Contains(the_geom, pt) and
                            adm0_name = %(adm0_name)s
                    """
                ).format(
                    table=sql.Identifier(sc["table"]),
                    label=sql.Identifier(sc["label"]),
                    geom=sql.Identifier(sc["geom"]),
                ),
                conn,
                params=dict(x=x, y=y, adm0_name=adm0_name),
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


DATA_ARRAYS: Dict[str, pingrid.DataArrayEntry] = open_data_arrays()

SEASON_LENGTH = 3.0

CLIPPING = {
    k: retrieve_geometry(DBPOOL, v["marker"], "National", v["adm0_name"])
    for k, v in CS.items()
}


DF = pd.read_csv("fbfmaproom.csv")
DF["year"] = DF["month"].apply(lambda x: pingrid.from_months_since(x).year)
DF["begin_year"] = DF["month"].apply(
    lambda x: pingrid.from_months_since(x - SEASON_LENGTH / 2).year
)
DF["end_year"] = DF["month"].apply(
    lambda x: pingrid.from_months_since(x + SEASON_LENGTH / 2).year
)
DF["label"] = DF.apply(
    lambda d: str(d["begin_year"])
    if d["begin_year"] == d["end_year"]
    else str(d["begin_year"]) + "/" + str(d["end_year"])[-2:],
    axis=1,
)
print(DF)


def generate_tables(config, table_columns, issue_month, season, freq, positions):
    year_min, year_max = config["seasons"][season]["year_range"]
    target_month = config["seasons"][season]["target_month"]
    freq_min, freq_max = freq

    df2 = DF[DF["adm0_name"] == config["adm0_name"]]
    df = pd.DataFrame({c["id"]: [] for c in table_columns})
    df["year"] = df2["year"]
    df["year_label"] = df2["label"]
    df["enso_state"] = df2["enso_state"]
    df["bad_year"] = df2["bad_year"].where(~df2["bad_year"].isna(), "")
    df["season"] = df2["month"]
    df = df.set_index("season")

    da = DATA_ARRAYS["rain"].data_array

    da["season"] = (
        da["T"] - target_month + SEASON_LENGTH / 2
    ) // SEASON_LENGTH * SEASON_LENGTH + target_month

    da = da.groupby("season").mean() * 90
    da = da.where(da["season"] % 12 == target_month, drop=True)

    mpolygon = MultiPolygon([Polygon([[x, y] for y, x in positions])])

    da = pingrid.average_over_trimmed(
        da, mpolygon, lon_name="X", lat_name="Y", all_touched=True
    )

    df3 = da.to_dataframe()

    df = df.join(df3, how="outer")

    df = df[(df["year"] >= year_min) & (df["year"] <= year_max)]

    df["rain_rank"] = df["prcp_est"].rank(
        method="first", na_option="keep", ascending=True
    )

    rain_rank_pct = df["prcp_est"].rank(
        method="first", na_option="keep", ascending=True, pct=True
    )
    df["rain_rank_pct"] = rain_rank_pct

    df["rain_yellow"] = (rain_rank_pct <= freq_max / 100).astype(int)
    df["rain_brown"] = (rain_rank_pct <= freq_min / 100).astype(int)

    da2 = DATA_ARRAYS["pnep"].data_array

    da2 = da2.sel(P=[freq_min, freq_max], drop=True)

    s = config["seasons"][season]["issue_months"][issue_month]
    l = config["seasons"][season]["leads"][issue_month]

    da2 = da2.where(da2["S"] % 12 == s, drop=True)
    da2 = da2.sel(L=l, drop=True)
    da2["S"] = da2["S"] + l

    da2 = pingrid.average_over_trimmed(
        da2, mpolygon, lon_name="X", lat_name="Y", all_touched=True
    )

    df4 = da2.to_dataframe().unstack()
    df4.columns = df4.columns.to_flat_index()

    df = df.join(df4, on="season", how="outer")

    df = df[(df["year"] >= year_min) & (df["year"] <= year_max)]

    df["forecast"] = df[("prob", freq_max)].apply(lambda x: f"{x:.2f}")

    pnep_max_rank_pct = df[("prob", freq_max)].rank(
        method="first", na_option="keep", ascending=True, pct=True
    )
    df["pnep_max_rank_pct"] = pnep_max_rank_pct
    df["pnep_yellow"] = (pnep_max_rank_pct <= freq_max / 100).astype(int)

    pnep_min_rank_pct = df[("prob", freq_min)].rank(
        method="first", na_option="keep", ascending=True, pct=True
    )
    df["pnep_min_rank_pct"] = pnep_min_rank_pct
    df["pnep_brown"] = (pnep_min_rank_pct <= freq_min / 100).astype(int)

    df = df[::-1]

    # df.to_csv("df.csv")

    df = df[
        [c["id"] for c in table_columns]
        + ["rain_yellow", "rain_brown", "pnep_yellow", "pnep_brown"]
    ]

    dfs = pd.DataFrame({c["id"]: [] for c in table_columns})
    dfs["year_label"] = [
        "Worthy-action:",
        "Act-in-vain:",
        "Fail-to-act:",
        "Worthy-Inaction:",
        "Rate:",
    ]
    dfs2 = pd.DataFrame({c["id"]: [c["name"]] for c in table_columns})
    dfs = dfs.append(dfs2)

    bad_year = df["bad_year"] == "Bad"
    dfs["enso_state"][:5] = hits_and_misses(df["enso_state"] == "El Niño", bad_year)
    dfs["forecast"][:5] = hits_and_misses(df["pnep_yellow"] == 1, bad_year)
    dfs["rain_rank"][:5] = hits_and_misses(df["rain_yellow"] == 1, bad_year)

    return df, dfs


def hits_and_misses(c1, c2):
    h1 = (c1 & c2).sum()
    m1 = (c1 & ~c2).sum()
    m2 = (~c1 & c2).sum()
    h2 = (~c1 & ~c2).sum()
    return [h1, m1, m2, h2, f"{(h1 + h2) / (h1 + h2 + m1 + m2) * 100:.2f}%"]


def calculate_bounds(pt, res):
    x, y = pt
    dx, dy = res
    cx = (x + dx / 2) // dx * dx
    cy = (y + dy / 2) // dy * dy
    return [[cx - dx / 2, cy - dy / 2], [cx + dx / 2, cy + dy / 2]]


def country(pathname: str) -> str:
    return pathname.split("/")[2]


@APP.callback(
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


@APP.callback(
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
            value=i,
        )
        for i, v in reversed(list(enumerate(c["issue_months"])))
    ]
    issue_month_value = len(c["issue_months"]) - 1
    return (
        year_min,
        year_max,
        year_max,
        issue_month_options,
        issue_month_value,
    )


"""
@APP.callback(
    Output("log", "children"),
    Input("map", "click_lat_lng"),
)
def _(position):
    return str(position)
"""


@APP.callback(
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
        title += ": " + str((round((x0 + x1) / 2, 2), round((y0 + y1) / 2, 2)))
    else:
        geom, attrs = retrieve_geometry(DBPOOL, position, mode, c["adm0_name"])
        print("*** geom geom: ", attrs)
        if geom is not None:
            xs, ys = geom[-1].exterior.coords.xy
            positions = list(zip(ys, xs))
            title += ": " + attrs["label"]
            content = str(
                dict(marker=(round(position[1], 2), round(position[0], 2))) | attrs
            )
    if positions is None:
        raise PreventUpdate
    return positions, [html.H2(title), html.P(content)]


@APP.callback(
    Output("table", "data"),
    Output("summary", "data"),
    Input("issue_month", "value"),
    Input("freq", "value"),
    Input("feature", "positions"),
    Input("location", "pathname"),
    State("season", "value"),
)
def _(issue_month, freq, positions, pathname, season):
    print("*** callback table:", issue_month, season, freq, len(positions), pathname)
    c = CS[country(pathname)]
    dft, dfs = generate_tables(c, TABLE_COLUMNS, issue_month, season, freq, positions)
    return dft.to_dict("records"), dfs.to_dict("records")


def slp(country, season, year, issue_month, freq_max):
    season_config = CS[country]["seasons"][season]
    l = season_config["leads"][issue_month]
    s = (
        pingrid.to_months_since(datetime.date(year, 1, 1))
        + season_config["target_month"]
        - l
    )
    p = freq_max
    return s, l, p


@APP.callback(
    Output("pne_layer", "url"),
    Input("year", "value"),
    Input("issue_month", "value"),
    Input("freq", "value"),
    Input("location", "pathname"),
    State("season", "value"),
)
def _(year, issue_month, freq, pathname, season):
    print("*** callback pne_layer:", year, issue_month, season, freq, pathname)
    country_name = country(pathname)
    _, freq_max = freq

    dae = DATA_ARRAYS["pnep"]
    s, l, p = slp(country_name, season, year, issue_month, freq_max)
    da = dae.data_array
    da = da.sel(S=s, L=l, P=p, drop=True)
    if (s, l, p) not in dae.interp2d:
        dae.interp2d[(s, l, p)] = pingrid.create_interp2d(da, da.dims)
        print("*** MISS: ", (s, l, p), len(dae.interp2d))

    return f"/pnep_tiles/{{z}}/{{x}}/{{y}}/{country_name}/{season}/{year}/{issue_month}/{freq_max}"


# Endpoints


@SERVER.route(
    f"/pnep_tiles/<int:tz>/<int:tx>/<int:ty>/<country>/<season>/<int:year>/<int:issue_month>/<int:freq_max>"
)
def tiles(tz, tx, ty, country, season, year, issue_month, freq_max):
    dae = DATA_ARRAYS["pnep"]
    s, l, p = slp(country, season, year, issue_month, freq_max)

    z = pingrid.produce_data_tile(dae.interp2d[(s, l, p)], tx, ty, tz, 256, 256)
    im = cv2.flip((z - dae.min_val) * 255 / (dae.max_val - dae.min_val), 0)

    im = pingrid.apply_colormap(im, dae.colormap)

    country_shape, country_attrs = CLIPPING[country]
    draw_attrs = pingrid.DrawAttrs(
        BGRA(255, 0, 0, 255), BGRA(0, 0, 0, 0), 1, cv2.LINE_4
    )
    shapes = [(country_shape, draw_attrs)]
    im = pingrid.produce_shape_tile(
        im, shapes, tx, ty, tz, oper="difference", tile_width=256, tile_height=256
    )

    im = pingrid.produce_test_tile(im, f"{tz}x{tx},{ty}")

    # cv2.imwrite(
    #    f"tiles/{tx},{ty}x{tz}.png",
    #    cv2.LUT(im.astype(np.uint8), np.fromiter(range(255, -1, -1), np.uint8)),
    # )

    cv2_imencode_success, buffer = cv2.imencode(".png", im)
    assert cv2_imencode_success
    io_buf = io.BytesIO(buffer)
    resp = flask.send_file(io_buf, mimetype="image/png")
    resp.headers["Cache-Control"] = "private, max-age=0, no-cache, no-store"
    return resp


if __name__ == "__main__":
    APP.run_server()
