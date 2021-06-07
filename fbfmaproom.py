from typing import Any, Dict, Tuple, Optional, Union
import os
import threading
import time
import io
from functools import lru_cache
import datetime
import urllib.parse
import yaml
import json
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
from shapely.geometry import Polygon, Point
from shapely.geometry.multipoint import MultiPoint
from psycopg2 import sql

import __about__ as about
import pyaconf
import pingrid
from pingrid import BGRA
import fbflayout


CONFIG = pyaconf.load(os.environ["CONFIG"])

DBPOOL = pingrid.init_dbpool("dbpool", CONFIG)

TABLE_COLUMNS = [
    dict(id="year_label", name="Year"),
    dict(id="enso_state", name="ENSO State"),
    dict(id="forecast", name="Forecast, %"),
    dict(id="rain_rank", name="Rain Rank"),
    dict(id="bad_year", name="Reported Bad Years"),
]

ZERO_SHAPE = [[[[0, 0], [0, 0], [0, 0], [0, 0]]]]

PFX = CONFIG["core_path"]
TILE_PFX = CONFIG["tile_path"]
ADMIN_PFX = CONFIG["admin_path"]

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
APP.title = "FBF--Maproom"

APP.layout = fbflayout.app_layout(TABLE_COLUMNS)


def open_data_array(
    config,
    country_key,
    dataset_key,
    var_key,
    val_min=None,
    val_max=None,
    reverse_colormap=False,
):
    cfg = config["countries"][country_key]["datasets"][dataset_key]
    if var_key is not None:
        ns = cfg["var_names"]
        da = xr.open_zarr(cfg["path"], decode_times=False)[ns[var_key]].transpose(
            ns["lat"], ns["lon"], ...
        )
    else:
        da = None
    if val_min is None:
        if "range" in cfg:
            val_min = cfg["range"][0]
        else:
            val_min = da.min().item()
    if val_max is None:
        if "range" in cfg:
            val_max = cfg["range"][1]
        else:
            val_max = da.max().item()
    colormap = pingrid.parse_colormap(cfg["colormap"])
    if reverse_colormap:
        colormap = colormap[::-1]
    # print("*** colormap:", dataset_key, colormap.shape)
    e = pingrid.DataArrayEntry(dataset_key, da, None, val_min, val_max, colormap)
    return e


@lru_cache
def open_vuln(country_key):
    return open_data_array(
        CONFIG,
        country_key,
        "vuln",
        None,
        val_min=None,
        val_max=None,
        reverse_colormap=False,
    )


@lru_cache
def open_pnep(country_key):
    return open_data_array(
        CONFIG,
        country_key,
        "pnep",
        "pnep",
        val_min=0.0,
        val_max=100.0,
        reverse_colormap=False,
    )


@lru_cache
def open_rain(country_key):
    return open_data_array(
        CONFIG, country_key, "rain", "rain", val_min=0.0, val_max=1000.0
    )


def slp(country_key, season, year, issue_month, freq_max):
    season_config = CONFIG["countries"][country_key]["seasons"][season]
    issue_month = season_config["issue_months"][issue_month]
    target_month = season_config["target_month"]

    l = (target_month - issue_month) % 12

    s = pingrid.to_months_since(datetime.date(year, 1, 1)) + target_month - l
    p = freq_max
    return s, l, p


@lru_cache
def select_pnep(country_key, season, year, issue_month, freq_max):
    config = CONFIG
    s, l, p = slp(country_key, season, year, issue_month, freq_max)
    ns = config["countries"][country_key]["datasets"]["pnep"]["var_names"]
    e = open_pnep(country_key)
    da = e.data_array
    da = da.sel({ns["issue"]: s, ns["pct"]: p}, drop=True)
    if ns["lead"] is not None:
        da = da.sel({ns["lead"]: l}, drop=True)
    interp = pingrid.create_interp(da, da.dims)
    dae = pingrid.DataArrayEntry(e.name, da, interp, e.min_val, e.max_val, e.colormap)
    return dae


@lru_cache
def select_rain(country_key, year, season):
    config = CONFIG["countries"][country_key]
    ns = config["datasets"]["rain"]["var_names"]
    season_config = config["seasons"][season]
    season_length = season_config["length"]
    target_month = season_config["target_month"]
    e = open_rain(country_key)
    t = pingrid.to_months_since(datetime.date(year, 1, 1)) + target_month
    da = e.data_array * season_length * 30.0
    da = da.sel({ns["time"]: t}, drop=True).fillna(0.0)
    interp = pingrid.create_interp(da, da.dims)
    dae = pingrid.DataArrayEntry(
        e.name, e.data_array, interp, e.min_val, e.max_val, e.colormap
    )
    return dae


@lru_cache
def open_enso(season_length):
    config = CONFIG
    dbpool = DBPOOL
    dc = config["dataframes"]["enso"]
    with dbpool.take() as cm:
        conn = cm.resource
        with conn:  # transaction
            df = pd.read_sql(
                sql.SQL(
                    """
                    select lower(adm0_name) as country_key, month_since_01011960,
                        enso_state, bad_year 
                    from {schema}.{table}
                    """
                ).format(
                    schema=sql.Identifier(dc["schema"]),
                    table=sql.Identifier(dc["table"]),
                ),
                conn,
            )

    df["year"] = df["month_since_01011960"].apply(
        lambda x: pingrid.from_months_since(x).year
    )
    df["begin_year"] = df["month_since_01011960"].apply(
        lambda x: pingrid.from_months_since(x - season_length / 2).year
    )
    df["end_year"] = df["month_since_01011960"].apply(
        lambda x: pingrid.from_months_since(x + season_length / 2).year
    )
    df["label"] = df.apply(
        lambda d: str(d["begin_year"])
        if d["begin_year"] == d["end_year"]
        else str(d["begin_year"]) + "/" + str(d["end_year"])[-2:],
        axis=1,
    )
    return df


def retrieve_geometry(
    country_key: str, point: Tuple[float, float], mode: str, year: Optional[int]
) -> Tuple[MultiPolygon, Dict[str, Any]]:
    df = retrieve_vulnerability(country_key, mode, year)
    x, y = point
    p = Point(x, y)
    geom, attrs = None, None
    for _, r in df.iterrows():
        minx, miny, maxx, maxy = r["the_geom"].bounds
        if minx <= x <= maxx and miny <= y <= maxy and r["the_geom"].contains(p):
            geom = r["the_geom"]
            attrs = {k: v for k, v in r.items() if k not in ("the_geom")}
            break
    return geom, attrs


@lru_cache
def retrieve_vulnerability(
    country_key: str, mode: str, year: Optional[int]
) -> pd.DataFrame:
    config = CONFIG["countries"][country_key]
    sc = config["shapes"][int(mode)]
    dbpool = DBPOOL
    with dbpool.take() as cm:
        conn = cm.resource
        with conn:  # transaction
            s = sql.Composed(
                [
                    sql.SQL("with v as ("),
                    sql.SQL(sc["vuln_sql"]),
                    sql.SQL("), g as ("),
                    sql.SQL(sc["sql"]),
                    sql.SQL(
                        """
                        ), a as (
                            select
                                key,
                                avg(vuln) as mean,
                                stddev_pop(vuln) as stddev
                            from v
                            group by key
                        )
                        select
                            g.label, g.key, g.the_geom,
                            v.year,
                            v.vuln as vulnerability,
                            a.mean as mean,
                            a.stddev as stddev,
                            v.vuln / a.mean as normalized,
                            coalesce(to_char(v.vuln,'999,999,999,999'),'N/A') as "Vulnerability",
                            coalesce(to_char(a.mean,'999,999,999,999'),'N/A') as "Mean",
                            coalesce(to_char(a.stddev,'999,999,999,999'),'N/A') as "Stddev",
                            coalesce(to_char(v.vuln / a.mean,'999,990D999'),'N/A') as "Normalized"
                        from (g left outer join a using (key))
                            left outer join v on(g.key=v.key and v.year=%(year)s)
                        """
                    ),
                ]
            )
            # print(s.as_string(conn))
            df = pd.read_sql(
                s,
                conn,
                params=dict(year=year),
            )
    # print("bytes: ", sum(df.the_geom.apply(lambda x: len(x.tobytes()))))
    df["the_geom"] = df["the_geom"].apply(lambda x: wkb.loads(x.tobytes()))
    df["the_geom"] = df["the_geom"].apply(
        lambda x: x if isinstance(x, MultiPolygon) else MultiPolygon([x])
    )
    return df


def generate_tables(
    country_key,
    config,
    table_columns,
    issue_month,
    season,
    freq,
    positions,
):
    df = pd.DataFrame({c["id"]: [] for c in table_columns})

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

    if positions == ZERO_SHAPE:
        return df, dfs

    season_config = config["seasons"][season]
    year_min, year_max = season_config["year_range"]
    season_length = season_config["length"]
    target_month = season_config["target_month"]
    freq_min, freq_max = freq

    df2 = open_enso(season_length)
    df2 = df2[df2["country_key"] == country_key]

    df["year"] = df2["year"]
    df["year_label"] = df2["label"]
    df["enso_state"] = df2["enso_state"]
    df["bad_year"] = df2["bad_year"].where(~df2["bad_year"].isna(), "")
    df["season"] = df2["month_since_01011960"]
    df = df.set_index("season")

    da = open_rain(country_key).data_array * season_length * 30
    ns = config["datasets"]["rain"]["var_names"]

    mpolygon = pingrid.mpoly_leaflet_to_shapely(positions)

    da = pingrid.average_over_trimmed(
        da, mpolygon, lon_name=ns["lon"], lat_name=ns["lat"], all_touched=True
    )

    df3 = da.to_dataframe()

    df = df.join(df3, how="outer")

    df = df[(df["year"] >= year_min) & (df["year"] <= year_max)]

    df["rain_rank"] = df[ns["rain"]].rank(
        method="first", na_option="keep", ascending=True
    )

    rain_rank_pct = df[ns["rain"]].rank(
        method="first", na_option="keep", ascending=True, pct=True
    )
    df["rain_rank_pct"] = rain_rank_pct

    df["rain_yellow"] = (rain_rank_pct <= freq_max / 100).astype(int)
    df["rain_brown"] = (rain_rank_pct <= freq_min / 100).astype(int)

    da2 = open_pnep(country_key).data_array
    ns = config["datasets"]["pnep"]["var_names"]

    da2 = da2.sel({ns["pct"]: [freq_min, freq_max]}, drop=True)

    s = config["seasons"][season]["issue_months"][issue_month]
    l = (target_month - s) % 12

    da2 = da2.where(da2[ns["issue"]] % 12 == s, drop=True)
    if ns["lead"] is not None:
        da2 = da2.sel({ns["lead"]: l}, drop=True)
    da2[ns["issue"]] = da2[ns["issue"]] + l

    da2 = pingrid.average_over_trimmed(
        da2, mpolygon, lon_name=ns["lon"], lat_name=ns["lat"], all_touched=True
    )

    df4 = da2.to_dataframe().unstack()
    df4.columns = df4.columns.to_flat_index()

    df = df.join(df4, how="outer")

    df = df[(df["year"] >= year_min) & (df["year"] <= year_max)]

    df["forecast"] = df[(ns["pnep"], freq_max)].apply(lambda x: f"{x:.2f}")

    pnep_max_rank_pct = df[(ns["pnep"], freq_max)].rank(
        method="first", na_option="keep", ascending=False, pct=True
    )
    df["pnep_max_rank_pct"] = pnep_max_rank_pct
    df["pnep_yellow"] = (pnep_max_rank_pct <= freq_max / 100).astype(int)

    pnep_min_rank_pct = df[(ns["pnep"], freq_min)].rank(
        method="first", na_option="keep", ascending=False, pct=True
    )
    df["pnep_min_rank_pct"] = pnep_min_rank_pct
    df["pnep_brown"] = (pnep_min_rank_pct <= freq_min / 100).astype(int)

    df = df[::-1]

    # df.to_csv("df.csv")

    df = df[
        [c["id"] for c in table_columns]
        + ["rain_yellow", "rain_brown", "pnep_yellow", "pnep_brown"]
    ]

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


def calculate_bounds(pt, res, origin):
    x, y = pt
    dx, dy = res
    x0, y0 = origin
    cx = (x - x0 + dx / 2) // dx * dx + x0
    cy = (y - y0 + dy / 2) // dy * dy + y0
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
    Output("pnep_colorbar", "colorscale"),
    Output("vuln_colorbar", "colorscale"),
    Output("mode", "options"),
    Output("mode", "value"),
    Input("location", "pathname"),
)
def _(pathname):
    country_key = country(pathname)
    c = CONFIG["countries"][country_key]
    season_options = [
        dict(
            label=c["seasons"][k]["label"],
            value=k,
        )
        for k in sorted(c["seasons"].keys())
    ]
    season_value = min(c["seasons"].keys())
    x, y = c["marker"]
    cx, cy = c["center"]
    pnep_cs = pingrid.to_dash_colorscale(open_pnep(country_key).colormap)
    vuln_cs = pingrid.to_dash_colorscale(open_vuln(country_key).colormap)
    mode_options = [
        dict(
            label=k["name"],
            value=str(i),
        )
        for i, k in enumerate(c["shapes"])
    ] + [dict(label="Pixel", value="pixel")]
    mode_value = "0"
    return (
        f"{PFX}/assets/{c['logo']}",
        [cy, cx],
        c["zoom"],
        [y, x],
        season_options,
        season_value,
        pnep_cs,
        vuln_cs,
        mode_options,
        mode_value,
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
    country_key = country(pathname)
    c = CONFIG["countries"][country_key]["seasons"][season]
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
    Output("geom_key", "value"),
    Output("marker_popup", "children"),
    Input("location", "pathname"),
    Input("marker", "position"),
    Input("mode", "value"),
    Input("year", "value"),
)
def _(pathname, position, mode, year):
    country_key = country(pathname)
    y, x = position
    c = CONFIG["countries"][country_key]
    title = "No Data"
    content = []
    positions = None
    if mode == "pixel":
        (x0, y0), (x1, y1) = calculate_bounds(
            (x, y), c["resolution"], c.get("origin", (0, 0))
        )
        pixel = MultiPoint([(x0, y0), (x1, y1)]).envelope
        geom, _ = retrieve_geometry(country_key, tuple(c["marker"]), "0", None)
        if pixel.intersects(geom):
            positions = [[[[y0, x0], [y1, x0], [y1, x1], [y0, x1], [y0, x0]]]]
            px = (x0 + x1) / 2
            pxs = "E" if px > 0.0 else "W" if px < 0.0 else ""
            py = (y0 + y1) / 2
            pys = "N" if py > 0.0 else "S" if py < 0.0 else ""
            title = f"{np.abs(py):.5f}° {pys} {np.abs(px):.5f}° {pxs}"
        key = None
    else:
        geom, attrs = retrieve_geometry(country_key, (x, y), mode, year)
        if geom is not None:
            positions = pingrid.mpoly_shapely_to_leaflet(geom)
            title = attrs["label"]
            fmt = lambda k: [html.B(k + ": "), attrs[k], html.Br()]
            content = (
                fmt("Vulnerability") + fmt("Mean") + fmt("Stddev") + fmt("Normalized")
            )
        key = attrs["key"]
    if positions is None:
        # raise PreventUpdate
        positions = ZERO_SHAPE
    return positions, key, [html.H3(title), html.Div(content)]


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
    country_key = country(pathname)
    config = CONFIG["countries"][country_key]
    dft, dfs = generate_tables(
        country_key,
        config,
        TABLE_COLUMNS,
        issue_month,
        season,
        freq,
        positions,
    )
    return dft.to_dict("records"), dfs.to_dict("records")


@APP.callback(
    Output("gantt", "href"),
    Input("issue_month", "value"),
    Input("freq", "value"),
    Input("feature", "positions"),
    Input("geom_key", "value"),
    Input("mode", "value"),
    Input("year", "value"),
    Input("location", "pathname"),
    State("season", "value"),
)
def _(issue_month, freq, positions, geom_key, mode, year, pathname, season):
    country_key = country(pathname)
    config = CONFIG["countries"][country_key]
    season_config = config["seasons"][season]
    if mode == "pixel":
        region = None
    else:
        label, _ = retrieve_geometry2(country_key, int(mode), geom_key)
        region = {
            "id": str(geom_key),
            "label": label,
        }
    res = dict(
        country=country_key,
        mode=mode,
        issue_year=year,
        freq=freq[0],  # [0] is temporary until the whole app is switched to single slider
        season={
            "id": season,
            "label": season_config["label"],
            "target_month": season_config["target_month"],
            "length": season_config["length"]
        },
        issue_month=config["seasons"][season]["issue_months"][issue_month],
        bounds=None,
        region=region,
    )
    # print("***:", res)
    url = CONFIG["gantt_url"] + urllib.parse.urlencode(dict(data=json.dumps(res)))
    return url


@APP.callback(
    Output("pnep_layer", "url"),
    Input("year", "value"),
    Input("issue_month", "value"),
    Input("freq", "value"),
    Input("location", "pathname"),
    State("season", "value"),
)
def _(year, issue_month, freq, pathname, season):
    country_key = country(pathname)
    _, freq_max = freq
    select_pnep(country_key, season, year, issue_month, freq_max)
    return f"{TILE_PFX}/pnep/{{z}}/{{x}}/{{y}}/{country_key}/{season}/{year}/{issue_month}/{freq_max}"


@APP.callback(
    Output("vuln_layer", "url"),
    Input("year", "value"),
    Input("location", "pathname"),
    Input("mode", "value"),
)
def _(year, pathname, mode):
    country_key = country(pathname)
    if mode != "pixel":
        retrieve_vulnerability(country_key, mode, year)
    return f"{TILE_PFX}/vuln/{{z}}/{{x}}/{{y}}/{country_key}/{mode}/{year}"


# Endpoints


def image_resp(im):
    cv2_imencode_success, buffer = cv2.imencode(".png", im)
    assert cv2_imencode_success
    io_buf = io.BytesIO(buffer)
    resp = flask.send_file(io_buf, mimetype="image/png")
    return resp


def yaml_resp(data):
    s = yaml.dump(data, default_flow_style=False, width=120, allow_unicode=True)
    resp = flask.Response(response=s, mimetype="text/x-yaml")
    resp.headers["Cache-Control"] = "private, max-age=0, no-cache, no-store"
    return resp


def tile(dae, tx, ty, tz, clipping=None, test_tile=False):
    z = pingrid.produce_data_tile(dae.interp, tx, ty, tz, 256, 256)
    im = (z - dae.min_val) * 255 / (dae.max_val - dae.min_val)
    im = pingrid.apply_colormap(im, dae.colormap)
    if clipping is not None:
        country_shape, _ = clipping
        draw_attrs = pingrid.DrawAttrs(
            BGRA(0, 0, 255, 255), BGRA(0, 0, 0, 0), 1, cv2.LINE_AA
        )
        shapes = [(country_shape, draw_attrs)]
        im = pingrid.produce_shape_tile(im, shapes, tx, ty, tz, oper="difference")
    if test_tile:
        im = pingrid.produce_test_tile(im, f"{tz}x{tx},{ty}")
    return image_resp(im)


@SERVER.route(
    f"{TILE_PFX}/pnep/<int:tz>/<int:tx>/<int:ty>/<country_key>/<season>/<int:year>/<int:issue_month>/<int:freq_max>"
)
def pnep_tiles(tz, tx, ty, country_key, season, year, issue_month, freq_max):
    dae = select_pnep(country_key, season, year, issue_month, freq_max)
    p = tuple(CONFIG["countries"][country_key]["marker"])
    clipping = retrieve_geometry(country_key, p, "0", None)
    resp = tile(dae, tx, ty, tz, clipping)
    return resp


@SERVER.route(
    f"{TILE_PFX}/vuln/<int:tz>/<int:tx>/<int:ty>/<country_key>/<mode>/<int:year>"
)
def vuln_tiles(tz, tx, ty, country_key, mode, year):
    im = pingrid.produce_bkg_tile(BGRA(0, 0, 0, 0), 256, 256)
    e = open_vuln(country_key)
    if mode != "pixel":
        df = retrieve_vulnerability(country_key, mode, year)
        shapes = [
            (
                r["the_geom"],
                pingrid.DrawAttrs(
                    BGRA(0, 0, 255, 255),
                    pingrid.with_alpha(
                        e.colormap[
                            min(
                                255,
                                int(
                                    (r["normalized"] - e.min_val)
                                    * 255
                                    / (e.max_val - e.min_val)
                                ),
                            )
                        ],
                        255,
                    )
                    if r["normalized"] is not None and not np.isnan(r["normalized"])
                    else BGRA(0, 0, 0, 0),
                    1,
                    cv2.LINE_AA,
                ),
            )
            for _, r in df.iterrows()
        ]
        im = pingrid.produce_shape_tile(im, shapes, tx, ty, tz, oper="intersection")
    return image_resp(im)


def cache_stats(f):
    cs = {}
    # cs |= f.cache_parameters()
    cs |= f.cache_info()._asdict()
    return {f.__name__: cs}


@SERVER.route(f"{ADMIN_PFX}/stats")
def stats():
    fs = [
        open_pnep,
        select_pnep,
        open_rain,
        select_rain,
        open_vuln,
        open_enso,
        retrieve_vulnerability,
    ]
    cs = {}
    for f in fs:
        cs |= cache_stats(f)

    ps = dict(
        pid=os.getpid(),
        active_count=threading.active_count(),
        current_thread_name=threading.current_thread().name,
        ident=threading.get_ident(),
        main_thread_ident=threading.main_thread().ident,
        stack_size=threading.stack_size(),
        threads={
            x.ident: dict(name=x.name, is_alive=x.is_alive(), is_daemon=x.daemon)
            for x in threading.enumerate()
        },
    )

    rs = dict(
        version=about.version,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        cache_stats=cs,
        process_stats=ps,
    )
    return yaml_resp(rs)


class InvalidRequest(Exception):
    def __init__(self, message):
        super().__init__()
        self.message = message

    def to_dict(self):
        return {"message": self.message}


@SERVER.errorhandler(InvalidRequest)
def invalid_api_usage(e):
    return flask.json.jsonify(e.to_dict()), 400


def parse_arg(name, conversion=str, required=True, default=None, multiple=False):
    assert not (multiple and default is not None)
    assert not (required and default is not None)

    vals = flask.request.args.getlist(name)
    if len(vals) == 0:
        if required:
            raise InvalidRequest(f"{name} is required")
        else:
            vals = [default]
    if len(vals) > 1 and not multiple:
        raise InvalidRequest(f"{name} must be provided only once")
    try:
        vals = list(map(conversion, vals))
    except Exception as e:
        raise InvalidRequest(f"{name} must be interpretable as {conversion}") from e
    if multiple:
        return vals
    else:
        return vals[0]


@SERVER.route("/pnep_percentile")
def pnep_percentile():
    """Let P(y) be the forecast probability of not exceeding the /freq/ percentile in year y.
    Let r be the rank of P(issue_year) among all the P(y).
    Returns r divided by the number of forecast years times 100,
    unless the forecast for issue_year is not yet available in which case it returns null."""
    # TODO better explanation

    country_key = parse_arg("country_key")
    mode = parse_arg("mode", int)
    season = parse_arg("season")
    issue_month = parse_arg("issue_month", int)
    issue_year = parse_arg("issue_year", int)
    freq = parse_arg("freq", float)
    bounds = parse_arg("bounds", required=False)
    region = parse_arg("region", required=False)

    if bounds is None and region is None:
        raise InvalidRequest("Either bounds or region must be provided")

    config = CONFIG["countries"][country_key]
    season_config = config["seasons"][season]
    ns = config["datasets"]["pnep"]["var_names"]

    s = (issue_year - 1960) * 12 + issue_month
    l = (season_config["target_month"] - issue_month) % 12

    pnep = open_pnep(country_key).data_array

    percentile = None
    if s in pnep[ns["issue"]]:
        available = True
        _, geom = retrieve_geometry2(country_key, mode, region)

        pnep = pnep.sel({ns["pct"]: freq}, drop=True)
        pnep = pnep.where(pnep[ns["issue"]] % 12 == issue_month, drop=True)

        if ns["lead"] is not None:
            pnep = pnep.sel({ns["lead"]: l}, drop=True)

        # TODO why do we do this in the original?
        # da2[ns["issue"]] = da2[ns["issue"]] + l

        pnep = pingrid.average_over_trimmed(
            pnep, geom, lon_name=ns["lon"], lat_name=ns["lat"], all_touched=True
        )

        selected_value = pnep.sel({ns["issue"]: s}, drop=True)
        rank = (pnep >= selected_value).sum().values
        percentile = rank / pnep[ns["issue"]].shape[0] * 100

    return {"percentile": percentile}


def retrieve_geometry2(country_key: str, mode: int, region_key: str):
    config = CONFIG["countries"][country_key]
    sc = config["shapes"][mode]
    with DBPOOL.take() as cm:
        conn = cm.resource
        with conn:  # transaction
            s = sql.SQL(sc["sql2"])
            df = pd.read_sql(s, conn, params=parse_key(region_key))
    if len(df) == 0:
        raise InvalidRequest(f"invalid region {region_key}")
    assert len(df) == 1
    geom = df["the_geom"].iloc[0]
    geom = wkb.loads(geom.tobytes())
    return df["label"].iloc[0], geom


def parse_key(s):
    if s[0] == "(":
        assert s[-1] == ")"
        s = s[1:-1]
        s = s.split(",")
    else:
        s = [s]
    return s


if __name__ == "__main__":
    APP.run_server(
        debug=False if CONFIG["mode"] == "prod" else True,
    )
