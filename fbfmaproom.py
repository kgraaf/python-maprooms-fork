from typing import Any, Dict, Tuple, List, Literal, Optional, Union, Callable, Hashable
import os
import time
import io
from functools import lru_cache
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
from shapely.geometry import Polygon, Point
from shapely.geometry.multipoint import MultiPoint
from psycopg2 import sql
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


def open_data_array(
    config, country_key, dataset_key, var_key, val_min=None, val_max=None
):
    cfg = config["countries"][country_key]["datasets"][dataset_key]
    ns = cfg["var_names"]
    da = xr.open_dataset(cfg["path"], decode_times=False)[ns[var_key]].transpose(
        ns["lat"], ns["lon"], ...
    )
    if val_min is None:
        val_min = da.min().item()
    if val_max is None:
        val_max = da.max().item()
    colormap = pingrid.parse_colormap(cfg["colormap"])
    e = pingrid.DataArrayEntry(dataset_key, da, None, val_min, val_max, colormap)
    return e


@lru_cache
def open_pnep(country_key):
    return open_data_array(
        CONFIG, country_key, "pnep", "pnep", val_min=0.0, val_max=100.0
    )


@lru_cache
def open_rain(country_key):
    return open_data_array(
        CONFIG, country_key, "rain", "rain", val_min=0.0, val_max=1000.0
    )


def slp(country_key, season, year, issue_month, freq_max):
    season_config = CONFIG["countries"][country_key]["seasons"][season]
    l = season_config["leads"][issue_month]
    s = (
        pingrid.to_months_since(datetime.date(year, 1, 1))
        + season_config["target_month"]
        - l
    )
    p = freq_max
    return s, l, p


@lru_cache
def select_pnep(country_key, season, year, issue_month, freq_max):
    config = CONFIG
    s, l, p = slp(country_key, season, year, issue_month, freq_max)
    ns = config["countries"][country_key]["datasets"]["pnep"]["var_names"]
    e = open_pnep(country_key)
    da = e.data_array
    da = da.sel({ns["issue"]: s, ns["lead"]: l, ns["pct"]: p}, drop=True)
    interp2d = pingrid.create_interp2d(da, da.dims)
    dae = pingrid.DataArrayEntry(
        e.name, e.data_array, interp2d, e.min_val, e.max_val, e.colormap
    )
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
    da = e.data_array
    da = seasonal_average(da, ns, target_month, season_length) * season_length * 30.0
    da = da.sel({"season": t}, drop=True).fillna(0.0)
    interp2d = pingrid.create_interp2d(da, da.dims)
    dae = pingrid.DataArrayEntry(
        e.name, e.data_array, interp2d, e.min_val, e.max_val, e.colormap
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
                    select adm0_code, adm0_name, month_since_01011960,
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
    config = CONFIG
    dbpool = DBPOOL
    adm0_name = config["countries"][country_key]["adm0_name"]
    sc = config["shapes"][mode]
    dc = config["dataframes"]["vuln"]
    with dbpool.take() as cm:
        conn = cm.resource
        with conn:  # transaction
            s = sql.SQL(
                """
                with v as (
                    select
                        ({key}) as key,
                        year,
                        sum(vulnerability) as vulnerability
                    from {vuln_schema}.{vuln_table}
                    where adm0_name = %(adm0_name)s
                    group by {label}, ({key}), year
                ),
                g as (
                    select
                        ({key}) as key,
                        {label} as label,
                        ST_AsBinary({geom}) as the_geom
                    from {geom_schema}.{geom_table}
                    where adm0_name = %(adm0_name)s
                ),
                a as (
                    select
                        key,
                        avg(vulnerability) as mean,
                        stddev_pop(vulnerability) as stddev
                    from v
                    group by key
                )
                select
                    g.label, g.key, g.the_geom,
                    v.year,
                    v.vulnerability,
                    a.mean as mean,
                    a.stddev as stddev,
                    v.vulnerability / a.mean as normalized,
                    coalesce(to_char(v.vulnerability,'999,999,999,999'),'N/A') as "Vulnerability",
                    coalesce(to_char(a.mean,'999,999,999,999'),'N/A') as "Mean",
                    coalesce(to_char(a.stddev,'999,999,999,999'),'N/A') as "Stddev",
                    coalesce(to_char(v.vulnerability / a.mean,'999,990D999'),'N/A') as "Normalized"
                from (g left outer join a using (key))
                    left outer join v on(g.key=v.key and v.year=%(year)s)
                """
            ).format(
                geom_schema=sql.Identifier(sc["schema"]),
                geom_table=sql.Identifier(sc["table"]),
                vuln_schema=sql.Identifier(dc["schema"]),
                vuln_table=sql.Identifier(dc["table"]),
                key=pingrid.sql_key(sc["key"]),
                geom=sql.Identifier(sc["geom"]),
                label=sql.Identifier(sc["label"]),
            )
            # print(s.as_string(conn))
            df = pd.read_sql(
                s,
                conn,
                params=dict(year=year, adm0_name=adm0_name),
            )
    # print("bytes: ", sum(df.the_geom.apply(lambda x: len(x.tobytes()))))
    df["the_geom"] = df["the_geom"].apply(lambda x: wkb.loads(x.tobytes()))
    df["the_geom"] = df["the_geom"].apply(
        lambda x: x if isinstance(x, MultiPolygon) else MultiPolygon([x])
    )
    return df


def seasonal_average(da, ns, target_month, season_length):
    da["season"] = (
        da[ns["time"]] - target_month + season_length / 2
    ) // season_length * season_length + target_month
    da = da.groupby("season").mean()
    da = da.where(da["season"] % 12 == target_month, drop=True)
    return da


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

    if positions == [[0, 0], [0, 0], [0, 0], [0, 0]]:
        return df, dfs

    season_config = config["seasons"][season]
    year_min, year_max = season_config["year_range"]
    season_length = season_config["length"]
    target_month = season_config["target_month"]
    freq_min, freq_max = freq

    df2 = open_enso(season_length)
    df2 = df2[df2["adm0_name"] == config["adm0_name"]]

    df["year"] = df2["year"]
    df["year_label"] = df2["label"]
    df["enso_state"] = df2["enso_state"]
    df["bad_year"] = df2["bad_year"].where(~df2["bad_year"].isna(), "")
    df["season"] = df2["month_since_01011960"]
    df = df.set_index("season")

    da = open_rain(country_key).data_array
    ns = config["datasets"]["rain"]["var_names"]
    da = seasonal_average(da, ns, target_month, season_length) * season_length * 30.0

    mpolygon = MultiPolygon([Polygon([[x, y] for y, x in positions])])

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
    l = config["seasons"][season]["leads"][issue_month]

    da2 = da2.where(da2[ns["issue"]] % 12 == s, drop=True)
    da2 = da2.sel({ns["lead"]: l}, drop=True)
    da2[ns["issue"]] = da2[ns["issue"]] + l

    da2 = pingrid.average_over_trimmed(
        da2, mpolygon, lon_name=ns["lon"], lat_name=ns["lat"], all_touched=True
    )

    df4 = da2.to_dataframe().unstack()
    df4.columns = df4.columns.to_flat_index()

    df = df.join(df4, on="season", how="outer")

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
    return (
        f"{PFX}/assets/{c['logo']}",
        [cy, cx],
        c["zoom"],
        [y, x],
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
        (x0, y0), (x1, y1) = calculate_bounds((x, y), c["resolution"])
        pixel = MultiPoint([(x0, y0), (x1, y1)]).envelope
        geom, _ = retrieve_geometry(country_key, tuple(c["marker"]), "national", None)
        if pixel.intersects(geom):
            positions = [[y0, x0], [y1, x0], [y1, x1], [y0, x1], [y0, x0]]
            px = (x0 + x1) / 2
            pxs = "E" if px > 0.0 else "W" if px < 0.0 else ""
            py = (y0 + y1) / 2
            pys = "N" if py > 0.0 else "S" if py < 0.0 else ""
            title = f"{np.abs(py):.5f}° {pys} {np.abs(px):.5f}° {pxs}"
    else:
        geom, attrs = retrieve_geometry(country_key, (x, y), mode, year)
        if geom is not None:
            xs, ys = geom[-1].exterior.coords.xy
            positions = list(zip(ys, xs))
            title = attrs["label"]
            fmt = lambda k: [html.B(k + ": "), attrs[k], html.Br()]
            content = (
                fmt("Vulnerability") + fmt("Mean") + fmt("Stddev") + fmt("Normalized")
            )
    if positions is None:
        # raise PreventUpdate
        positions = [[0, 0], [0, 0], [0, 0], [0, 0]]
    return positions, [html.H3(title), html.Div(content)]


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
    return f"/pnep_tiles/{{z}}/{{x}}/{{y}}/{country_key}/{season}/{year}/{issue_month}/{freq_max}"


@APP.callback(
    Output("rain_layer", "url"),
    Input("year", "value"),
    Input("location", "pathname"),
    State("season", "value"),
)
def _(year, pathname, season):
    country_key = country(pathname)
    select_rain(country_key, year, season)
    return f"/rain_tiles/{{z}}/{{x}}/{{y}}/{country_key}/{season}/{year}"


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
    return f"/vuln_tiles/{{z}}/{{x}}/{{y}}/{country_key}/{mode}/{year}"


# Endpoints


def image_resp(im):
    cv2_imencode_success, buffer = cv2.imencode(".png", im)
    assert cv2_imencode_success
    io_buf = io.BytesIO(buffer)
    resp = flask.send_file(io_buf, mimetype="image/png")
    resp.headers["Cache-Control"] = "private, max-age=0, no-cache, no-store"
    return resp


def tile(dae, tx, ty, tz, clipping=None, test_tile=False):
    z = pingrid.produce_data_tile(dae.interp2d, tx, ty, tz, 256, 256)
    im = cv2.flip((z - dae.min_val) * 255 / (dae.max_val - dae.min_val), 0)
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
    f"/pnep_tiles/<int:tz>/<int:tx>/<int:ty>/<country_key>/<season>/<int:year>/<int:issue_month>/<int:freq_max>"
)
def pnep_tiles(tz, tx, ty, country_key, season, year, issue_month, freq_max):
    dae = select_pnep(country_key, season, year, issue_month, freq_max)
    p = tuple(CONFIG["countries"][country_key]["marker"])
    clipping = retrieve_geometry(country_key, p, "national", None)
    resp = tile(dae, tx, ty, tz, clipping)
    return resp


@SERVER.route(
    f"/rain_tiles/<int:tz>/<int:tx>/<int:ty>/<country_key>/<season>/<int:year>"
)
def rain_tiles(tz, tx, ty, country_key, season, year):
    dae = select_rain(country_key, year, season)
    p = tuple(CONFIG["countries"][country_key]["marker"])
    clipping = retrieve_geometry(country_key, p, "national", None)
    resp = tile(dae, tx, ty, tz, clipping)
    return resp


@SERVER.route(f"/vuln_tiles/<int:tz>/<int:tx>/<int:ty>/<country_key>/<mode>/<int:year>")
def vuln_tiles(tz, tx, ty, country_key, mode, year):
    config = CONFIG["countries"][country_key]
    df = retrieve_vulnerability(country_key, mode, year)
    im = pingrid.produce_bkg_tile(BGRA(0, 0, 0, 0), 256, 256)
    if mode != "pixel":
        vmax = df["normalized"].max()
        shapes = [
            (
                r["the_geom"],
                pingrid.DrawAttrs(
                    BGRA(0, 0, 255, 255),
                    BGRA(
                        0,
                        0,
                        255,
                        int(r["normalized"] / vmax * 255)
                        if r["normalized"] is not None
                        and not np.isnan(r["normalized"])
                        and not np.isnan(vmax)
                        else 0,
                    ),
                    1,
                    cv2.LINE_AA,
                ),
            )
            for _, r in df.iterrows()
        ]
        im = pingrid.produce_shape_tile(im, shapes, tx, ty, tz, oper="intersection")
    return image_resp(im)


if __name__ == "__main__":
    APP.run_server()
