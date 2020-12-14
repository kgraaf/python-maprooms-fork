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
from dash.dependencies import Output, Input

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


# Functions and definitions


FuncInterp2d = Callable[[np.ndarray, np.ndarray], np.ndarray]


class DataArrayEntry(NamedTuple):
    name: str
    data_array: xr.DataArray
    interp2d: Optional[FuncInterp2d]
    min_val: Optional[float]
    max_val: Optional[float]


class Extent(NamedTuple):
    dim: str
    left: float
    right: float
    point_width: float


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


def extents(da: xr.DataArray, defaults: Optional[List[Extent]] = None) -> List[Extent]:
    if defaults is None:
        defaults = [Extent(dim, np.nan, np.nan, np.nan) for dim in da.dims]
    return [extent(da, k, defaults[i]) for i, k in enumerate(da.dims)]


def g_lon(tx: int, tz: int) -> float:
    return tx * 360 / 2 ** tz - 180


def g_lat(tx: int, tz: int) -> float:
    return tx * 180 / 2 ** tz - 90


def g_lat_3857(tx: int, tz: int) -> float:
    a = 2 * math.pi * tx / 2 ** tz - math.pi
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
    f = interpolate.interp2d(x, y, z, kind="linear", copy=False, bounds_error=True)
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
        (a + (b - a) / 2.0 for a, b in tile_extents(g_lat_3857, tx, tz, tile_height)),
        np.double,
    )
    z = interp2d(x, y)
    return z


import cv2
import numpy as np


def produce_test_tile(w: int = 256, h: int = 256) -> np.ndarray:
    line_thickness = 1
    line_type = cv2.LINE_4  # cv2.LINE_4 | cv2.LINE_8 | cv2.LINE_AA
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


def open_data_arrays():
    rs = {}
    bath = xr.open_dataset("bath432.nc", decode_times=False)["bath"].transpose("Y", "X")
    xs = np.fromiter(
        ((-180 + 0.41570438799076215) + i * 0.8314087759815243 for i in range(433)),
        np.double,
    )
    ys = np.fromiter(
        ((-90 + 0.4147465437788018) + i * 0.8294930875576036 for i in range(217)),
        np.double,
    )
    bath = bath.assign_coords(X=xs, Y=ys)
    rs["bath"] = DataArrayEntry(
        "bath", bath, create_interp2d(bath, bath.dims), -9964.0, 7964.0
    )
    print(bath)

    rain = xr.open_dataset("rain.nc", decode_times=False)["prcp_est"].transpose(
        "Y", "X", ...
    )
    rs["rain"] = DataArrayEntry("rain", rain, None, None, None)
    print(rain)
    # print(from_months_since_v(rain["T"].values))

    pnep = xr.open_dataset("pnep.nc", decode_times=False)["pne"].transpose(
        "Y", "X", ...
    )
    pnep["T"] = pnep["S"] + pnep["L"]
    rs["pnep"] = DataArrayEntry("pnep", pnep, None, None, None)
    print(pnep)
    # print(from_months_since_v(pnep["S"].values))
    # print(from_months_since_v(pnep["T"].values))
    # print(pnep.sel(S=274, L=2.5).values)

    return rs


DATA_ARRAYS: Dict[str, DataArrayEntry] = open_data_arrays()


# Server


server = flask.Flask(__name__)
app = dash.Dash(
    __name__,
    server=server,
    url_base_pathname="/fbfmaproom/",
    meta_tags=[
        {"name": "description", "content": "content description 1234"},
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
    ],
)


# Layout


map = dl.Map(
    [
        dl.LayersControl(
            [
                dl.BaseLayer(
                    dl.TileLayer(
                        url="https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
                    ),
                    name="street",
                    checked=True,
                ),
                dl.BaseLayer(
                    dl.TileLayer(
                        url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
                    ),
                    name="topo",
                    checked=False,
                ),
                dl.Overlay(
                    dl.TileLayer(
                        url="/tiles/bath/{z}/{x}/{y}",
                        opacity=0.3,
                    ),
                    name="rain",
                    checked=True,
                ),
            ],
            position="topleft",
        ),
        dl.ScaleControl(imperial=False),
    ],
    # center=(8, 40),
    zoom=0,
    style={"width": "100%", "height": "100%", "position": "absolute"},
    # crs="EPSG3857",
    # crs="EPSG4326",
    # crs="Simple",
)


app.layout = html.Div(
    [
        map,
        html.Div(
            [
                html.H2("FBF—Maproom"),
                html.Label("Issue month:"),
                dcc.Dropdown(
                    id="issue_month",
                    options=[
                        dict(label=pd.to_datetime(v, format="%m").month_name(), value=v)
                        for v in range(8, 12)
                    ],
                    value=10,
                    clearable=False,
                ),
                html.Br(),
                html.Label("Year:"),
                daq.Slider(
                    id="year",
                    min=1982,
                    max=2020,
                    value=2020,
                    handleLabel={"showCurrentValue": True, "label": " "},
                    marks={1990: "1990", 2000: "2000", 2010: "2010", 2020: "2020"},
                    step=1,
                ),
                html.Br(),
            ],
            id="info",
            className="info",
            style={
                "position": "absolute",
                "top": "10px",
                "right": "10px",
                "z-index": "1000",
                "height": "80%",
                "width": "600px",
                "pointer-events": "auto",
                "padding-left": "25px",
                "padding-right": "25px",
            },
        ),
    ]
)


# Callbacks


# Endpoints


@server.route(f"/tiles/<data_array>/<int:tz>/<int:tx>/<int:ty>")
def tiles(data_array, tz, tx, ty):
    # dae = DATA_ARRAYS[data_array]
    # z = produce_tile(dae.interp2d, tx, ty, tz, 256, 256)
    # im = cv2.flip((z - dae.min_val) * 255 / (dae.max_val - dae.min_val), 0)
    im = produce_test_tile(256, 256)
    cv2_imencode_success, buffer = cv2.imencode(".png", im)
    assert cv2_imencode_success
    io_buf = io.BytesIO(buffer)
    resp = flask.send_file(io_buf, mimetype="image/png")
    resp.headers["Cache-Control"] = "private, max-age=0, no-cache, no-store"
    return resp


if __name__ == "__main__":
    app.run_server()
