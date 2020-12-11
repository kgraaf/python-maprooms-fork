import sys
import os
import json
import time
import math
import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import xarray as xr

from PIL import Image, ImageOps, ImageDraw
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


def from_months_since(x, year_since=1960):
    int_x = int(x)
    return datetime.date(
        int_x // 12 + year_since, int_x % 12 + 1, int(30 * (x - int_x) + 1)
    )


from_months_since_v = np.vectorize(from_months_since)


CONFIG = pyaconf.load(os.environ["CONFIG"])


# Data

bath = xr.open_dataset("bath432.nc", decode_times=False)
print(bath)

rain = xr.open_dataset("rain.nc", decode_times=False)
print(rain)
# print(from_months_since_v(rain["T"].values))

pnep = xr.open_dataset("pnep.nc", decode_times=False)
pnep["T"] = pnep["S"] + pnep["L"]
print(pnep)
# print(from_months_since_v(pnep["S"].values))
# print(from_months_since_v(pnep["T"].values))
# print(pnep.sel(S=274, L=2.5).values)


print(rain["prcp_est"].min(), rain["prcp_est"].max())
im = Image.fromarray(rain["prcp_est"].sel(T=276.5).values / 2000 * 255.0).convert("L")
im = ImageOps.flip(im)
im.save("image.png")

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
                        url="/tiles/rain35/{z}/{x}/{y}",
                        opacity=0.25,
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


def tile_to_x(x, z):
    return x / 2 ** z * 360 - 180


def tile_to_y(y, z):
    return y / 2 ** z * 180 - 90


def tile_to_y_3857(y, z):
    n = math.pi - 2 * math.pi * y / 2 ** z
    return 180 / math.pi * math.atan(0.5 * (math.exp(n) - math.exp(-n)))


def tile_to_slice_x(x, z):
    r1 = tile_to_x(x, z)
    r2 = tile_to_x(x + 1, z)

    if r2 % 360 == -180:
        r1 = r1 % 360 + 360
        r2 = r2 % 360 + 360
    elif r1 == 180:
        r1 -= 360
        r2 -= 360
    elif r2 % 360 == 180 and r2 != 180:
        r1 = r1 % 360
        r2 = r2 % 360
    return slice(r1, r2)


def tile_to_slice_y(y, z):
    r1 = tile_to_y_3857(y, z)
    r2 = tile_to_y_3857(y + 1, z)
    return slice(r2, r1)




DPI = 100


@server.route(f"/tiles/<dataset>/<int:z>/<int:x>/<int:y>")
def tiles(dataset, z, x, y):
    slice_x = tile_to_slice_x(x, z)
    slice_y = tile_to_slice_y(y, z)
    print(
        f"*** tiles: {dataset=!r}, {z=!r}, {x=!r}, {y=!r}, "
        f"{slice_x=!r}, {slice_y=!r}"
    )
    da = bath["bath"].sel(X=slice_x, Y=slice_y)
    print(da)


    ts = np.fromiter((x / 256.0 for x in range(257)), np.double)
    vs = np.fromiter((tile_to_y_3857(y + d, z) for d in ts), np.double)
    fig, ax = plt.subplots()
    ax.plot(ts, vs)
    ax.grid()
    fig.savefig("test.pdf")


    fig = Figure(figsize=(256 / DPI, 256 / DPI), dpi=DPI)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1], projection=crs.Mercator())
    ax.set_xmargin(0)
    ax.set_ymargin(0)
    ax.set_frame_on(False)
    # ax.set_axis_off()
    # ax.set_extent((slice_x.start, slice_x.stop, slice_y.start, slice_y.stop))
    da.plot.imshow(ax=ax, add_colorbar=False, transform=crs.PlateCarree())

    f_out = BytesIO()
    fig.savefig("fig.png", format="png")
    fig.savefig(f_out, format="png")
    f_out.seek(0)

    im = Image.open(f_out).convert("RGB")
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0) + im.size, outline=(0, 0, 255))
    draw.line((0, 0) + im.size, fill=(0, 0, 255))
    draw.line((0, im.size[1], im.size[0], 0), fill=(0, 0, 255))
    draw.text((128, 128), f"{z},{x},{y}", (255, 0, 0))
    f_out.seek(0)
    im.save(f_out, "PNG", transparency=(255, 255, 255))
    f_out.seek(0)

    resp = flask.send_file(f_out, mimetype="image/png")
    resp.headers["Cache-Control"] = "private, max-age=0, no-cache, no-store"
    return resp


@server.route(f"/datalibrary/<crop>")
def datalibrary(crop):
    r = requests.get(
        f"http://iridl.ldeo.columbia.edu/home/.remic/.Leap/.WRSI/.Meher/.FinalIcat/Crop/({crop.capitalize()})/VALUE/X/Y/fig-/colors/-fig//XOVY/1/psdef//plotaxislength/5000/psdef//plotborder/0/psdef/.png"
    )
    f_in = BytesIO(r.content)
    img = Image.open(f_in)
    img = img.convert("RGBA")
    imgnp = np.array(img)
    white = np.sum(imgnp[:, :, :3], axis=2)
    white_mask = np.where(white == 255 * 3, 1, 0)
    alpha = np.where(white_mask, 0, imgnp[:, :, -1])
    imgnp[:, :, -1] = alpha
    img = Image.fromarray(np.uint8(imgnp))
    f_out = BytesIO()
    img.save(f_out, "PNG")
    f_out.seek(0)
    return flask.send_file(f_out, mimetype="image/png")


if __name__ == "__main__":
    app.run_server()
