import sys
import os
import json
import time
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


def from_months_since(x, year_since=1960):
    int_x = int(x)
    return datetime.date(
        int_x // 12 + year_since, int_x % 12 + 1, int(30 * (x - int_x) + 1)
    )


from_months_since_v = np.vectorize(from_months_since)


CONFIG = pyaconf.load(os.environ["CONFIG"])


# Data

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
                        url="/tiles/rain11/{z}/{x}/{y}",
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
    center=(8, 40),
    zoom=6,
    style={"width": "100%", "height": "100%", "position": "absolute"},
    # crs="EPSG3857",
    crs="EPSG4326",
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


@server.route(f"/tiles/<dataset>/<z>/<x>/<y>")
def tiles(dataset, z, x, y):
    print(f"*** tiles: {dataset=!r}, {z=!r}, {x=!r}, {y=!r}")
    # im = Image.new("RGB", (256, 256), (255, 255, 255, 0))
    im = Image.open("bath432-256x256.png").convert("RGB")
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0) + im.size, outline=(0, 0, 255))
    draw.line((0, 0) + im.size, fill=(0, 0, 255))
    draw.line((0, im.size[1], im.size[0], 0), fill=(0, 0, 255))
    draw.text((128, 128), f"{z},{x},{y}", (255, 0, 0))
    f_out = BytesIO()
    im.save(f_out, "PNG", transparency=(255, 255, 255))
    f_out.seek(0)
    return flask.send_file(f_out, mimetype="image/png")


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
