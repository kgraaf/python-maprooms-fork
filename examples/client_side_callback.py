import dash_html_components as html
import dash_leaflet as dl
import dash_core_components as dcc
import dash_leaflet.express as dlx
from dash import Dash
from dash.dependencies import Output, Input
from dash_extensions.javascript import assign

cities = [
    dict(name="Aalborg", lat=57.0268172, lon=9.837735),
    dict(name="Aarhus", lat=56.1780842, lon=10.1119354),
    dict(name="Copenhagen", lat=55.6712474, lon=12.5237848),
]

dd_options = [dict(value=c["name"], label=c["name"]) for c in cities]
dd_defaults = [o["value"] for o in dd_options]

geojson = dlx.dicts_to_geojson([{**c, **dict(tooltip=c["name"])} for c in cities])

geojson_filter = assign(
    "function(feature, context){return context.props.hideout.includes(feature.properties.name);}"
)

app = Dash()
app.layout = html.Div(
    [
        dl.Map(
            children=[
                dl.TileLayer(),
                dl.GeoJSON(
                    data=geojson,
                    options=dict(filter=geojson_filter),
                    hideout=dd_defaults,
                    id="geojson",
                ),
            ],
            style={
                "width": "100%",
                "height": "50vh",
                "margin": "auto",
                "display": "block",
            },
            id="map",
        ),
        dcc.Dropdown(
            id="dd", value=dd_defaults, options=dd_options, clearable=False, multi=True
        ),
    ]
)

app.clientside_callback(
    "function(x){return x;}", Output("geojson", "hideout"), Input("dd", "value")
)

if __name__ == "__main__":
    app.run_server()
