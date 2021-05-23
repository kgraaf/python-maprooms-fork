import dash_html_components as html
import dash_leaflet as dl
import dash_leaflet.express as dlx
from dash import Dash
from dash_extensions.javascript import assign

countries = [
    dict(name="Denmark", iso2="dk", lat=56.26392, lon=9.501785),
    dict(name="Sweden", iso2="se", lat=59.334591, lon=18.063240),
    dict(name="Norway", iso2="no", lat=59.911491, lon=9.501785),
]

geojson = dlx.dicts_to_geojson([{**c, **dict(tooltip=c["name"])} for c in countries])

draw_flag = assign(
    """function(feature, latlng){
const flag = L.icon({iconUrl: `https://flagcdn.com/64x48/${feature.properties.iso2}.png`, iconSize: [64, 48]});
return L.marker(latlng, {icon: flag});
}"""
)

app = Dash()
app.layout = html.Div(
    [
        dl.Map(
            children=[
                dl.TileLayer(),
                dl.GeoJSON(
                    data=geojson,
                    options=dict(pointToLayer=draw_flag),
                    zoomToBounds=True,
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
    ]
)

if __name__ == "__main__":
    app.run_server()
