import dash_core_components as dcc
import dash_html_components as html
import dash_table as table
import dash_daq as daq
import dash_leaflet as dl
import dash_leaflet.express as dlx


def app_layout(table_columns):
    return html.Div(
        [
            dcc.Location(id="location", refresh=True),
            map_layout(),
            logo_layout(),
            command_layout(),
            table_layout(table_columns),
        ]
    )


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
                        name="PNE",
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
                        value="Pixel",
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
                    html.Label("Season:"),
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


def table_layout(table_columns):
    return html.Div(
        [
            html.Div(id="log"),
            dcc.Loading(
                [
                    table.DataTable(
                        id="summary",
                        columns=table_columns,
                        page_action="none",
                        style_table={
                            "height": "auto",
                            "overflowY": "scroll",
                        },
                        css=[
                            {
                                "selector": "tr:first-child",
                                "rule": "display: none",
                            },
                            {
                                "selector": "tr:not(last-child)",
                                "rule": "font-weight: bold; background-color: rgb(255, 255, 255);",
                            },
                            {
                                "selector": "tr:last-child",
                                "rule": "font-weight: bold; background-color: rgb(241, 241, 241);",
                            },
                        ],
                        style_cell={
                            "whiteSpace": "normal",
                            "height": "auto",
                            "textAlign": "center",
                            "border": "1px solid rgb(150, 150, 150)",
                            "background-color": "rgba(255, 255, 255, 0)",
                        },
                        style_data_conditional=[
                            {
                                "if": {
                                    "filter_query": "{year_label} != 'Year'",
                                    "column_id": "year_label",
                                },
                                "color": "rgb(67, 104, 176)",
                                "font-weight": "bold",
                            },
                        ],
                    ),
                    table.DataTable(
                        id="table",
                        columns=table_columns,
                        page_action="none",
                        style_table={
                            "height": "400px",
                            "overflowY": "scroll",
                        },
                        css=[
                            {
                                "selector": "tr:first-child",
                                "rule": "display: none",
                            },
                        ],
                        style_cell={
                            "whiteSpace": "normal",
                            "height": "auto",
                            "textAlign": "center",
                            "border": "1px solid rgb(150, 150, 150)",
                            "backgroundColor": "rgb(248, 248, 248)",
                        },
                        fixed_rows={
                            "headers": False,
                            "data": 0,
                        },
                        style_data_conditional=[
                            {
                                "if": {
                                    "filter_query": "{rain_rank_cat} = 1",
                                    "column_id": "rain_rank",
                                },
                                "backgroundColor": "rgb(251, 177, 57)",
                                "color": "black",
                            },
                            {
                                "if": {
                                    "filter_query": "{rain_rank_cat} = 2",
                                    "column_id": "rain_rank",
                                },
                                "backgroundColor": "rgb(161, 83, 22)",
                                "color": "white",
                            },
                            {
                                "if": {
                                    "filter_query": "{pnep_rank_cat} = 1",
                                    "column_id": "forecast",
                                },
                                "backgroundColor": "rgb(251, 177, 57)",
                                "color": "black",
                            },
                            {
                                "if": {
                                    "filter_query": "{pnep_rank_cat} = 2",
                                    "column_id": "forecast",
                                },
                                "backgroundColor": "rgb(161, 83, 22)",
                                "color": "white",
                            },
                            {
                                "if": {
                                    "filter_query": "{enso_state} = 'El Niño'",
                                    "column_id": "enso_state",
                                },
                                "backgroundColor": "rgb(172, 23, 25)",
                                "color": "white",
                            },
                            {
                                "if": {
                                    "filter_query": "{enso_state} = 'La Niña'",
                                    "column_id": "enso_state",
                                },
                                "backgroundColor": "rgb(24, 101, 152)",
                                "color": "white",
                            },
                            {
                                "if": {
                                    "filter_query": "{enso_state} = 'Neutral'",
                                    "column_id": "enso_state",
                                },
                                "backgroundColor": "rgb(98, 98, 98)",
                                "color": "white",
                            },
                            {
                                "if": {
                                    "filter_query": "{bad_year} = 'Bad'",
                                    "column_id": "bad_year",
                                },
                                "backgroundColor": "rgb(64, 9, 101)",
                                "color": "white",
                            },
                        ],
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
