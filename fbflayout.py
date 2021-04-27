import dash_core_components as dcc
import dash_html_components as html
import dash_table as table
import dash_daq as daq
import dash_leaflet as dlf
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
    return dlf.Map(
        [
            dlf.LayersControl(
                [
                    dlf.BaseLayer(
                        dlf.TileLayer(
                            url="https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
                        ),
                        name="Street",
                        checked=True,
                    ),
                    dlf.BaseLayer(
                        dlf.TileLayer(
                            url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
                        ),
                        name="Topo",
                        checked=False,
                    ),
                    dlf.Overlay(
                        dlf.TileLayer(opacity=0.8, id="pnep_layer"),
                        name="Forecast",
                        checked=True,
                    ),
                    dlf.Overlay(
                        dlf.TileLayer(opacity=0.8, id="vuln_layer"),
                        name="Vulnerability",
                        checked=False,
                    ),
                ],
                position="topleft",
                id="layers_control",
            ),
            dlf.LayerGroup(
                [
                    dlf.Polygon(
                        positions=[(0, 0), (0, 0)],
                        color="rgb(49, 109, 150)",
                        fillColor="orange",
                        fillOpacity=0.1,
                        weight=1,
                        id="feature",
                    ),
                    dlf.Marker(
                        [
                            dlf.Popup(id="marker_popup"),
                        ],
                        position=(0, 0),
                        draggable=True,
                        id="marker",
                    ),
                ],
                id="pixel_layer",
            ),
            dlf.ScaleControl(imperial=False, position="topleft"),
            dlf.Colorbar(
                "Vulnerability",
                id="vuln_colorbar",
                position="bottomleft",
                width=300,
                height=10,
                min=0,
                max=5,
                nTicks=5,
                opacity=0.8,
            ),
            dlf.Colorbar(
                id="pnep_colorbar",
                position="bottomleft",
                width=300,
                height=10,
                min=0,
                max=100,
                nTicks=5,
                opacity=0.8,
                tooltip=True,
            ),
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
                                value=v.lower(),
                            )
                            for v in ["National", "Regional", "District", "Pixel"]
                        ],
                        value="national",
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
                        step=5,
                        value=(15, 30),
                        marks={k: dict(label=f"{k}%") for k in range(10, 91, 10)},
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
            html.Div(
                html.A(
                    html.Img(
                        src="assets/ganttit.png",
                        style={"cursor": "pointer"},
                    ),
                    id="gantt",
                    href="https://fist-fbf-gantt.iri.columbia.edu/Nov/2020/Pixel/Dec-Jan-Feb/1/30/15/",
                    target="_blank",
                ),
                style={
                    "width": "100px",
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
                                "rule": "font-weight: bold; background-color: rgb(255, 255, 255); color: rgb(251, 101, 57);",
                            },
                            {
                                "selector": "tr:last-child",
                                "rule": "font-weight: bold; background-color: rgb(241, 241, 241); color: black",
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
                            "height": "420px",
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
                                    "filter_query": "{rain_yellow} = 1 && {rain_brown} != 1",
                                    "column_id": "rain_rank",
                                },
                                "backgroundColor": "rgb(251, 177, 57)",
                                "color": "black",
                            },
                            {
                                "if": {
                                    "filter_query": "{rain_brown} = 1 && {rain_yellow} != 1",
                                    "column_id": "rain_rank",
                                },
                                "backgroundColor": "rgb(161, 83, 22)",
                                "color": "white",
                            },
                            {
                                "if": {
                                    "filter_query": "{rain_brown} = 1 && {rain_yellow} = 1",
                                    "column_id": "rain_rank",
                                },
                                "backgroundColor": "rgb(161, 83, 22)",
                                "color": "rgb(255, 226, 178)",
                            },
                            {
                                "if": {
                                    "filter_query": "{pnep_yellow} = 1 && {pnep_brown} != 1",
                                    "column_id": "forecast",
                                },
                                "backgroundColor": "rgb(251, 177, 57)",
                                "color": "black",
                            },
                            {
                                "if": {
                                    "filter_query": "{pnep_brown} = 1 && {pnep_yellow} != 1",
                                    "column_id": "forecast",
                                },
                                "backgroundColor": "rgb(161, 83, 22)",
                                "color": "white",
                            },
                            {
                                "if": {
                                    "filter_query": "{pnep_brown} = 1 && {pnep_yellow} = 1",
                                    "column_id": "forecast",
                                },
                                "backgroundColor": "rgb(161, 83, 22)",
                                "color": "rgb(255, 226, 178)",
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
