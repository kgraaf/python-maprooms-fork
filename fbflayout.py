import dash_core_components as dcc
import dash_html_components as html
import dash_table as table
import dash_daq as daq
import dash_leaflet as dlf
import dash_leaflet.express as dlx
import dash_bootstrap_components as dbc
import csv

def app_layout(table_columns):
    return html.Div(
        [
            dcc.Location(id="location", refresh=True),
            map_layout(),
            logo_layout(),
            command_layout(),
            table_layout(table_columns),
            disclaimer_layout(),
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
            "width": "120px",
            "left": "90px",
            "z-index": "1000",
            "height": "fit-content",
            "pointer-events": "auto",
            "padding-left": "10px",
            "padding-right": "10px",
        },
    )

def disclaimer_layout():
    return html.Div(
        [html.H5("This is not an Official Government Maproom")],
        id="disclaimer_panel",
        className="info",
        style={
            "position": "absolute",
            "width": "fit-content",
            "z-index": "1000",
            "height": "fit-content",
            "bottom":"0",
            "right":"0",
            "pointer-events": "auto",
            "padding-left": "10px",
            "padding-right": "10px",
        },
    )

def command_layout():
    return html.Div(
        [
            dcc.Input(id="geom_key", type="hidden"),
            html.Div([
                            dbc.Button("ℹ️", id="table_pop_up",),
                            dbc.Popover([
                                dbc.PopoverHeader("Help"),
                                dbc.PopoverBody(
                                [ html.P("Mode - Refers to the spatial resolution such as: National, Regional, District and Pixel level"),
                                html.P("Issue month - Refers to the month in which the forecast is issued"),
                                html.P("Season - Refers to the rainy season being forecasted"),
                                html.P("Year - Refers to the year being forecasted"),
                                html.P("Severity - Refers to the level of drought severity being targeted"),
                                html.P("Frequency of triggered forecasts - The sliders are used to set the frequency of forecast triggered"),
                                html.P("Map - The map shows the forecast and vulnerability for the selected year and spatial resolution. Click on the layers to view"),
                                ]),
                                        ],
                            id="table_header_body",
                            target="table_pop_up",
                            placement="left",
                            trigger='legacy'
                            )
                        ]
                    ),
            html.Div(
                [

                    html.Label("Mode:"),
                    dcc.Dropdown(
                        id="mode",
                        clearable=False,
                    ),
                ],
                style={
                    "width": "120px",
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
                    "width": "120px",
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
                    "width": "95px",
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
                            "height": "36px",
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
                    "width": "95px",
                    "display": "inline-block",
                    "padding": "10px",
                    "vertical-align": "top",
                },
            ),
            html.Div(
                [
                    html.Label("Severity:"),
                    dcc.Dropdown(
                        id="severity",
                        clearable=False,
                        options=[
                            dict(label="Low", value=0),
                            dict(label="Medium", value=1),
                            dict(label="High", value=2),
                        ],
                        value=0,
                    ),
                ],
                style={
                    "width": "95px",
                    "display": "inline-block",
                    "padding": "10px",
                    "vertical-align": "top",
                },
            ),
            html.Div(
                [
                    html.Label("Frequency of triggered forecasts:"),
                    dcc.Slider(
                        id="freq",
                        min=5,
                        max=95,
                        step=5,
                        value=30,
                        marks={k: dict(label=f"{k}%") for k in range(10, 91, 10)},
                    ),
                ],
                style={
                    "width": "350px",
                    "display": "inline-block",
                    "padding": "10px",
                    "vertical-align": "top",
                },
            ),
            html.Div(
                dcc.Loading(
                    html.A(
                        html.Img(
                            src="assets/ganttit.png",
                            style={"cursor": "pointer"},
                        ),
                        id="gantt",
                        target="_blank",
                    ),
                    type="dot",
                    parent_style={"height": "100%"},
                    style={"opacity": 0.2},
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
            "height": "fit-content",
            "pointer-events": "auto",
            "padding-left": "10px",
            "padding-right": "10px",
        },
    )

def table_layout(table_columns):
    return html.Div(
        [            html.Div([
                                    dbc.Button("ℹ️", id="pop_up", style={
#                                        "position": "absolute",
#                                        "top": "0px",
#                                        "right": "0px",
#                                        "z-index": "1000",
                                    }),
                                    dbc.Popover([
                                        dbc.PopoverHeader("Help"),
                                        dbc.PopoverBody(
                                [ html.P("ENSO State - Displays whether an El Nino, Neutral or La Nina state occurred during the year"),
                                html.P("Forecast - Displays all the historical flexible forecast for the selected issue month and location"),
                                html.P("Rain Rank - Presents the ranking of the rainfall for the year compared to all the years"),
                                html.P("Reported Bad Year - Historical drought years based on farmers recollection"),
                                html.P("Worthy Action - Drought was forecasted and a ‘bad year’ occurred"),
                                html.P("Act-in-Vain - Drought was forecasted but a ‘bad year’ did not occur"),
                                html.P("Fail-to-Act - No drought was forecasted but a ‘bad year’ occurred"),
                                html.P("Worthy-Inaction - No drought was forecasted, and no ‘bad year’ occurred"),
                                html.P("Rate - Gives the percentage of worthy-action and worthy-inactions")],
                                        ),
                                                ],
                                    id="header_body",
                                    target="pop_up",
                                    placement="left",
                                    trigger='legacy'
                                    )
                                ]
                            ),
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
                            "height": "350px",
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
            "height": "fit-content",
            "width": "600px",
            "pointer-events": "auto",
            "padding-left": "10px",
            "padding-right": "10px",
        },
    )
