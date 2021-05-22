import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table as table
import dash_leaflet as dlf


def app_layout():
    return dbc.Container(
        fluid=True,
        style={"padding-left": "0px", "padding-right": "0px"},
        children=[
            dcc.Location(id="location", refresh=True),
            navbar(),
            dbc.Row(
                no_gutters=True,
                children=[
                    dbc.Col(
                        sm=12,
                        md=2,
                        style={"background-color": "#eeeeee"},
                        children=column1_layout(),
                    ),
                    dbc.Col(
                        sm=12,
                        md=10,
                        style={"background-color": "white"},
                        children=[
                            dbc.Row(
                                no_gutters=True,
                                children=[
                                    dbc.Col(
                                        md=12,
                                        lg=6,
                                        style={
                                            "background-color": "white",
                                        },
                                        children=column2_layout(),
                                    ),
                                    dbc.Col(
                                        md=12,
                                        lg=6,
                                        style={
                                            "background-color": "#ddffdd",
                                            "height": "500px",
                                        },
                                        children=column3_layout(),
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def navbar():
    return dbc.Navbar(
        sticky="top",
        children=[
            html.A(
                dbc.Row(
                    [
                        dbc.Col(
                            html.Img(
                                src="assets/Ethiopia_IRI_98x48.png",
                                height="30px",
                            )
                        ),
                        dbc.Col(dbc.NavbarBrand("Onset Maproom", className="ml-2")),
                    ],
                    align="center",
                    no_gutters=True,
                ),
                href="https://iridl.ldeo.columbia.edu",
            ),
            dbc.NavbarToggler(id="navbar-toggler"),
            dbc.Collapse(
                dbc.Row(
                    no_gutters=True,
                    className="ml-auto flex-nowrap mt-3 mt-md-0",
                    align="center",
                    children=[
                        dbc.Col(
                            dbc.Select(
                                id="select",
                                value="en",
                                bs_size="sm",
                                options=[
                                    {"label": "English", "value": "en"},
                                    {
                                        "label": "Español",
                                        "value": "es",
                                        "disabled": True,
                                    },
                                    {"label": "Française", "value": "fr"},
                                    {"label": "Русский", "value": "ru"},
                                    {"label": "العربية", "value": "ar"},
                                    {"label": "हिन्दी", "value": "hi"},
                                    {"label": "中文", "value": "zh"},
                                ],
                            ),
                        ),
                    ],
                ),
                id="navbar-collapse",
                navbar=True,
            ),
        ],
        color="dark",
        dark=True,
    )


def column1_layout():
    return dbc.Container(
        fluid=True,
        style={"padding-bottom": "1rem", "padding-top": "1rem"},
        children=[
            dbc.FormGroup(
                row=True,
                inline=True,
                children=[
                    dbc.Label("Date:", size="sm", html_for="date_input", width="auto"),
                    dbc.Col(
                        dbc.Select(
                            id="date_input",
                            value="onset",
                            bs_size="sm",
                            options=[
                                {"label": "Onset", "value": "onset"},
                                {"label": "Cessation", "value": "cessation"},
                            ],
                        ),
                        width="auto",
                    ),
                ],
            ),
            dbc.FormGroup(
                row=True,
                inline=True,
                children=[
                    dbc.Label(
                        "Yearly statistics:",
                        size="sm",
                        html_for="yearly_stats_input",
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Select(
                            id="yearly_stats_input",
                            value="onset",
                            bs_size="sm",
                            options=[
                                {"label": "Mean", "value": "mean"},
                                {"label": "Standard deviation", "value": "stddev"},
                                {"label": "Probability of exceedance", "value": "pe"},
                            ],
                        ),
                        width="auto",
                    ),
                ],
            ),
            dbc.FormGroup(
                row=True,
                inline=True,
                children=[
                    dbc.Label(
                        "Wet day definition:",
                        size="sm",
                        html_for="wet_day_def_input",
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Input(
                            id="wet_day_def_input",
                            value=0,
                            bs_size="sm",
                            type="number",
                            min=0,
                            max=10,
                            step=1,
                        ),
                        width="auto",
                    ),
                ],
            ),
            dbc.Button("Update", size="sm", color="primary"),
        ],
    )


def column2_layout():
    return dbc.Container(
        fluid=True,
        style={"padding": "0rem"},
        children=[
            dlf.Map(
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
                        ],
                        position="topleft",
                        id="layers_control",
                    ),
                    dlf.ScaleControl(imperial=False, position="topleft"),
                ],
                id="map",
                style={
                    "width": "100%",
                    "height": "500px",
                    # "position": "absolute",
                },
            ),
        ],
    )


def column3_layout():
    return ["Column 3"]
