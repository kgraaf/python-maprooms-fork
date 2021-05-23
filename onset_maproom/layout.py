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
            navbar_layout(),
            dbc.Row(
                no_gutters=True,
                children=[
                    dbc.Col(
                        sm=12,
                        md=2,
                        style={
                            "background-color": "white",
                            "border-style": "solid",
                            "border-color": "#eeeeee",
                            "border-width": "0px 1px 0px 0px",
                        },
                        children=column1_content(),
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
                                        width=12,
                                        style={
                                            "background-color": "white",
                                        },
                                        children=column2_content(),
                                    ),
                                ],
                            ),
                            dbc.Row(
                                no_gutters=True,
                                children=[
                                    dbc.Col(
                                        width=12,
                                        style={
                                            "background-color": "white",
                                            "min-height": "100px",
                                            "border-style": "solid",
                                            "border-color": "#eeeeee",
                                            "border-width": "1px 0px 0px 0px",
                                        },
                                        children=column3_content(),
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def navbar_layout():
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


def column1_content():
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
                            value="mean",
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


def column2_content():
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
                    dlf.ScaleControl(imperial=False, position="bottomleft"),
                ],
                id="map",
                style={
                    "width": "100%",
                    "height": "300px",
                },
            ),
        ],
    )


def column3_content():
    return dbc.Container(
        fluid=True,
        style={"padding": "1rem"},
        children=[
            dbc.Tabs(
                children=[
                    dbc.Tab(
                        tab1_content(),
                        label="Description",
                    ),
                    dbc.Tab(tab2_content(), label="Charts"),
                    dbc.Tab(tab3_content(), label="Tables"),
                    dbc.Tab(tab4_content(), label="Documentation"),
                ],
            ),
        ],
    )


def tab1_content():
    return dbc.Container(
        fluid=True,
        style={"padding": "1rem"},
        children=[
            html.H5("Historical Onset and Cessation Date"),
            html.P(
                "The Maproom explores historical rainy season onset and cessation "
                "dates based on user-defined definitions. The date when the rainy "
                "season starts is critical to agriculture planification, in "
                "particular for planting."
            ),
            html.P(
                "By enabling the exploration of the history of onset dates, "
                "the Maproom allows to understand the spatial and temporal variability "
                "of this phenomenon and therefore characterize the risk for a successful "
                "agricultural campaign associated with it."
            ),
            tab2_content(),
        ],
    )


def tab2_content():
    return html.Img(
        style={"height": "300px"},
        src="https://iridl.ldeo.columbia.edu/dlcharts/render/905cdac6e87a58586967e115a18e615d01530ddd?_wd=1200px&_ht=600px&_langs=en&_mimetype=image%2Fpng&region=bb%3A39.375%3A7.125%3A39.625%3A7.375%3Abb&waterBalanceCess=3&drySpellCess=10&plotrange2=15",
    )


def tab3_content():
    return "tab 3"


def tab4_content():
    return "tab 4"
