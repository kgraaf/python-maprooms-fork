import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table as table
import dash_leaflet as dlf


IRI_BLUE = "rgb(25,57,138)"
IRI_GRAY = "rgb(113,112,116)"
LIGHT_GRAY = "#eeeeee"


def app_layout():
    return dbc.Container(
        [
            dcc.Location(id="location", refresh=True),
            navbar_layout(),
            dbc.Row(
                [
                    dbc.Col(
                        column1_content(),
                        sm=12,
                        md=4,
                        style={
                            "background-color": "white",
                            "border-style": "solid",
                            "border-color": LIGHT_GRAY,
                            "border-width": "0px 1px 0px 0px",
                        },
                    ),
                    dbc.Col(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        column2_content(),
                                        width=12,
                                        style={
                                            "background-color": "white",
                                        },
                                    ),
                                ],
                                no_gutters=True,
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        column3_content(),
                                        width=12,
                                        style={
                                            "background-color": "white",
                                            "min-height": "100px",
                                            "border-style": "solid",
                                            "border-color": LIGHT_GRAY,
                                            "border-width": "1px 0px 0px 0px",
                                        },
                                    ),
                                ],
                                no_gutters=True,
                            ),
                        ],
                        sm=12,
                        md=8,
                        style={"background-color": "white"},
                    ),
                ],
                no_gutters=True,
            ),
        ],
        fluid=True,
        style={"padding-left": "0px", "padding-right": "0px"},
    )


def navbar_layout():
    return dbc.Navbar(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(
                            html.Img(
                                src="assets/Ethiopia_IRI_98x48.png",
                                height="30px",
                            )
                        ),
                        dbc.Col(
                            dbc.NavbarBrand(
                                "Climate and Agriculture / Onset Maproom",
                                className="ml-2",
                            )
                        ),
                    ],
                    align="center",
                    no_gutters=True,
                ),
                href="https://iridl.ldeo.columbia.edu",
            ),
            dbc.NavbarToggler(id="navbar-toggler"),
            dbc.Collapse(
                dbc.Row(
                    [
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
                    no_gutters=True,
                    className="ml-auto flex-nowrap mt-3 mt-md-0",
                    align="center",
                ),
                id="navbar-collapse",
                navbar=True,
            ),
        ],
        sticky="top",
        color=IRI_GRAY,
        dark=True,
    )


def column1_content():
    return dbc.Container(
        [
            html.H5(
                [
                    "Historical Onset and Cessation Date ",
                ]
            ),
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
            dbc.FormGroup(
                [
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
                row=True,
                inline=True,
            ),
            dbc.FormGroup(
                [
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
                row=True,
                inline=True,
            ),
            dbc.FormGroup(
                [
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
                row=True,
                inline=True,
            ),
            html.P(
                "This map shows the monthly geopotential height climatology "
                "and anomalies at the 250 hPa pressure level in the atmosphere using "
                "the 1981-2010 base period for the month shown."
            ),
            html.P(
                "Monthly geopotential height climatology values are shown as "
                "grey contours with a contour interval of 60 geopotential meters (gpm). "
                "Positive geopotential height anomalies are shown in yellow and orange, "
                "and negative anomalies are shown in shades of blue and are also "
                "expressed in units of gpm."
            ),
            html.H5("Dataset Documentation"),
            html.P(
                "Monthly Geopotential Height Anomaly at 250 hPa. "
                "Data: NCEP-NCAR Reanalysis monthly geopotential height at "
                "250 hPa on a 2.5° lat/lon grid."
            ),
        ],
        fluid=True,
        className="scrollable-panel",
        style={"padding-bottom": "1rem", "padding-top": "1rem"},
    )


def column2_content():
    return dbc.Container(
        [
            dlf.Map(
                [
                    dlf.LayersControl(
                        [
                            dlf.BaseLayer(
                                dlf.TileLayer(
                                    url="https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
                                ),
                                name="Street",
                                checked=False,
                            ),
                            dlf.BaseLayer(
                                dlf.TileLayer(
                                    url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
                                ),
                                name="Topo",
                                checked=True,
                            ),
                        ],
                        position="topleft",
                        id="layers_control",
                    ),
                    dlf.ScaleControl(imperial=False, position="bottomleft"),
                ],
                id="map",
                center=[3.5, -74],
                zoom=7,
                style={
                    "width": "100%",
                    "height": "93vh",
                },
            ),
        ],
        fluid=True,
        style={"padding": "0rem"},
    )


def column3_content():
    return html.Img(
        style={"width": "600px"},
        src=(
            "https://iridl.ldeo.columbia.edu/dlcharts/render/"
            "905cdac6e87a58586967e115a18e615d01530ddd?_wd=1200px&_ht=600px"
            "&_langs=en&_mimetype=image%2Fpng"
            "&region=bb%3A39.375%3A7.125%3A39.625%3A7.375%3Abb"
            "&waterBalanceCess=3&drySpellCess=10&plotrange2=15"
        ),
    )
