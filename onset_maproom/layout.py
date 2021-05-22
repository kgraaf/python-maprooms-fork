import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table as table
import dash_leaflet as dlf


def app_layout():
    return dbc.Container(
        fluid=True,
        children=[
            dcc.Location(id="location", refresh=True),
            dbc.Navbar(
                children=[
                    html.A(
                        # Use row and col to control vertical alignment of logo / brand
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Img(
                                        src="assets/Ethiopia_IRI_98x48.png",
                                        height="30px",
                                    )
                                ),
                                dbc.Col(
                                    dbc.NavbarBrand("Onset Maproom", className="ml-2")
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
                                    dbc.Input(
                                        type="search",
                                        placeholder="Search",
                                        bs_size="sm",
                                    )
                                ),
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
                                        ],
                                    ),
                                ),
                            ],
                            # no_gutters=True,
                            className="ml-auto flex-nowrap mt-3 mt-md-0",
                            align="center",
                        ),
                        id="navbar-collapse",
                        navbar=True,
                    ),
                ],
                color="dark",
                dark=True,
            ),
            dbc.Row(
                children=[
                    dbc.Col(
                        sm=12,
                        md=3,
                        lg=2,
                        style={"background-color": "red", "height": "100px"},
                        children=["Column 1"],
                    ),
                    dbc.Col(
                        sm=12,
                        md=9,
                        lg=10,
                        style={"background-color": "white"},
                        children=[
                            dbc.Row(
                                children=[
                                    dbc.Col(
                                        md=12,
                                        lg=6,
                                        style={
                                            "background-color": "yellow",
                                            "height": "200px",
                                        },
                                        children=["Column 2"],
                                    ),
                                    dbc.Col(
                                        md=12,
                                        lg=6,
                                        style={
                                            "background-color": "green",
                                            "height": "700px",
                                        },
                                        children=["Column 3"],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
