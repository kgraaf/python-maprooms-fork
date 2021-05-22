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
                                        src="assets/Ethiopia_IRI_98x48.png", height="30px"
                                    )
                                ),
                                dbc.Col(dbc.NavbarBrand("Onset Maproom", className="ml-2")),
                            ],
                            align="center",
                            no_gutters=True,
                        ),
                        href="https://plot.ly",
                    ),
                    dbc.NavbarToggler(id="navbar-toggler"),
                    dbc.Collapse(
                        dbc.Row(
                            [
                                dbc.Col(dbc.Input(type="search", placeholder="Search")),
                                dbc.Col(
                                    dbc.Button(
                                        "Search", color="primary", className="ml-2"
                                    ),
                                    width="auto",
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
                color="dark",
                dark=True,
            ),
            dbc.Row(
                children=[
                    dbc.Col(
                        width=12,
                        children=[
                            "Hello",
                        ],
                    ),
                ],
            ),
            dbc.Row(
                children=[
                    dbc.Col(
                        sm=12,
                        md=2,
                        style={"background-color": "red"},
                        children=["Column 1"],
                    ),
                    dbc.Col(
                        sm=12,
                        md=10,
                        style={"background-color": "gray"},
                        children=[
                            dbc.Row(
                                children=[
                                    dbc.Col(
                                        md=12,
                                        lg=6,
                                        style={"background-color": "yellow"},
                                        children=["Column 2"],
                                    ),
                                    dbc.Col(
                                        md=12,
                                        lg=6,
                                        style={"background-color": "green"},
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
