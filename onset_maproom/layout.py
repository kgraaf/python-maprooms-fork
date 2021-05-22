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
            dbc.NavbarSimple(
                children=[
                    dbc.NavItem(dbc.NavLink("Page 1", href="#")),
                    dbc.DropdownMenu(
                        children=[
                            dbc.DropdownMenuItem("More pages", header=True),
                            dbc.DropdownMenuItem("Page 2", href="#"),
                            dbc.DropdownMenuItem("Page 3", href="#"),
                        ],
                        nav=True,
                        in_navbar=True,
                        label="More",
                    ),
                ],
                brand="Onset Maproom",
                brand_href=None,
                color="dark",
                dark=True,
                fluid=True,
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
