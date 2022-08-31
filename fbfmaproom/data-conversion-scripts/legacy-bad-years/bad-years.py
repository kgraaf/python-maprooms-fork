import fbfmaproom
import os
import pandas as pd
import psycopg2
import pyaconf

"""Translates legacy boolean bad years datasets from the db to zarr"""

def fetch_bad_years(country_key):
    config = CONFIG
    conn = psycopg2.connect(
        dbname=dbconfig['name'],
        host=dbconfig['host'],
        port=dbconfig['port'],
        user=dbconfig['user'],
        password=dbconfig['password'],
    )
    df = pd.read_sql_query(
        psycopg2.sql.SQL(
            """
            select month_since_01011960, bad_year2 as bad
            from public.fbf_maproom
            where lower(adm0_name) = %(country_key)s
            """
        ),
        conn,
        params={"country_key": country_key},
        dtype={"bad": float},
    )
    df["T"] = df["month_since_01011960"].apply(fbfmaproom.from_month_since_360Day)
    df = df.drop("month_since_01011960", axis=1)
    df = df.set_index("T")
    return df


config_files = os.environ["CONFIG"].split(":")

CONFIG = {}
for fname in config_files:
    CONFIG = pyaconf.merge([CONFIG, pyaconf.load(fname)])
dbconfig = CONFIG['dbpool']

# Countries for which the zarr bad years data is a copy of the
# db. These are legacy bad years datasets. New ones go directly to
# zarr.
DB_BAD_YEARS_COUNTRIES = ['malawi', 'madagascar', 'madagascar-ond', 'niger', 'guatemala']
for country in DB_BAD_YEARS_COUNTRIES:
    ds_key = f'{country}/bad-years'
    if ds_key in opts.datasets:
        print(ds_key)
        df = fetch_bad_years(country)
        zarrpath = f'{opts.datadir}/{ds_key}.zarr'
        df.to_xarray().to_zarr(zarrpath)
