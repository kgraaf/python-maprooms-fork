import cftime
import pandas as pd

bad_years = {2021, 2017, 2016, 2014, 2011, 2009, 2008, 2006, 2005, 2000, 1999, 1992, 1985, 1984, 1983}


# MAM
df = pd.DataFrame(
    index=[cftime.Datetime360Day(y, 4, 16) for y in range(1983, 2022)],
)
df.index.name = 'T'
df['bad'] = df.index.to_series().apply(lambda x: x.year in bad_years)
df.to_xarray().to_zarr('/home/aaron/scratch/iri/data/aaron/fbf-candidate/ethiopia/bad-years-mam.zarr')


# OND
df = pd.DataFrame(
    index=[cftime.Datetime360Day(y, 11, 16) for y in range(1983, 2022)],
)
df.index.name = 'T'
df['bad'] = df.index.to_series().apply(lambda x: x.year in bad_years)
df.to_xarray().to_zarr('/home/aaron/scratch/iri/data/aaron/fbf-candidate/ethiopia/bad-years-ond.zarr')
