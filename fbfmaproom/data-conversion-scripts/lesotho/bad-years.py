import cftime
import datetime
import pandas as pd
import xarray as xr

df = pd.read_csv('years_bysector_Lesotho2022 - years_bysector_Lesotho2022.csv')
gb = df.groupby('organization')
def column(name, df):
    df['T'] = df['year'].apply(lambda y: cftime.Datetime360Day(y, 11, 16))
    return df.set_index('T')['rank_bysector'].rename(name).to_xarray()
ds = xr.merge(
    column(k, v) for k, v in df.groupby('organization')
)
#ds.to_zarr('/home/aaron/scratch/iri/data/aaron/fbf-candidate/lesotho/bad-years-ond.zarr')

ds['T'] = ds['T'] + datetime.timedelta(days=60)
ds.to_zarr('/home/aaron/scratch/iri/data/aaron/fbf-candidate/lesotho/bad-years.zarr')

ds['T'] = list(map(
    lambda x: cftime.Datetime360Day(x.year, (x.month - 1 - 2) % 12 + 1, x.day),
    ds['T'].values
))
ds.to_zarr(f'{opts.datadir}/lesotho/bad-years-ond.zarr')
