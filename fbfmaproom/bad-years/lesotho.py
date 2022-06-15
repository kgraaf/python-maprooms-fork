import cftime
import pandas as pd


df = pd.read_csv('FbF Lesotho Historical Vulnerable Years by sector.xlsx - Sheet1.csv')
df['year'] = df['year'].str.slice(0,4).astype(int) + 1
df['year'] = df['year'].apply(lambda x: cftime.Datetime360Day(x, 1, 16))
df = df.set_index('year')
df = df.rename_axis(index='T')

ds = df.to_xarray()
ds = ds.broadcast(

ds.to_zarr('/home/aaron/scratch/iri/data/aaron/fbf-candidate/lesotho/bad-years.zarr')


