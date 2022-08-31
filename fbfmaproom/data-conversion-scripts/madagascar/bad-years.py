import cftime
import pandas as pd

df = pd.read_csv('MadaBadYear4Tool0622Workshp - MadaBadYear4Tool0622Workshp.csv')
df = df.set_index(df['Year'].apply(lambda x: cftime.Datetime360Day(x, 11, 16)).rename('T'))
df = df.drop('Year', axis='columns')
df = df.rename({'Rank': 'rank'}, axis='columns')
ds = df.to_xarray()
ds.to_zarr('/home/aaron/scratch/iri/data/aaron/fbf-candidate/madagascar/bad-years-ond.zarr')
