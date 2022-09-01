import cftime
import openpyxl
import pandas as pd
import xarray as xr

wb = openpyxl.load_workbook('Djibouti_bad-years_JAS.xlsx')
sheet = wb['Sheet1']
cols = list(sheet.columns)

years_col = cols[0]
assert years_col[0].value == 'Years'
years = list(map(lambda x: cftime.Datetime360Day(x.value, 8, 16), years_col[1:]))

jas_col = cols[1]
assert jas_col[0].value == 'JAS'
colors = {
    'FFFF0000': 1,  # red (driest)
    'FFFFC000': 2,  # orange
    'FFFFFF00': 3,  # yellow
    'FFBDD6EE': 4,  # light blue
    'FF9CC2E5': 4,  # another light blue
    'FF92D050': 5,  # light green
    'FF00B050': 6,  # dark green (wettest)
}
severities = list(map(lambda x: colors.get(x.fill.fgColor.rgb), jas_col[1:]))

df = pd.DataFrame(index=years, data={'bad': severities})
df.index.name = 'T'

ds = df.to_xarray()
print(ds)
ds.to_zarr('djibouti-bad-years.zarr')
