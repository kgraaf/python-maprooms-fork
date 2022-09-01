# Convert rain gauge data:
# NB: the pandas read_excel function depends on openpyxl.

import pandas as pd
import xarray as xr
import cftime

fname = "/data/aaron/fbf/original-data/niger/Cumul_juine_juiy_monthly_rainfall_1991_2022.xlsx"
data = pd.read_excel(fname, sheet_name="array and analysis", usecols="AQ", skiprows=42, nrows=32)
data_fixed = pd.read_excel(fname, sheet_name="array and analysis", usecols="AS", skiprows=42, nrows=32)
data['Average'][data.index[-1]] = data_fixed['Unnamed: 44'][data_fixed.index[-1]]
t = pd.read_excel(fname, sheet_name="array and analysis", usecols="AR", skiprows=42, nrows=32)
T = t.apply(lambda y: cftime.datetime(y, 8, 16, calendar='360_day'), axis=1)
xr.Dataset(data.rename(T)).rename_dims({'dim_0': 'T'}).rename_vars({'dim_0': 'T'}).to_zarr("niger-station-spi-jj.zarr")
