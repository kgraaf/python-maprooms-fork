import numpy as np
import pandas as pd
import xarray as xr

def read_zarr_data(zarr_path):
  zarr_data = xr.open_zarr(zarr_path)
  return zarr_data

#Onset Date function

def onset_date(daily_rain, early_start_day, early_start_month, search_days, rainy_day, running_days, running_total, min_rainy_days, dry_days, dry_spell, time_coord="T"):
  onset_date = daily_rain[
    (daily_rain[time_coord].dt.day==early_start_day)
    &
    (daily_rain[time_coord].dt.month==early_start_month)
  ]
  onset_date = xr.DataArray(
    data=np.random.randint(0, high=search_days, size=onset_date.shape).astype('timedelta64[D]'),
    dims=onset_date.dims,
    coords=onset_date.coords,
    attrs=dict(
        description="Onset Date",
    ),
  )
#Tip to get dates from timedelta  early_start_day
#  onset_date = onset_date[time_coord] + onset_date  
  return onset_date

def strftimeb2int(strftimeb):
  strftimeb_all = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12
  }
  strftimebint=strftimeb_all[strftimeb]
  return strftimebint
