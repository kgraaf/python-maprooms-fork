import numpy as np
import pandas as pd
import xarray as xr

def read_zarr_data(zarr_path):
  zarr_data = xr.open_zarr(zarr_path)
  return zarr_data

#Onset Date function

def onset_date(daily_rain, early_start_day, early_start_month, search_days, rainy_day, running_days, running_total, min_rainy_days, dry_days, dry_spell, time_coord="T"):
  """Fonction reproducing Ingrid onsetDate function
  http://iridl.ldeo.columbia.edu/dochelp/Documentation/details/index.html?func=onsetDate
  with the exception that:
  output is a random deltatime rather than an actual onset
  earlyStart input is now 2 arguments: day and month (as opposed to 1)
  output is now the timedelta with each year's earlyStart (as opposed to the date itself)
  """
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
