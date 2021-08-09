import numpy as np
import pandas as pd
import xarray as xr

def read_zarr_data(zarr_path):
  zarr_data = xr.open_zarr(zarr_path)
  return zarr_data

#Onset Date function

def onset_date(dailyRain, early_start, search_days, rainy_day, running_days, running_total, min_rainy_days, dry_days, dry_spell):
  onset_date = xr.Dataset()
  onset_date["onset_date"] = dailyRain.precip[
    (dailyRain['T'].dt.day==int(early_start.partition(" ")[0]))
    &
    (dailyRain['T'].dt.strftime("%b")==early_start.partition(" ")[2])
  ]
  onset_date["onset_date"] = xr.DataArray(
    data=np.random.randint(0, high=search_days, size=onset_date["onset_date"].shape).astype('timedelta64[D]'),
    dims=onset_date["onset_date"].dims,
    coords=onset_date["onset_date"].coords,
    attrs=dict(
        description="Onset Date",
    ),
  )
  onset_date["onset_date"] = onset_date["T"] + onset_date["onset_date"]  
  return onset_date
