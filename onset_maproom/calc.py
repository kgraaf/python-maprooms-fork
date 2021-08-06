import numpy as np
import pandas as pd
import xarray as xr

def read_zarr_data(zarr_path):
  zarr_data = xr.open_zarr(zarr_path)
  return zarr_data

#Onset Date function

def OnsetDate_dummy(dailyRain, params):
  onset_date_dummy = xr.Dataset()
  onset_date_dummy["onset_date"] = dailyRain.precip[
    (dailyRain['time'].dt.day==int(params["earlyStart"].partition(" ")[0]))
    &
    (dailyRain['time'].dt.strftime("%b")==params["earlyStart"].partition(" ")[2])
  ]
  onset_date_dummy["onset_date"] = xr.DataArray(
    data=np.random.randint(0, high=params["searchDays"], size=onset_date_dummy["onset_date"].shape).astype('timedelta64[D]'),
    dims=onset_date_dummy["onset_date"].dims,
    coords=onset_date_dummy["onset_date"].coords,
    attrs=dict(
        description="Onset Date",
    ),
  )
  onset_date_dummy["onset_date"] = onset_date_dummy["time"] + onset_date_dummy["onset_date"]  
  return onset_date_dummy
