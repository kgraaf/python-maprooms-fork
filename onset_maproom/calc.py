import numpy as np
import pandas as pd
import xarray as xr

#Tools to read Zarr version of daily ENACTS rainfall data

def read_zarr_data(zarr_path):
  zarr_data = xr.open_zarr(zarr_path)
  return zarr_data

#Onset Date function

def pyonset_date_dummy(dailyRain, params):
  this_onset_date = xr.Dataset()
  this_onset_date['time'] = dailyRain.time[
    (dailyRain['time'].dt.day==int(params["earlyStart"].partition(" ")[0]))
    &
    (dailyRain['time'].dt.strftime("%b")==params["earlyStart"].partition(" ")[2])
  ]
  for i in dailyRain.coords:
    if i != "time":
      this_onset_date[i] = dailyRain[i]
  this_onset_date["onset"] = np.random.randint(0, high=params["searchDays"], size=this_onset_date.shape)

#  this_onset_date = xr.Dataset()
#  this_onset_date["onset"] = dailyRain.precip[
#    (dailyRain['time'].dt.day==int(params["earlyStart"].partition(" ")[0]))
#    &
#    (dailyRain['time'].dt.strftime("%b")==params["earlyStart"].partition(" ")[2])
#  ]
#  this_onset_date["onset"] = np.random.randint(0, high=params["searchDays"], size=this_onset_date["onset"].shape)
#  this_onset_date["onset"] = np.random.default_rng().integers(low=0, high=params["searchDays"], size=this_onset_date.sizes['time'])
  return this_onset_date

