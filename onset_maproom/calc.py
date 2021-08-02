import os
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import pyaconf

CONFIG = pyaconf.load(os.environ["CONFIG"])

DR_PATH = CONFIG["daily_rainfall_path"]

#Tools to read Zarr version of daily ENACTS rainfall data

RR_MRG_ZARR = Path(DR_PATH)

def read_zarr_data(zarr_path):
  zarr_data = xr.open_zarr(zarr_path)
  return zarr_data

#Setting up the stage to make an onset_date function

params_onset = {
  "earlyStart": "1 Feb",
  "searchDays": 60,
  "wetThreshold": 0,
  "runningDays": 3,
  "runningTotal": 20,
  "minRainyDays": 1,
  "dryDays": 7,
  "drySpell": 21,
}

def pyonset_date(dailyRain, params):
  rr_mrg = read_zarr_data(dailyRain)
  this_onset_date = rr_mrg
  return this_onset_date

#Dummy to test that the reading works
#that_onset_date = pyonset_date(RR_MRG_ZARR, params_onset)
#print(that_onset_date)
