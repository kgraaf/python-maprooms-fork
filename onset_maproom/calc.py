import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
from pathlib import Path

#Read daily files of daily rainfall data
#Concatenate them agains time index made up from filenames
RR_MRG_PATH = Path("/Data/data23/NMA_Ethiopia_v7/ALL_NEW/Rainfall/daily/")
RR_MRG_FILE = list(sorted(RR_MRG_PATH.glob("rr_mrg_200*_ALL.nc")))
rr_mrg = xr.concat(
  [xr.open_dataset(f) for f in RR_MRG_FILE],
  pd.Index(
    [dt.datetime(
      int(i.stem.partition("rr_mrg_")[2].partition("_")[0][0:4]),
      int(i.stem.partition("rr_mrg_")[2].partition("_")[0][4:6]),
      int(i.stem.partition("rr_mrg_")[2].partition("_")[0][6:8])
    )
    for i in RR_MRG_FILE],
    name="time"
  )
)

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

def pyonset_date(params):
    this_onset_date = params
    return this_onset_date

toto = pyonset_date(params_onset)
print(toto)
