import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
from pathlib import Path

RR_MRG_PATH = Path("/Data/data23/NMA_Ethiopia_v7/ALL_NEW/Rainfall/daily/")
RR_MRG_FILE = list(sorted(RR_MRG_PATH.glob("rr_mrg_200001*_ALL.nc")))
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

