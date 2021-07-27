import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

RR_MRG_PATH = Path("/Data/data23/NMA_Ethiopia_v7/ALL_NEW/Rainfall/daily/")
RR_MRG_FILE = list(RR_MRG_PATH.glob("rr_mrg_20060503_ALL.nc"))
RR_MRG = xr.open_dataset("/Data/data23/NMA_Ethiopia_v7/ALL_NEW/Rainfall/daily/rr_mrg_20060503_ALL.nc")

RR_MRG
