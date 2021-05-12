import xarray as xr
import os

datadir = "/data/aaron/fbf"
base = "http://iridl.ldeo.columbia.edu"

datasets = [
    (
        "rain-malawi",
        "/SOURCES/.NOAA/.NCEP/.CPC/.Merged_Analysis/.monthly/.latest/.ver2/.prcp_est/X/32/36/RANGE/Y/-17/-9/RANGE/T/(Dec-Feb)/seasonalAverage/data.nc",
    ),
    (
        "pnep-malawi",
        "/home/.remic/.IRI/.FD/.NMME_Seasonal_HFcast_Combined/.malawi/.nonexceed/.prob/data.nc",
    ),
    (
        "rain-madagascar",
        "/home/.rijaf/.Madagascar_v3/.ALL/.monthly/.rainfall/.rfe/T/(Dec-Feb)/seasonalAverage/data.nc",
    ),
    (
        "pnep-madagascar",
        "/home/.aaron/.DGM/.Forecast/.Seasonal/.NextGen/.Madagascar_South/.PRCP/.pne/S/(1%20Sep)/(1%20Oct)/(1%20Nov)/VALUES/L/removeGRID/data.nc",
    ),
    (
        "rain-ethiopia",
        "/home/.xchourio/.ACToday/.Ethiopia/.CPT/.NextGen/.MAM_PRCP/.Somali/.NextGen/.History/.obs/data.nc",
    ),
    (
        "pnep-ethiopia",
        "/home/.xchourio/.ACToday/.Ethiopia/.CPT/.NextGen/.MAM_PRCP/.Somali/.NextGen/.FbF/.pne/P//P//percentile/0/5/5/95/NewEvenGRID/replaceGRID/data.nc",
    ),
    (
        'rain-niger',
        '/home/.remic/.DNM/.Forecasts/.NextGen/.PRCPPRCP_CCAFCST_JAS/.NextGen/.FbF/.obs/data.nc'
    ),
    (
        'pnep-niger',
        '/0/home/.aaron/.DNM/.Forecasts/.NextGen/.PRCPPRCP_CCAFCST_JAS/.NextGen/.FbF/.pne/S/(1%20Jan)/(1%20Feb)/(1%20Mar)/(1%20Apr)/(1%20May)/(1%20Jun)/VALUES/data.nc'
    ),
]

for name, urlpath in datasets:
    print(name)
    ncfilepath = "%s/%s.nc" % (datadir, name)
    if os.path.exists(ncfilepath):
        timeopt = "--time-cond %s" % ncfilepath
    else:
        timeopt = ""
    os.system(
        "curl %s -o %s 'http://iridl.ldeo.columbia.edu%s'" % (timeopt, ncfilepath, urlpath)
    )
    assert os.path.exists(ncfilepath)
    zarrpath = "%s/%s.zarr" % (datadir, name)
    if (os.path.exists(zarrpath) and
        os.path.getctime(zarrpath) >= os.path.getctime(ncfilepath)):
        print("Zarr already exists")
    else:
        print("Converting to zarr")
        ds = xr.open_dataset(ncfilepath, decode_times=False)
        ds.to_zarr(zarrpath)
