from __future__ import print_function
import os

datadir = '/data/aaron/fbf'
base = 'http://iridl.ldeo.columbia.edu'

datasets = [
    ('rain-noaa.nc', '/SOURCES/.NOAA/.NCEP/.CPC/.Merged_Analysis/.monthly/.latest/.ver2/.prcp_est/data.nc'),
    ('pnep-malawi.nc', '/home/.remic/.IRI/.FD/.NMME_Seasonal_HFcast_Combined/.malawi/.nonexceed/.prob/data.nc'),
    ('rain-madagascar.nc', '/home/.rijaf/.Madagascar_v3/.ALL/.monthly/.rainfall/.rfe/data.nc'),
    ('pnep-madagascar.nc', '/home/.aaron/.DGM/.Forecast/.Seasonal/.NextGen/.Madagascar_South/.PRCP/.pne/L/removeGRID/data.nc'),
    ('rain-ethiopia.nc', '/home/.xchourio/.ACToday/.Ethiopia/.CPT/.NextGen/.MAM_PRCP/.Somali/.NextGen/.History/.obs/'),
    ('pnep-ethiopia.nc', '/home/.xchourio/.ACToday/.Ethiopia/.CPT/.NextGen/.MAM_PRCP/.Somali/.NextGen/.FbF/.pne//L//months/ordered/%5B0.0%5D/NewGRID/addGRID/L//pointwidth/3.0/put/data.nc'),
]

for name, urlpath in datasets:
    print(name)
    filepath = '%s/%s' % (datadir, name)
    if os.path.exists(filepath):
        timeopt = '--time-cond %s' % filepath
    else:
        timeopt = ''
    os.system('curl %s -o %s http://iridl.ldeo.columbia.edu%s' %
           (timeopt, filepath, urlpath))
