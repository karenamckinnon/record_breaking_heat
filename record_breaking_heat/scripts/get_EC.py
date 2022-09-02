# Code to process Env Canada data
# The data should first be downloaded with the get_BC_data.sh script
# Stations are identified for download using the Station ID but then saved with their Climate ID

from glob import glob
import pandas as pd
from subprocess import check_call

datadir = '/home/data/EnvCanada'
f_inv = 'EnvCanada_station_inventory.csv'

start_year = 1925
end_year = 2021

inv = pd.read_csv('%s/%s' % (datadir, f_inv), header=2)

lats_PNW = 43, 57
lons_PNW = -123, -115
tmpdir = '%s/tmp' % datadir
savedir = '%s/csv' % datadir

cmd = 'mkdir -p %s' % savedir
check_call(cmd.split())
cmd = 'mkdir -p %s' % tmpdir
check_call(cmd.split())

has_data = ((inv['DLY First Year'] <= start_year) & (inv['DLY Last Year'] >= end_year) &
            (inv['Province'] == 'BRITISH COLUMBIA'))

in_domain = ((inv['Latitude (Decimal Degrees)'] >= lats_PNW[0]) &
             (inv['Latitude (Decimal Degrees)'] <= lats_PNW[1]) &
             (inv['Longitude (Decimal Degrees)'] >= lons_PNW[0]) &
             (inv['Longitude (Decimal Degrees)'] <= lons_PNW[1]))

inv_use = inv.loc[has_data & in_domain]

these_stations = inv_use['Station ID'].values
these_ids = inv_use['Climate ID'].values

keepcols = ['Longitude (x)', 'Latitude (y)', 'Station Name', 'Climate ID',
            'Date/Time', 'Max Temp (°C)', 'Min Temp (°C)', 'Mean Temp (°C)']

for station in these_ids:
    savename = '%s/EC_%s.csv' % (savedir, station)
    files = sorted(glob('%s/*%s*.csv' % (tmpdir, station)))
    all_df = []
    for f in files:
        df = pd.read_csv(f)
        df = df[keepcols]
        df.columns = ['lat', 'lon', 'name', 'id', 'date', 'TX', 'TN', 'Tavg']
        all_df.append(df)
    all_df = pd.concat(all_df).reset_index()

    all_df.to_csv(savename, index=False)
