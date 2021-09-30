import numpy as np
import pandas as pd
import xarray as xr
import os

ghcnd_dir = '/home/data/GHCND'
f_station_list = '%s/ghcnd-stations.txt' % ghcnd_dir
f_inventory = '%s/ghcnd-inventory.txt' % ghcnd_dir

datadir = '/home/kmckinnon/record_breaking_heat/data'
var_names = (['TMIN', 'TMAX'])

# Pull information from inventory
namestr = [0, 11]
latstr = [12, 20]
lonstr = [21, 30]
varstr = [31, 35]
startstr = [36, 40]
endstr = [41, 45]
for ct_v, this_var in enumerate(var_names):

    with open(f_inventory, 'r') as f:
        name = []
        lon = []
        lat = []
        start = []
        end = []

        for line in f:
            var = line[varstr[0]:varstr[1]]
            if (var == this_var):
                name.append(line[namestr[0]:namestr[1]])  # station name
                lat.append(line[latstr[0]:latstr[1]])  # station latitude
                lon.append(line[lonstr[0]:lonstr[1]])  # station longitude
                start.append(line[startstr[0]:startstr[1]])  # start year of station data
                end.append(line[endstr[0]:endstr[1]])  # end year of station data

        this_dict = [{'name': name, 'lat': lat, 'lon': lon, 'start': start, 'end': end}
                     for name, lat, lon, start, end in zip(name, lat, lon, start, end)]

        if ct_v == 0:
            inventory_dict = {this_var: this_dict}
        else:
            inventory_dict[this_var] = this_dict

# Select stations to use (US only)
yr_start = 1960
yr_end = 2021

for ct_v, this_var in enumerate(var_names):
    station_list = []
    lons = []
    lats = []

    for key in inventory_dict[this_var]:
        this_name = key['name']
        this_start = float(key['start'])
        this_end = float(key['end'])

        if ((this_name[:2] == 'US') & (this_start <= yr_start) & (this_end >= yr_end)):

            # Pass through any station that has TMIN or TMAX
            if this_name not in station_list:

                station_list.append(this_name)
                lons.append(float(key['lon']))
                lats.append(float(key['lat']))

# Get data for each station
# ------------------------------
# Variable   Columns   Type
# ------------------------------
# ID            1-11   Character
# YEAR         12-15   Integer
# MONTH        16-17   Integer
# ELEMENT      18-21   Character
# VALUE1       22-26   Integer
# MFLAG1       27-27   Character
# QFLAG1       28-28   Character
# SFLAG1       29-29   Character
# VALUE2       30-34   Integer
# MFLAG2       35-35   Character
# QFLAG2       36-36   Character
# SFLAG2       37-37   Character
#   .           .          .
#   .           .          .
#   .           .          .
# VALUE31    262-266   Integer
# MFLAG31    267-267   Character
# QFLAG31    268-268   Character
# SFLAG31    269-269   Character
# ------------------------------

# These variables have the following definitions:

# ID         is the station identification code.  Please see "ghcnd-stations.txt"
#            for a complete list of stations and their metadata.
# YEAR       is the year of the record.

# MONTH      is the month of the record.

# ELEMENT    is the element type.   There are five core elements as well as a number
#            of addition elements.

#            The five core elements are:

#            PRCP = Precipitation (tenths of mm)
#            SNOW = Snowfall (mm)
#            SNWD = Snow depth (mm)
#            TMAX = Maximum temperature (tenths of degrees C)
#            TMIN = Minimum temperature (tenths of degrees C)

date_str = pd.date_range(start='%04i-01-01' % yr_start, end='%04i-08-31' % yr_end, freq='D')

yearstr = [11, 15]
monstr = [15, 17]
varstr = [17, 21]
datastr = [21, 269]
data1str = [21, 26]
data2str = [29, 34]

for counter, this_station in enumerate(station_list):
    print('%i/%i' % (counter, len(station_list)))
    this_file = '%s/USHCN/%s.dly' % (ghcnd_dir, this_station)
    data_vec = np.nan*np.ones(len(date_str))

    if os.path.isfile(this_file):

        for this_var in var_names:
            savename = '%s/%s_%s.nc' % (datadir, this_station, this_var)
            if os.path.isfile(savename):
                continue
            with open(this_file, 'r') as f:
                for line in f:
                    if this_var == line[varstr[0]: varstr[1]]:
                        this_year = line[yearstr[0]: yearstr[1]]
                        if ((float(this_year) >= yr_start) and (float(this_year) <= yr_end)):
                            mon = line[monstr[0]: monstr[1]]  # the month of data
                            data = line[datastr[0]: datastr[1]]  # the data

                            days = [data[i*8:i*8+8] for i in np.arange(0, 31, 1)]
                            mflag = [days[i][5] for i in np.arange(31)]  # getting the mflag
                            qflag = [days[i][6] for i in np.arange(31)]  # getting the qflag
                            sflag = [days[i][7] for i in np.arange(31)]  # getting the sflag
                            values = [days[i][:5] for i in np.arange(31)]  # getting the data values
                            values_np = np.array(values).astype(int)  # converting to a numpy array

                            # set missing to NaN
                            is_missing = (values_np == -9999)
                            values_np = values_np.astype(float)
                            values_np[is_missing] = np.nan

                            # removing any that fail the quality control flag or have
                            # L = temperature appears to be lagged with respect to reported hour of observation
                            is_bad = (np.array(qflag) != ' ') | (np.array(mflag) == 'L')
                            values_np[is_bad] = np.nan

                            date_idx = (date_str.month == int(mon)) & (date_str.year == int(this_year))
                            data_vec[date_idx] = values_np[:np.sum(date_idx)]/10  # change to degrees Celsius

            # Save data
            this_da = xr.DataArray(data_vec, dims='time', coords={'time': date_str})
            this_da['lat'] = lats[counter]
            this_da['lon'] = lons[counter]
            this_da.to_netcdf(savename)
