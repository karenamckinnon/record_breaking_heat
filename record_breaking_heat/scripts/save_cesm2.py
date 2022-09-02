# Code to save the relevant files from CESM2 for analysis in the GRL_figures code

import xarray as xr
from glob import glob

# where to save the data
tmp_dir = '/glade/scratch/mckinnon/cesm2'

# location of CESM2-LE on glade
cesm2_daily_dir = '/glade/campaign/cgd/cesm/CESM2-LE/timeseries/atm/proc/tseries/day_1'

# variable to use
this_var = 'TREFHTMX'

# range of latitudes to save
lat1 = 40
lat2 = 70
lats = slice(lat1, lat2)  # upper midlatitudes -- major summer heatwaves including PNW, Siberia

# subseasonal span of time to analyze
doy_start = 166  # June 15 (non leap)
doy_end = 196  # July 15 (non leap)

# need to  run both forcings (uncomment the lower one after running the first)
forcing = 'BHISTsmbb'
start_years = '18500101-18591231'
# forcing = 'BSSP370smbb'
# start_years = '20150101-20241231'

# get landmask from a land file (hack)
land_dir = '/glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/'
land_dir += 'b.e21.B1850.f09_g17.CMIP6-piControl.001/lnd/proc/tseries/month_1'
fname = 'b.e21.B1850.f09_g17.CMIP6-piControl.001.clm2.h0.ACTUAL_IMMOB.000101-009912.nc'
ls_mask = xr.open_dataset('%s/%s' % (land_dir, fname))['landfrac'] == 1

# files span ensemble member and years
tmp = sorted(glob('%s/%s/b.e21.%s.f09_g17.*.%s.nc' % (cesm2_daily_dir, this_var, forcing, start_years)))
macro = [f.split('/')[-1].split('LE2-')[-1].split('.')[0] for f in tmp]
micro = [f.split('/')[-1].split('LE2-')[-1].split('.')[1] for f in tmp]

# issue with different rounding of coordinates
da = xr.open_dataset(tmp[0])[this_var]
ls_mask = ls_mask.assign_coords({'lat': da.lat,
                                 'lon': da.lon})

# # Load in all files, subset to desired domain / time of year, and save
da_all = []
for this_macro, this_micro in zip(macro, micro):
    print('%s, %s' % (this_macro, this_micro))
    fname = ('b.e21.%s.f09_g17.LE2-%s.%s.cam.h1.%s.????????-????????.nc' %
             (forcing, this_macro, this_micro, this_var))
    files = sorted(glob('%s/%s/%s' % (cesm2_daily_dir, this_var, fname)))

    da = xr.open_mfdataset(files)[this_var]

    da = da.sel(lat=lats)
    time_idx = (da.time.dt.dayofyear >= doy_start) & (da.time.dt.dayofyear <= doy_end)
    da = da.sel(time=time_idx)
    da = da.load()
    da = da.where(ls_mask.sel(lat=lats) == 1)
    da.to_netcdf('%s/cesm2_%s_%s_%s-%s.nc' % (tmp_dir, forcing, this_var, this_macro, this_micro))
