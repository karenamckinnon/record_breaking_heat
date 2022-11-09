from helpful_utilities.general import lowpass_butter
import numpy as np
import pandas as pd
import xarray as xr
import os
from datetime import datetime
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from matplotlib import colors
from scipy import stats
from statsmodels.regression.quantile_regression import QuantReg


def get_ghcnd_inventory_dict(var_names, f_inventory):
    """Read in GHCND inventory data to produce dictionary of relevant information

    Parameters
    ----------
    var_names : list
        List of desired GHCND variable names, e.g. (['TMAX', 'TMIN'])
    f_inventory : str
        Local location of GHCND inventory file, ghcnd-inventory.txt

    Returns
    -------
    inventory_dict : dict
        Formatted dictionary with station name, lat, lon, start, end times

    """
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

    return inventory_dict


def get_ghcnd_station_list(var_names, inventory_dict, lats_use, lons_use, yr_start, yr_end):
    """Get list of GHCND stations and their metadata to use for the analysis.

    Parameters
    ----------
    var_names : list
        List of desired GHCND variable names, e.g. (['TMAX', 'TMIN'])
    inventory_dict : dict
        Formatted dictionary with station name, lat, lon, start, end times
    lats_use : tuple
        Lower (S) and upper (N) latitude bounds for the desired domain
    lons_use : tuple
        Lower (W) and upper (E) longitude bounds for the desired domain
    yr_start : int
        First year in which the station should have data
    yr_end : int
        Last year in which the station should have data

    Returns
    -------
    station_list : list
        IDs of relevant stations
    lats : list
        Latitudes of relevant stations, of the same length as station_list
    lons : list
        Longitudes of relevant stations, of the same length as station_list

    """
    for ct_v, this_var in enumerate(var_names):
        station_list = []
        lons = []
        lats = []

        for key in inventory_dict[this_var]:
            this_name = key['name']
            this_start = float(key['start'])
            this_end = float(key['end'])
            this_lon = float(key['lon'])
            this_lat = float(key['lat'])

            do_use = ((this_start <= yr_start) & (this_end >= yr_end)
                      & (this_lon >= lons_use[0]) & (this_lon <= lons_use[1])
                      & (this_lat >= lats_use[0]) & (this_lat <= lats_use[1]))

            if do_use:

                # Pass through any station that has either variable
                if this_name not in station_list:

                    station_list.append(this_name)
                    lons.append(this_lon)
                    lats.append(this_lat)

    return station_list, lats, lons


def get_ghcnd_ds(station_list, var_names, datadir, yr_start, yr_end, subset_years=True):
    """Create dataset of desired GHCND data at desired stations for desired timespan.

    Parameters
    ----------
    station_list : list
        IDs of relevant stations
    var_names : list
        List of desired GHCND variable names, e.g. (['TMAX', 'TMIN'])
    datadir : str
        Local directory for GHCND netcdf files
        Note that these files have been created using scripts/get_ghcnd.py
    yr_start : int
        First year in which the station should have data
    yr_end : int
        Last year in which the station should have data
    subset_years : bool
        Indicate whether dataset should be limited to specified years

    Returns
    -------
    ds_T : xarray.Dataset
        Contains the relevant data for the desired stations
    """

    ds = []
    for this_var in var_names:
        all_data = []
        all_stations = []
        for this_station in station_list:
            fname = '%s/%s_%s.nc' % (datadir, this_station, this_var)
            if not os.path.isfile(fname):
                continue
            this_da = xr.open_dataarray(fname)
            all_data.append(this_da)
            all_stations.append(this_station)

        all_data = xr.concat(all_data, dim='station')
        all_data['station'] = list(all_stations)
        all_data = all_data.rename(this_var)

        ds.append(all_data)

    ds_T = xr.merge(ds)
    if subset_years:
        ds_T = ds_T.sel(time=slice('%04i' % yr_start, '%04i' % yr_end))

    return ds_T


def get_EC_list(datadir, f_inv, lats_use, lons_use, yr_start, yr_end):
    """Get list of EC stations and their metadata to use for the analysis.
    Note that you can bulk download EC data via
    https://dd.weather.gc.ca/climate/observations/

    Parameters
    ----------
    datadir : str
        Local directory for EC files
    lats_use : tuple
        Lower (S) and upper (N) latitude bounds for the desired domain
    lons_use : tuple
        Lower (W) and upper (E) longitude bounds for the desired domain
    yr_start : int
        First year in which the station should have data
    yr_end : int
        Last year in which the station should have data

    Returns
    -------
    these_ids : list
        IDs for relevant EC stations in domain
    """

    inv = pd.read_csv('%s/%s' % (datadir, f_inv), header=2)
    has_data = ((inv['DLY First Year'] <= yr_start) & (inv['DLY Last Year'] >= yr_end))

    in_domain = ((inv['Latitude (Decimal Degrees)'] >= lats_use[0]) &
                 (inv['Latitude (Decimal Degrees)'] <= lats_use[1]) &
                 (inv['Longitude (Decimal Degrees)'] >= lons_use[0]) &
                 (inv['Longitude (Decimal Degrees)'] <= lons_use[1]))

    inv_use = inv.loc[has_data & in_domain]
    these_ids = inv_use['Climate ID'].values

    return these_ids


def get_ISD_list(datadir, f_inv, lats_use, lons_use, yr_start, yr_end):
    """Get list of ISD stations and their metadata to use for the analysis.
    ISD data available at https://www1.ncdc.noaa.gov/pub/data/noaa/

    Parameters
    ----------
    datadir : str
        Local directory for ISD files
    lats_use : tuple
        Lower (S) and upper (N) latitude bounds for the desired domain
    lons_use : tuple
        Lower (W) and upper (E) longitude bounds for the desired domain
    yr_start : int
        First year in which the station should have data
    yr_end : int
        Last year in which the station should have data

    Returns
    -------
    inv_use : list
        IDs and metadata for relevant ISD stations in domain
    """

    inv = pd.read_csv('%s/%s' % (datadir, f_inv))

    yrs_begin = np.array([int(str(t)[:4]) for t in inv['BEGIN']])
    yrs_end = np.array([int(str(t)[:4]) for t in inv['END']])

    has_data = (yrs_begin <= yr_start) & (yrs_end >= yr_end)

    in_domain = ((inv['LAT'] >= lats_use[0]) &
                 (inv['LAT'] <= lats_use[1]) &
                 (inv['LON'] >= lons_use[0]) &
                 (inv['LON'] <= lons_use[1]))

    inv_use = inv.loc[in_domain & has_data]
    return inv_use


def average_resample_ISD(isd_dir, isd_inventory, lats_PNW, lons_PNW,
                         start_year_ISD_min, start_year_ISD_max, end_year_ISD):
    """
    Average ISD data to hourly then calculate daily mean, max, and min.

    Parameters
    ----------
    isd_dir : str
        Parent location of downloaded ISD data
    isd_inventory : str
        Filename for inventory of stations
    lats_PNW : tuple
        Latitude range for domain (lower, upper)
    lons_PNW : tuple
        Longitude range for domain (-180, 180) (westmost, eastmost)
    start_year_ISD_min : int
        Remove ISD data before this year
    start_year_ISD_max : int
        Demand that station data starts by this year
    end_year_ISD : int
        Remove ISD data after this year

    Returns
    -------
    return_names : list
        Names of potential netcdf files (some may not exist if insufficient data)

    """

    # get stations that start at or before the later start year
    isd_names = get_ISD_list(isd_dir, isd_inventory, lats_PNW, lons_PNW, start_year_ISD_max, end_year_ISD)
    return_names = []
    for this_row in isd_names.iterrows():
        id_str = '%s-%s' % (this_row[1]['USAF'], this_row[1]['WBAN'])

        fname = '%s/csv/%s_hourly.csv' % (isd_dir, id_str)
        nc_fname = '%s/%s.nc' % (isd_dir, id_str)
        return_names.append(nc_fname)

        # If file is available and we have not made the netcdf file
        if (os.path.isfile(fname) & (not os.path.isfile(nc_fname))):
            print(id_str)
            this_df = pd.read_csv(fname)
            this_df = this_df.rename(columns={'Unnamed: 0': 'datetime'})
            dates_list = [np.datetime64(datetime.strptime(date, '%Y-%m-%d %H:%M:%S')) for date in this_df['datetime']]

            da = xr.DataArray(this_df['TMP'], dims='time', coords={'time': np.array(dates_list)})

            da = da.sel({'time': (da['time.year'] >= start_year_ISD_min) & (da['time.year'] <= end_year_ISD)})

            this_start_year = da['time.year'].min()
            this_end_year = da['time.year'].max()

            # Case where the data starts early, but then is missing in the middle
            if (this_start_year > (start_year_ISD_max)) | (this_end_year < end_year_ISD):
                continue

            # Interpolate to hourly (original data has variable time stamps)
            da_interp = da.resample(time='1H').mean()

            # count number of samples in day
            nsamples = da_interp.resample(time='1D').count()

            tavg = da_interp.resample(time='1D').mean()
            tmin = da_interp.resample(time='1D').min()
            tmax = da_interp.resample(time='1D').max()

            ds = xr.merge(({'TAVG': tavg}, {'TMAX': tmax}, {'TMIN': tmin}, {'N': nsamples}))

            ds.to_netcdf(nc_fname)

    return return_names


def get_GMT(lowpass_freq=1/10, gmt_fname='/home/data/BEST/Land_and_Ocean_complete.txt', butter_order=3):
    """Load and low-pass filter GMT time series using a Butterworth filter.

    Parameters
    ----------
    lowpass_freq : float
        Desired cutoff frequency for Butterworth filter (in 1/yr)
    gmt_fname : str
        Local location of GMT time series from BEST
    butter_order : int
        Desired order for Butterworth filter
    """

    gmt_fname = gmt_fname
    gmt_data = pd.read_csv(gmt_fname, comment='%', header=None, delim_whitespace=True).loc[:, :2]
    gmt_data.columns = ['year', 'month', 'GMTA']

    stop_idx = np.where(gmt_data['year'] == gmt_data['year'][0])[0][12] - 1
    gmt_data = gmt_data.loc[:stop_idx, :]

    # Perform lowpass filtering
    # using 1/10 years to avoid large underestimates of trend at the end of the record
    time_gmt = pd.date_range(start='%04i-%02i' % (gmt_data['year'][0], gmt_data['month'][0]),
                             periods=len(gmt_data), freq='M')

    if lowpass_freq is not None:
        gmt_smooth = lowpass_butter(12, lowpass_freq, butter_order, gmt_data['GMTA'].values)
        gmt_data = gmt_data.assign(GMTA_lowpass=gmt_smooth)
        da_gmt = xr.DataArray(gmt_data['GMTA_lowpass'], dims={'time'}, coords={'time': time_gmt})
    else:
        da_gmt = xr.DataArray(gmt_data['GMTA'], dims={'time'}, coords={'time': time_gmt})

    return da_gmt


def fit_seasonal_trend(da, varname, nseasonal, ninteract, lastyear=2020, return_beta=False):
    """
    Parameters
    ----------
    da : xr.DataArray
        Data to fit
    varname : str
        Name of variable being fit
    nseasonal : int
        Number of seasonal harmonics to use
    ninteract : int
        Number of harmonics that can change with climate change
    lastyear : int
        The last year of data to use when fitting the model. Allows us to exclude 2021.
    return_beta : bool
        Whether to return the associated regression coefficients.

    Returns
    -------
    ds_fitted : xr.Dataset
        Dataset containing original data, fitted data, and residual
    """

    # number of predictors
    npred = 2 + 2*nseasonal + 2*ninteract  # seasonal harmonics + intercept + trend + seasonal x trend
    nt = len(da.time)

    # fit on this data only
    da_fit = da.sel(time=slice('%04i' % lastyear)).copy()

    # create design matrix
    # seasonal harmonics
    doy = da['time.dayofyear']
    omega = 1/365.25

    #  lobal mean temperature time series as a stand-in for climate change in the regression model
    da_gmt = get_GMT()
    # resample GMT to daily, and match data time stamps
    da_gmt = da_gmt.resample(time='1D').interpolate('linear')
    cc = da_gmt.sel(time=da['time'])
    cc -= np.mean(cc)

    X = np.empty((npred, nt))
    X[0, :] = np.ones((nt, ))
    X[1, :] = cc
    for i in range(nseasonal):
        s = np.exp(2*(i + 1)*np.pi*1j*omega*doy)
        X[(2 + 2*i):(2 + 2*(i+1)), :] = np.vstack((np.real(s), np.imag(s)))

    current_count = 2 + 2*nseasonal
    for i in range(ninteract):
        s = np.exp(2*(i + 1)*np.pi*1j*omega*doy)
        X[(current_count + 2*i):(current_count + 2*(i+1)), :] = np.vstack((np.real(s), np.imag(s)))*cc.values

    # pull out the part of the design matrix for model fitting
    past_idx = np.isin(da.time, da_fit.time)
    X_fit = X[:, past_idx]

    X_mat = np.matrix(X).T
    X_fit_mat = np.matrix(X_fit).T

    if 'station' in da.coords:  # station data, will have missing values, so need to loop through
        ds_fitted = []
        ds_residual = []

        for this_station in da.station:
            this_X = X_fit_mat.copy()
            this_y = da_fit.sel({'station': this_station}).values.copy()
            has_data = ~np.isnan(this_y)

            if np.isnan(this_y).all():
                continue

            this_y = this_y[has_data]
            this_X = this_X[has_data, :]
            this_y = np.matrix(this_y).T

            # fit on data ending on lastyear
            beta = np.linalg.multi_dot(((np.dot(this_X.T, this_X)).I, this_X.T, this_y))

            # predict on full dataset
            yhat = np.dot(X_mat, beta)

            yhat = np.array(yhat).flatten()
            residual = da.sel({'station': this_station}).copy() - yhat

            ds_fitted.append(da.sel({'station': this_station}).copy(data=yhat))
            ds_residual.append(residual)

        ds_fitted = xr.concat(ds_fitted, dim='station')
        ds_fitted = ds_fitted.to_dataset(name='%s_fit' % varname)
        ds_fitted['%s_residual' % varname] = xr.concat(ds_residual, dim='station')

    else:  # reanalysis
        s = da_fit.shape
        if len(s) == 3:
            nt_fit = s[0]
            nlat = s[1]
            nlon = s[2]
        elif len(s) == 1:  # case where we are considering one gridbox only
            nt_fit = s
            nlat = 1
            nlon = 1
        vals = da_fit.values.reshape((nt_fit, nlat*nlon))
        y_mat = np.matrix(vals)

        # fit on data ending on lastyear
        beta = np.linalg.multi_dot(((np.dot(X_fit_mat.T, X_fit_mat)).I, X_fit_mat.T, y_mat))

        # predict on full dataset
        yhat = np.dot(X_mat, beta)
        ds_fitted = da.copy(data=np.array(yhat).reshape((nt, nlat, nlon))).to_dataset(name='%s_fit' % varname)

        residual = da - ds_fitted['%s_fit' % varname]
        ds_fitted['%s_residual' % varname] = residual

    # Repetitive but helpful
    ds_fitted['%s_full' % varname] = ds_fitted['%s_residual' % varname] + ds_fitted['%s_fit' % varname]

    if return_beta:
        return ds_fitted, beta
    else:
        return ds_fitted


def plot_PWN(ax, extent, plotcrs, datacrs, cmap,
             da_cf, levels_cf, extend_cf, add_colorbar, da_c, levels_c,
             scatter_x, scatter_y, scatter_c, scatter_s, bounds_scatter,
             **kwargs):
    """Make flexible plot of data in the PNW domain. Allows for filled contour, empty contour, and scatter.

    kwards include:
        marker : for scatter plot
        lats_box : latitudes if you want to draw a box
        lons_box : longitudes if you want to draw a box
        title : string to include as a title
        title_size : size of title font (required if including title)
        ocean_top : bool for whether the ocean should be plotted on top in gray

    """

    ax.set_extent(extent, crs=ccrs.PlateCarree())

    if 'marker' in kwargs.keys():
        marker = kwargs['marker']
    else:
        marker = 'o'

    # Set up contour plot
    if da_cf is not None:
        cf = da_cf.plot.contourf(ax=ax, levels=levels_cf, cmap=cmap,
                                 add_colorbar=add_colorbar, zorder=1, transform=datacrs,
                                 extend=extend_cf)
    else:
        cf = None

    if scatter_x is not None:
        # Set up scatter plot
        norm = colors.BoundaryNorm(bounds_scatter, cmap.N)

        sc = ax.scatter(scatter_x,
                        scatter_y,
                        s=scatter_s,
                        c=scatter_c,
                        norm=norm,
                        cmap=cmap,
                        marker=marker,
                        transform=datacrs,
                        zorder=5, edgecolor='k')
    else:
        sc = None

    if da_c is not None:
        cp = da_c.plot.contour(ax=ax,
                               levels=levels_c,
                               transform=datacrs,
                               zorder=5,
                               colors='k')
    else:
        cp = None

    if ('lats_box' in kwargs.keys()) & ('lons_box' in kwargs.keys()):
        lats_box = kwargs['lats_box']
        lons_box = kwargs['lons_box']
        ax.plot([lons_box[0], lons_box[0]],
                [lats_box[0], lats_box[1]],
                color='k',
                transform=datacrs, zorder=4)

        ax.plot([lons_box[1], lons_box[1]],
                [lats_box[0], lats_box[1]],
                color='k',
                transform=datacrs, zorder=4)

        ax.plot([lons_box[0], lons_box[1]],
                [lats_box[0], lats_box[0]],
                color='k',
                transform=datacrs, zorder=4)

        ax.plot([lons_box[0], lons_box[1]],
                [lats_box[1], lats_box[1]],
                color='k',
                transform=datacrs, zorder=4)

    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'], fontsize=kwargs['title_size'])
    else:
        ax.set_title('')

    states_provinces = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='50m',
            facecolor='none')

    if 'ocean_top' in kwargs.keys():
        ax.add_feature(cfeature.OCEAN, color='lightgray', zorder=4)
    else:
        ax.add_feature(cfeature.OCEAN, color='lightgray', zorder=0)
    ax.add_feature(cfeature.LAKES, color='gray', zorder=2)
    ax.add_feature(cfeature.COASTLINE, color='black', zorder=4)
    ax.add_feature(cfeature.BORDERS, color='black', zorder=4)
    ax.add_feature(states_provinces, edgecolor='gray', zorder=4)

    return cf, cp, sc


def calc_station_stats(ds_fitted, varnames, doy_start, doy_end, last_year=2020):
    """Calculate skewness, kurtosis, and standard deviation for each station time series.
    It is necessary to loop through because scipy.stats cannot handle missing data.

    Parameters
    ----------
    ds_fitted : xarray.Dataset
        Contains temperature anomalies from station data
    varnames : list
        List of variable names in ds_fitted
    doy_start : int
        First day of year of desired season
    doy_end : int
        Last day of year of desired season
    last_year : int
        The last year to include in the estimation of the stats. Default is 2020 so that 2021 is not included.

    Returns
    -------
    ds_S : xarray.Dataset
        The skewness of each variable
    ds_K : xarray.Dataset
        The kurtosis of each variable
    ds_sigma : xarray.Dataset
        The standard deviation of each variable
    ds_rho1 : xarray.Dataset
        The lag-1 day autocorrelation of each variable
    """

    time_idx = (ds_fitted.time.dt.dayofyear >= doy_start) & (ds_fitted.time.dt.dayofyear <= doy_end)

    ds_S = []
    ds_K = []
    ds_sigma = []
    ds_rho1 = []

    for this_var in varnames:

        S = []
        K = []
        sigma = []
        rho1 = []

        for this_station in ds_fitted.station:

            this_ts = ds_fitted['%s_residual' % this_var]
            # for autocorrelation:
            this_ts_lag1 = this_ts.copy().roll(time=-1, roll_coords=False)

            this_ts = this_ts.sel(station=this_station, time=time_idx).sel(time=slice('%04i' % last_year))
            this_ts_lag1 = this_ts_lag1.sel(station=this_station,
                                            time=time_idx).sel(time=slice('%04i' % last_year))

            has_data = ~np.isnan(this_ts)
            this_ts = this_ts[has_data]
            S.append(stats.skew(this_ts))
            K.append(stats.kurtosis(this_ts))
            sigma.append(np.std(this_ts))
            rho1.append(xr.corr(this_ts, this_ts_lag1, dim='time').values)

        da_S = ds_fitted['%s_residual' % this_var][:, 0].copy(data=S).rename('S_%s' % this_var)
        da_K = ds_fitted['%s_residual' % this_var][:, 0].copy(data=K).rename('K_%s' % this_var)
        da_sigma = ds_fitted['%s_residual' % this_var][:, 0].copy(data=sigma).rename('sigma_%s' % this_var)
        da_rho1 = ds_fitted['%s_residual' % this_var][:, 0].copy(data=rho1).rename('rho1_%s' % this_var)

        ds_S.append(da_S)
        ds_K.append(da_K)
        ds_sigma.append(da_sigma)
        ds_rho1.append(da_rho1)

    ds_S = xr.merge(ds_S)
    ds_K = xr.merge(ds_K)
    ds_sigma = xr.merge(ds_sigma)
    ds_rho1 = xr.merge(ds_rho1)

    return ds_S, ds_K, ds_sigma, ds_rho1


def fit_qr_trend(da, doy_start, doy_end, qs_to_fit, nboot, max_iter=10000, lastyear=2020,
                 gmt_fname='/home/data/BEST/Land_and_Ocean_complete.txt', lowpass_freq=1/10, butter_order=3):
    """Fit a quantile regression model with GMT as covariate.

    Parameters
    ----------
    da : xr.DataArray
        Contains desired variable as a function of station and time
    doy_start : int
        Day of year of beginning of season being fit
    doy_end : int
        Day of year of end of the season being fit
    qs_to_fit : np.array
        Array of quantiles to fit (independently - noncrossing is not enforced)
    nboot : int
        Number of times to bootstrap data (block size of one year) and refit QR model
    max_iter : int
        Maximum number of iterations for QuantReg model
    lastyear : int
        Last year to calculate trend with
    gmt_fname : str
        Local location of GMT time series from BEST
    lowpass_freq : float
        Desired cutoff frequency for Butterworth filter (in 1/yr)
    butter_order : int
        Desired order for Butterworth filter

    Returns
    -------
    ds_QR : xr.Dataset
        Contains all quantile regression trends and pvals, as well as bootstrapped trends

    """

    # fit on this data only
    time_idx = (da.time.dt.dayofyear >= doy_start) & (da.time.dt.dayofyear <= doy_end)
    this_da = da.sel(time=time_idx).sel(time=slice('%04i' % lastyear)).copy()

    # global mean temperature time series as a stand-in for climate change in the regression model
    da_gmt = get_GMT(lowpass_freq=lowpass_freq, gmt_fname=gmt_fname, butter_order=butter_order)
    # resample GMT to daily, and match data time stamps
    da_gmt = da_gmt.resample(time='1D').interpolate('linear')
    cc = da_gmt.sel(time=this_da['time'])
    cc -= np.mean(cc)

    beta_qr = np.nan*np.ones((len(this_da.station), len(qs_to_fit), 2))
    pval_qr = np.nan*np.ones((len(this_da.station), len(qs_to_fit)))
    beta_qr_boot = np.nan*np.ones((len(this_da.station), len(qs_to_fit), nboot))

    np.random.seed(123)
    for station_count, this_station in enumerate(this_da.station):
        if station_count % 100 == 0:
            print('%i/%i' % (station_count, len(this_da.station)))

        this_x = cc
        this_y = this_da.sel(station=this_station)

        pl = ~np.isnan(this_y)
        if np.sum(pl) == 0:  # case of no data
            continue

        this_x_vec = this_x[pl].values
        this_y_vec = this_y[pl].values

        # Add jitter since data is rounded to 0.1
        half_width = 0.05
        jitter = 2*half_width*np.random.rand(len(this_y_vec)) - half_width
        this_y_vec += jitter

        this_x_vec = np.vstack((np.ones(len(this_x_vec)), this_x_vec)).T

        model = QuantReg(this_y_vec, this_x_vec)

        for ct_q, q in enumerate(qs_to_fit):
            mfit = model.fit(q=q, max_iter=max_iter)
            beta_qr[station_count, ct_q, :] = mfit.params
            pval_qr[station_count, ct_q] = mfit.pvalues[-1]

        # Bootstrap with block size of one year to assess significance of differences
        yrs = np.unique(this_y['time.year'])
        for kk in range(nboot):

            new_yrs = np.random.choice(yrs, size=len(yrs))
            x_boot = []
            y_boot = []
            for yy in new_yrs:
                x_boot.append(this_x.sel(time=slice('%04i' % yy, '%04i' % yy)))
                y_boot.append(this_y.sel(time=slice('%04i' % yy, '%04i' % yy)))

            x_boot = xr.concat(x_boot, dim='time')
            y_boot = xr.concat(y_boot, dim='time')

            pl = ~np.isnan(y_boot)
            if np.sum(pl) == 0:  # case of no data
                continue
            this_x_vec = x_boot[pl].values
            this_y_vec = y_boot[pl].values

            # Add jitter since data is rounded to 0.1
            jitter = 2*half_width*np.random.rand(len(this_y_vec)) - half_width
            this_y_vec += jitter

            this_x_vec = np.vstack((np.ones(len(this_x_vec)), this_x_vec)).T

            model = QuantReg(this_y_vec, this_x_vec)

            for ct_q, q in enumerate(qs_to_fit):
                mfit = model.fit(q=q, max_iter=max_iter)
                beta_qr_boot[station_count, ct_q, kk] = mfit.params[-1]

    ds_QR = xr.Dataset(data_vars={'beta_QR': (('station', 'qs', 'order'), beta_qr),
                                  'pval_QR': (('station', 'qs'), pval_qr),
                                  'beta_QR_boot': (('station', 'qs', 'sample'), beta_qr_boot)},
                       coords={'station': da.station,
                               'qs': qs_to_fit,
                               'sample': np.arange(nboot),
                               'lat': da.lat,
                               'lon': da.lon,
                               'order': np.arange(2)})

    return ds_QR
