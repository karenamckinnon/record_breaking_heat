import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import xarray as xr


def fit_seasonal_trend_model(ds, this_var, cc_predictor=None, start_year=1960):
    """Fit a basic statistical model to data: linear trend in time, 5 seasonal bases, annual basis can change in time.
    Designed for a single time series (a la station data).

    Parameters
    ----------
    ds : xr.Dataset
        Contains data that is indexed by "this_var"
    this_var : str
        The variable to analyze, e.g. TMAX or TMIN
    cc_predictor : None or xr.DataArray
        Predictor for climate change. If None, linear in time. Otherwise, uses passed through array
    start_year : int
        The first year of data to fit model

    Returns
    -------
    this_ds : xr.Dataset
        Original data and fitted model
    """

    # fit model on data prior to 2020
    ds_T_prior = ds.sel(time=slice('%04i' % start_year, '2020'))

    if cc_predictor is None:
        # "climate change" is linear in time
        year_frac = ds_T_prior['time.year'] + ds_T_prior['time.dayofyear']/365
        mu_year_frac = np.mean(year_frac)
        year_frac -= mu_year_frac
        cc = year_frac
    else:
        cc = cc_predictor.sel(time=slice('%04i' % start_year, '2020')).values
        cc_mu = np.mean(cc)
        cc -= cc_mu
    doy = ds_T_prior['time.dayofyear']
    omega = 1/365
    # annual bases
    s1 = np.exp(2*np.pi*1j*omega*doy)
    s2 = np.exp(4*np.pi*1j*omega*doy)
    s3 = np.exp(6*np.pi*1j*omega*doy)
    s4 = np.exp(8*np.pi*1j*omega*doy)
    s5 = np.exp(10*np.pi*1j*omega*doy)

    # set up predictors
    this_df = ds_T_prior[this_var].to_dataframe()
    this_df['cc'] = cc
    this_df['real_s1'] = np.real(s1)
    this_df['imag_s1'] = np.imag(s1)
    this_df['real_s2'] = np.real(s2)
    this_df['imag_s2'] = np.imag(s2)
    this_df['real_s3'] = np.real(s3)
    this_df['imag_s3'] = np.imag(s3)
    this_df['real_s4'] = np.real(s4)
    this_df['imag_s4'] = np.imag(s4)
    this_df['real_s5'] = np.real(s5)
    this_df['imag_s5'] = np.imag(s5)
    # first annual cycle can change in time
    this_df['real_s1_yr'] = np.real(s1)*cc
    this_df['imag_s1_yr'] = np.imag(s1)*cc

    formula = '%s ~ cc + real_s1 + imag_s1 + real_s2 + imag_s2 + real_s3 + imag_s3' % this_var
    formula += ' + real_s4 + imag_s4 + real_s5 + imag_s5 + real_s1_yr + imag_s1_yr'
    model = smf.ols(formula=formula, data=this_df).fit()
    yhat_prior = model.predict(this_df)

    if cc_predictor is None:
        # Predict for 2021
        time_2021 = ds.sel(time=slice('2021', '2021'))['time']
        time_2021_frac = time_2021['time.year'] + time_2021['time.dayofyear']/365
        time_2021_frac -= mu_year_frac
        cc_2021 = time_2021_frac
    else:
        cc_2021 = cc_predictor.sel(time=slice('2021', '2021')).values
        cc_2021 -= cc_mu
    doy = time_2021['time.dayofyear']
    s1 = np.exp(2*np.pi*1j*omega*doy)
    s2 = np.exp(4*np.pi*1j*omega*doy)
    s3 = np.exp(6*np.pi*1j*omega*doy)
    s4 = np.exp(8*np.pi*1j*omega*doy)
    s5 = np.exp(10*np.pi*1j*omega*doy)

    future_df = pd.DataFrame({'cc': cc_2021})
    future_df['real_s1'] = np.real(s1)
    future_df['imag_s1'] = np.imag(s1)
    future_df['real_s2'] = np.real(s2)
    future_df['imag_s2'] = np.imag(s2)
    future_df['real_s3'] = np.real(s3)
    future_df['imag_s3'] = np.imag(s3)
    future_df['real_s4'] = np.real(s4)
    future_df['imag_s4'] = np.imag(s4)
    future_df['real_s5'] = np.real(s5)
    future_df['imag_s5'] = np.imag(s5)

    future_df['real_s1_yr'] = np.real(s1)*cc_2021
    future_df['imag_s1_yr'] = np.imag(s1)*cc_2021

    yhat_2021 = model.predict(future_df)

    # Combine past and future predictions
    yhat = np.hstack((yhat_prior, yhat_2021))

    # and put in dataset with actual
    this_ds = ds[this_var].sel(time=slice('%04i' % start_year, '2021')).to_dataset()
    this_ds['yhat'] = xr.DataArray(yhat, dims='time', coords={'time': this_ds.time})

    return this_ds
