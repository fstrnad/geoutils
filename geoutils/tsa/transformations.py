from importlib import reload
import geoutils.utils.general_utils as gut
import geoutils.utils.statistic_utils as sut
from statsmodels.tsa.ar_model import AutoReg
import xarray as xr
import scipy.signal as sig
import numpy as np
from scipy.fftpack import fft, fftfreq
import pandas as pd
import geoutils.tsa.filters as flt
import nitime.algorithms as spectrum
from tqdm import tqdm

from scipy.stats import chi2
# frequency of measurements per day


def compute_fft(ts, freq_m=1,
                window='hannig',
                K=1,
                cutoff=1):
    if isinstance(ts, xr.DataArray):
        data = ts.data
    else:
        data = ts
    data = sut.standardize(data)
    N = len(data)
    T = 1/freq_m  # measurements per day
    if window == 'multitaper':
        NW = int((K+1)/2)
        w, power, nu = spectrum.multi_taper_psd(data,
                                                Fs=T,
                                                NW=NW,
                                                adaptive=True,
                                                jackknife=False)
        Kmax = nu[0] / 2
        lb, ub = chi2conf(Kmax, power)
        results = pd.DataFrame(
            {'freq': w, 'Power': power, 'lb': lb, 'ub': ub})
    else:
        if window == 'blackman':
            window = sig.blackman(N)
        elif window == 'hamming':
            window = sig.hamming(N)
        elif window == 'hannig':
            window = sig.hannig(N)
        else:
            window = 1

        ts_fft = fft(data*window, N)[:N//2]  # only positive values
        power = np.abs(ts_fft)[0:N//2]
        w = fftfreq(N, T)[:N//2]  # The corresponding positive frequencies,
        # w, power = spectrum.periodogram(data*window, Fs=T)
        if cutoff > 1:
            power = flt.apply_butter_filter(power,
                                            cutoff=cutoff)
        var_ps = w * power
        results = pd.DataFrame(
            {'freq': w, 'Power': power, 'var_ps': var_ps, 'period': 1./w})

    return results


# Define a short function to compute confidence bounds.
def chi2conf(K, Sxx=1, ci=.95):
    '''
    Returns confidence bounds computed using chi-square
    distribution.
    Input:
        K (int): Number of tapers
        Sxx (array): Multitaper spectrum (optional)
        ci ([0.5, 1]): Confidence level (optional)
    Output:
        lb, ub: lower and upper bounds
    '''
    ub = 2 * K / chi2.ppf(1 - ci, 2 * K) * Sxx
    lb = 2 * K / chi2.ppf(ci, 2 * K) * Sxx
    return lb, ub


def fft_period(ts, dp=365):
    x = np.arange(len(ts))
    ft_abs = np.abs(np.fft.fft(ts))

    # get the list of frequencies
    num = np.size(x)
    freq = [i / num for i in list(range(num))]
    # get the list of spectrums
    spectrum = ft_abs.real*ft_abs.real+ft_abs.imag*ft_abs.imag
    nspectrum = spectrum/spectrum[0]

    results = pd.DataFrame({'freq': freq, 'nspectrum': nspectrum})
    results['period'] = results['freq'] / (1/dp)

    results['period_round'] = results['period'].round()
    grouped_p = results.groupby('period_round')['nspectrum'].sum()

    return results, grouped_p


def ar1(x, lags=1):
    if type(x) == xr.DataArray:
        x = x.data
    model = AutoReg(x, lags=lags, old_names=False)

    param_names = ['a0', 'a1', 'a2', 'a3', 'a4',
                   'a5', 'a6', 'a7', 'a8', 'a9', 'a10']
    res = model.fit()
    res.summary()
    params = res.params
    param_dict = gut.mk_dict_2_lists(param_names[:len(params)], params)

    return param_dict


def ar1_surrogates(x, N=1000):
    """In an AR(1) model
        x(t) - <x> = \gamma(x(t-1) - <x>) + \alpha z(t) ,
    where <x> is the process mean, \gamma and \alpha are process
    parameters and z(t) is a Gaussian unit-variance white noise.

    Args:
        x (np.ndarraarray): array of the time series.
    """
    reload(sut)
    ar1_dict = ar1(x, lags=1)
    x = sut.standardize(dataset=x)
    T = np.size(x)
    surr_arr = [x]
    # x_shift = np.roll(x, 1)
    for n in tqdm(range(N)):
        surr_ts = [x[0]]
        rand_nums = np.random.normal(0, 1, T)
        for t in range(T-1):
            # or + ar1_dict['a0']
            surr_ts.append(ar1_dict['a1'] * surr_ts[t] + rand_nums[t])
        surr_arr.append(surr_ts)

    return sut.standardize(np.array(surr_arr), axis=1)


def ar1_surrogates_spectrum(x, N=1000, cutoff=1,
                            window=1,
                            fft_prop='Power'):
    """Compute the power spectrum of an AR1 model
    """
    surr_arr = []
    surr_ts_arr = ar1_surrogates(x=x, N=N)
    for surr_ts in surr_ts_arr:
        surr_fft = compute_fft(ts=surr_ts, cutoff=cutoff, window=window)
        surr_arr.append(surr_fft[fft_prop])
    surr_arr = np.array(surr_arr)
    lb = np.quantile(surr_arr[1:], q=0.05, axis=0)  # First is original TS
    ub = np.quantile(surr_arr[1:], q=0.95, axis=0)  # First is original TS
    return {
        'period': surr_fft['period'],
        'freq': surr_fft['freq'],
        'Power': surr_arr[0],
        'surr': surr_arr,
        'lb': lb, 'ub': ub,
        'surr_ts': surr_ts_arr}
