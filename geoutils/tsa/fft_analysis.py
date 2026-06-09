from scipy.fft import irfft
import geoutils.utils.time_utils as tu
from statsmodels.tsa.arima_process import ArmaProcess
from importlib import reload
import geoutils.utils.general_utils as gut
import geoutils.utils.statistic_utils as sut
from statsmodels.tsa.ar_model import AutoReg
import xarray as xr
import scipy.signal as sig
import numpy as np
from scipy.fft import rfft, rfftfreq
import pandas as pd
import geoutils.tsa.filters as flt
import nitime.algorithms as spectrum
from tqdm import tqdm
from scipy.stats import chi2
# frequency of measurements per day
reload(gut)
reload(sut)
reload(tu)
reload(flt)


def compute_fft(ts, freq_m=1,
                window=None,
                K=1,
                cutoff=1,
                stdize=False):
    if isinstance(ts, xr.DataArray):
        data = ts.values
    else:
        data = ts
    if stdize:
        data = sut.standardize(data)

    ts = flt.apply_butter_filter(ts,
                                 cutoff=cutoff)
    T = len(data)  # Duration
    sample_rate = int(freq_m)  # how many samples per Unit of time
    N = sample_rate * T
    if window == 'multitaper':
        NW = int((K+1)/2)
        w, power, nu = spectrum.multi_taper_psd(data,
                                                Fs=T,
                                                NW=NW,
                                                adaptive=True,
                                                jackknife=False)
        Kmax = nu[0] / 2
        lb, ub = chi2conf(Kmax, power)
    else:
        if window == 'blackman':
            window = sig.blackman(N)
        elif window == 'hamming':
            window = sig.hamming(N)
        elif window == 'hannig':
            window = sig.hannig(N)
        elif window is None or window == 'linear':
            gut.myprint('No window size for FFT!', verbose=False)
            window = 1
        else:
            raise ValueError(f'This window does not exist: {window}!')

        ts_fft = rfft(data*window)
        power = np.abs(ts_fft)[1:]  # only positive values
        # The corresponding positive frequencies, # exclude w=0
        w = rfftfreq(N, 1/sample_rate)[1:]
        # w, power = spectrum.periodogram(data*window, Fs=T)
        if cutoff > 1:
            power = flt.apply_butter_filter(power,
                                            cutoff=cutoff)

        var_ps = w * power

    points_per_freq = int(len(w) / (sample_rate/2))

    results = {'freq': w,
               'power': power,
               'var_ps': var_ps,
               'period': 1./w,
               'fft': ts_fft,
               'ppf': points_per_freq,
               }

    return results


def inverse_fft(ts_fft):
    ts = irfft(ts_fft)
    return ts


def revmove_harmonics(ts, num_harmonics=1):
    fft_dict = compute_fft(ts)
    fft = fft_dict['fft']
    sorted_peaks = sut.find_and_sort_peaks(fft)
    for peak in sorted_peaks[:num_harmonics]:
        fft[peak-1: peak+2] = 0

    ifft = inverse_fft(fft)

    return xr.DataArray(data=ifft,
                        dims=ts.dims,
                        coords={'time': ts.time[:-1]},
                        name=ts.name)

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


def fft_by_year(ts, fft_prop='power',
                window='blackman',
                cutoff=1):
    year_arr = tu.split_by_year(ds=ts)
    len_year = len(year_arr[0])
    fft_arr = []
    for year_ts in year_arr:
        this_fft = compute_fft(ts=year_ts, window=window, cutoff=cutoff)
        this_len_year = len(year_ts)
        if len_year != this_len_year:
            fft_arr.append(this_fft[fft_prop][:-1])
        else:
            fft_arr.append(this_fft[fft_prop])
    mean_fft = np.mean(fft_arr, axis=0)
    return {
        'period': this_fft['period'],
        'freq': this_fft['freq'],
        fft_prop: mean_fft,
        'surr': None}


def generate_ar1_surrogates(data, N):
    """
    Generate surrogates of a time series based on the AR(1) model.

    Parameters:
        data (array-like): The original time series data.
        N (int): The number of surrogates to generate.

    Returns:
        array-like: An array of surrogates generated based on the AR(1) model.

    """

    # Estimate AR(1) model parameters
    mu = np.mean(data)
    sigma = np.std(data)
    rho = np.corrcoef(data[:-1], data[1:])[0, 1]

    surrogates = [data]
    T = len(data)
    for _ in tqdm(range(N)):
        # Generate white noise with the same length as the data
        noise = np.random.normal(0, sigma, T)

        # Initialize the surrogate series
        surrogate = np.zeros_like(data)
        surrogate[0] = data[0]

        # Generate the surrogate series based on the AR(1) model
        for i in range(1, T):
            surrogate[i] = mu + rho * (surrogate[i - 1] - mu) + noise[i]

        surrogates.append(surrogate)

    return np.array(surrogates)


def ar1(x, lags=25, verbose=False):
    if type(x) == xr.DataArray:
        x = x.data
    model = AutoReg(x, lags=lags, old_names=False)
    param_names = ['a0', 'a1', 'a2', 'a3', 'a4',
                   'a5', 'a6', 'a7', 'a8', 'a9', 'a10']
    res = model.fit()

    gut.myprint(res.summary(), verbose=verbose)

    params = res.params
    params[0] = 1
    param_dict = gut.mk_dict_2_lists(param_names[:len(params)], params)
    param_dict['model'] = res

    return param_dict


def ar1_surrogates(data, N=1000, lags=10, verbose=False):
    """In an AR(1) model
        x(t) - <x> = \gamma(x(t-1) - <x>) + \alpha z(t) ,
    where <x> is the process mean, \gamma and \alpha are process
    parameters and z(t) is a Gaussian unit-variance white noise.

    Args:
        x (np.ndarraarray): array of the time series.
    """
    reload(sut)

    # data = sut.standardize(dataset=data)
    T = np.size(data)
    surr_arr = [data[:]]
    ar1_dict = ar1(data, lags=lags)
    gut.myprint(ar1_dict['model'].params, verbose=verbose)
    # x_shift = np.roll(x, 1)
    for n in range(N):
        surr_ts = [data[0]]
        rand_nums = np.random.normal(0, 1, T)
        for t in range(T-1):
            # or + ar1_dict['a0']
            surr_ts.append(ar1_dict['a1'] * surr_ts[t] + rand_nums[t])
        surr_arr.append(surr_ts)
    # sur_ts = ar1_dict['model'].predict(start=0, end=len(data)-1, dynamic=False)
    # surr_arr.append(sur_ts[lags:])

    # return sut.standardize(np.array(surr_arr), axis=1)
    # AR_object = ArmaProcess(
    #     np.array(ar1_dict['model'].params), ma=np.array([1]))
    # for n in range(N):
    #     simulated_data = AR_object.generate_sample(nsample=T)
    #     surr_arr.append(simulated_data)
    return {'surrogates': np.array(surr_arr),
            'model': ar1_dict
            }


def ar1_surrogates_spectrum(ts, N=1000, cutoff=1,
                            window='blackman',
                            fft_prop='power',
                            lags=1):
    """Compute the power spectrum of an AR1 model
    """
    year_arr = tu.split_by_year(ds=ts)
    len_year = len(year_arr[0])
    year_fft = []
    year_fft95 = []
    year_fft5 = []
    for year_ts in tqdm(year_arr):
        surr_arr = []
        surr_ts_dict = ar1_surrogates(data=year_ts, N=N, lags=lags)
        surr_ts_arr = surr_ts_dict['surrogates']
        for surr_ts in surr_ts_arr:
            surr_fft = compute_fft(ts=surr_ts, cutoff=cutoff, window=window)
            if len(year_ts) != len_year:
                surr_arr.append(surr_fft[fft_prop][:-1])
            else:
                surr_arr.append(surr_fft[fft_prop])
        year_fft.append(np.quantile(surr_arr, q=0.5, axis=0))
        year_fft95.append(np.quantile(surr_arr, q=0.9, axis=0))
        year_fft5.append(np.quantile(surr_arr, q=0.1, axis=0))

    return {
        'period': surr_fft['period'],
        'freq': surr_fft['freq'],
        fft_prop: np.mean(year_fft, axis=0),
        f'{fft_prop}95': np.mean(year_fft95, axis=0),
        f'{fft_prop}5': np.mean(year_fft5, axis=0),
    }
