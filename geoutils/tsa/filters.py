import xarray as xr
from scipy.signal import filtfilt, cheby1, argrelmax,  find_peaks
import numpy as np
import scipy.ndimage as ndim
import geoutils.utils.general_utils as gut
from importlib import reload
import xrscipy.signal.extra as dsp_extra


reload(gut)


def lowpass(x, cutoff, order=None, dim='time'):
    if isinstance(x, xr.DataArray):
        coords = x.coords
        dims = gut.get_dims(x)
        if 'time' in dims:
            x = x.assign_coords(time=np.arange(len(x.time)))
    x_filt = dsp_extra.lowpass(x, f_cutoff=cutoff, dim=dim)
    x_filt = x_filt.transpose(*dims)
    if isinstance(x, xr.DataArray):
        x_filt = xr.DataArray(
            data=x_filt.data,
            coords=coords,
            dims=dims
        )
    return x_filt


def bandpass(x, cutoff_low, cutoff_high, order=None, dim='time'):
    if isinstance(x, xr.DataArray):
        coords = x.coords
        dims = gut.get_dims(x)
        if 'time' in dims:
            x = x.assign_coords(time=np.arange(len(x.time)))
    x_filt = dsp_extra.bandpass(x, f_low=cutoff_low,
                                f_high=cutoff_high,
                                dim=dim)
    x_filt = x_filt.transpose(*dims)
    if isinstance(x, xr.DataArray):
        x_filt = xr.DataArray(
            data=x_filt.data,
            coords=coords,
            dims=dims
        )
    return x_filt


def cheby_lowpass(cutoff, fs, order, rp):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = cheby1(order, rp, normal_cutoff, btype='low', analog=False)
    return b, a


def cheby_lowpass_filter(x, cutoff, fs, order, rp):
    b, a = cheby_lowpass(cutoff, fs, order, rp)
    y = filtfilt(b, a, x)
    return y


def apply_cheby_filter(ts,
                       cutoff=4,
                       order=8,
                       fs=1,
                       rp=.05):
    fcutoff = .95 * 1. / cutoff
    ts_lp = cheby_lowpass_filter(ts, fcutoff, fs, order, rp)
    if type(ts) == xr.DataArray:
        ts_lp = xr.DataArray(
            data=ts_lp,
            dims=ts.dims,
            coords=ts.coords,
            name=ts.name
        )

    return ts_lp


def apply_Savitzky_Golay_filter(ts, wl=1, order=3):
    from scipy.signal import savgol_filter

    ts_sav_gol = savgol_filter(ts,
                               window_length=wl,
                               polyorder=order
                               )
    if type(ts) == xr.DataArray:
        ts_sav_gol = xr.DataArray(
            data=ts_sav_gol,
            dims=ts.dims,
            coords=ts.coords,
            name=ts.name
        )

    return ts_sav_gol


def apply_butter_filter(ts,
                        cutoff,
                        order=3,
                        fs=50):
    from scipy.signal import butter, filtfilt
    # nyq = 0.5 * fs
    # normal_cutoff = cutoff / nyq  # Nyquist Frequency
    if cutoff > 1:
        normal_cutoff = 1 / cutoff
        # Get the filter coefficients
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        ts_lp = filtfilt(b, a, ts)
        if type(ts) == xr.DataArray:
            ts_lp = xr.DataArray(
                data=ts_lp,
                dims=ts.dims,
                coords=ts.coords,
                name=ts.name
            )
    elif cutoff < 1:
        gut.myprint(f'High-pass filter not implemented yet!')
    else:
        ts_lp = ts

    return ts_lp


def apply_med_filter(ts,
                     size=7):
    """This function performs a median average of size 7 on a given ts

    Args:
        ts (ndarray): nd array, eg. a time series
        size (int, optional): Size of the rolling median average. Defaults to 7.

    Returns:
        ndarray: median filtered time series
    """

    ts_med_fil = ndim.median_filter(ts, size=size)

    return ts_med_fil


def compute_lead_lag_corr(ts1, ts2, lag=0, corr_method='spearman'):
    import scipy.stats as st
    Nx = len(ts1)
    if Nx != len(ts2):
        raise ValueError('x and y must be equal length')
    if lag != 0:
        print('WARNING! Input 2 is shifted!')
        nts2_shift = np.roll(ts2, lag, axis=0)
    else:
        nts2_shift = ts2
    if corr_method == 'spearman':
        corr, p_val = st.spearmanr(
            ts1, nts2_shift, axis=0, nan_policy='propagate')
    elif corr_method == 'pearson':
        corr = np.corrcoef(ts1.T, nts2_shift)

    return corr
