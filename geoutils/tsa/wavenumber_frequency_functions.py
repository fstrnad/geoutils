"""
This file has been adopted from the NCL to Python translation project:
https://github.com/brianpm/wavenumber_frequency/blob/master/wavenumber_frequency_functions.py
Only slight changes have been made to enable compatibility with the geoutils package.
"""
import xarray as xr
import numpy as np
from scipy.signal import convolve2d, detrend
import logging
# logging.basicConfig(level=logging.DEBUG)


def helper():
    """Prints all the functions that are included in this module."""
    f = [
        "decompose2SymAsym(arr)",
        "rmvAnnualCycle(data, spd, fCrit)",
        "convolvePosNeg(arr, k, dim, boundary_index)",
        "simple_smooth_kernel()",
        "smooth_wavefreq(data, kern=None, nsmooth=None, freq_ax=None, freq_name=None)",
        "resolveWavesHayashi( varfft: xr.DataArray, nDayWin: int, spd: int ) -> xr.DataArray",
        "split_hann_taper(series_length, fraction)",
        "spacetime_power(data, segsize=96, noverlap=60, spd=1, latitude_bounds=None, dosymmetries=False, rmvLowFrq=False)",
        "genDispersionCurves(nWaveType=6, nPlanetaryWave=50, rlat=0, Ahe=[50, 25, 12])"]
    [print(fe) for fe in f]


def decompose2SymAsym(arr):
    """Mimic NCL function to decompose into symmetric and asymmetric parts.

    arr: xarra DataArray

    return: DataArray with symmetric in SH, asymmetric in NH

    Note:
        This function produces indistinguishable results from NCL version.
    """
    lat_dim = arr.dims.index('lat')
    # flag to follow NCL convention and put symmetric component in SH
    # & asymmetric in NH
    # method: use flip to reverse latitude, put in DataArray for coords, use loc/isel
    # to assign to negative/positive latitudes (exact equator is left alone)
    data_sym = 0.5*(arr.values + np.flip(arr.values, axis=lat_dim))
    data_asy = 0.5*(arr.values - np.flip(arr.values, axis=lat_dim))
    data_sym = xr.DataArray(data_sym, dims=arr.dims, coords=arr.coords)
    data_asy = xr.DataArray(data_asy, dims=arr.dims, coords=arr.coords)
    out = arr.copy()  # might not be best to copy, but is safe
    out.loc[{'lat': arr['lat'][arr['lat'] < 0]}
            ] = data_sym.isel(lat=data_sym.lat < 0)
    out.loc[{'lat': arr['lat'][arr['lat'] > 0]}
            ] = data_asy.isel(lat=data_asy.lat > 0)
    return out


def rmvAnnualCycle(data, spd, fCrit):
    """remove frequencies less than fCrit from data.

    data: xarray DataArray
    spd: sampling frequency in samples-per-day
    fCrit: frequency threshold; remove frequencies < fCrit

    return: xarray DataArray, shape of data

    Note: fft/ifft preserves the mean because z = fft(x), z[0] is the mean.
          To keep the mean here, we need to keep the 0 frequency.

    Note: This function reproduces the results from the NCL version.

    Note: Two methods are available, one using fft/ifft and the other rfft/irfft.
          They both produce output that is indistinguishable from NCL's result.
    """
    dimz = data.sizes
    ntim = dimz['time']
    time_ax = list(data.dims).index('time')
    # Method 1: Uses the complex FFT, returns the negative frequencies too, but they
    # should be redundant b/c they are conjugate of positive ones.
    cf = np.fft.fft(data.values, axis=time_ax)
    freq = np.fft.fftfreq(ntim, spd)
    cf[(freq != 0) & (np.abs(freq) < fCrit), ...] = 0.0  # keeps the mean
    z = np.fft.ifft(cf, n=ntim, axis=0)
    # Method 2: Uses the real FFT. In this case,
    # cf = np.fft.rfft(data.values, axis=time_ax)
    # freq = np.linspace(1, (ntim*spd)//2, (ntim*spd)//2) / ntim
    # fcrit_ndx = np.argwhere(freq < fCrit).max()
    # if fcrit_ndx > 1:
    #     cf[1:fcrit_ndx+1, ...] = 0.0
    # z = np.fft.irfft(cf, n=ntim, axis=0)
    z = xr.DataArray(z.real, dims=data.dims, coords=data.coords)
    return z


def convolvePosNeg(arr, k, dim, boundary_index):
    """Apply convolution of (arr, k) excluding data at boundary_index in dimension dim.

    arr: numpy ndarray of data
    k: numpy ndarray, same dimension as arr, this should be the kernel
    dim: integer indicating the axis of arr to split
    boundary_index: integer indicating the position to split dim

    Split array along dim at boundary_index;
    perform convolution on each sub-array;
    reconstruct output array from the two subarrays;
    the values of output at boundary_index of dim will be same as input.

    `convolve2d` is `scipy.signal.convolve2d()`
    """
    # arr: numpy ndarray
    oarr = arr.copy()  # maybe not good to make a fresh copy every time?
    # first pass is [0 : boundary_index)
    slc1 = [slice(None)] * arr.ndim
    slc1[dim] = slice(None, boundary_index)
    arr1 = arr[tuple(slc1)]
    ans1 = convolve2d(arr1, k, boundary='symm', mode='same')
    # second pass is [boundary_index+1, end]
    slc2 = [slice(None)] * arr.ndim
    slc2[dim] = slice(boundary_index+1, None)
    arr2 = arr[tuple(slc2)]
    ans2 = convolve2d(arr2, k, boundary='symm', mode='same')
    # fill in the output array
    oarr[tuple(slc1)] = ans1
    oarr[tuple(slc2)] = ans2
    return oarr


def simple_smooth_kernel():
    """Provide a very simple smoothing kernel."""
    kern = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]])
    return kern / kern.sum()


def smooth_wavefreq(data, kern=None, nsmooth=None, freq_ax=None, freq_name=None):
    """Apply a convolution of (data,kern) nsmooth times.
       The convolution is applied separately to the positive and negative frequencies.
       Either the name (freq_name: str) or axis index (freq_ax: int) of frequency is required, with the name preferred.
    """
    assert isinstance(data, xr.DataArray)
    if kern is None:
        kern = simple_smooth_kernel()
    if nsmooth is None:
        nsmooth = 20
    if freq_name is not None:
        axnum = list(data.dims).index(freq_name)
        # <-- THIS IS SUPPOSED TO BE THE INDEX AT FREQ==0.0
        nzero = data.sizes[freq_name] // 2
    elif freq_ax is not None:
        axnum = freq_ax
        nzero = data.shape[freq_ax] // 2
    else:
        raise ValueError(
            "smooth_wavefreq needs to know how to find frequency dimension.")
    # this is a custom function to skip 0-frequency (mean)
    smth1pass = convolvePosNeg(data, kern, axnum, nzero)
    # note: the convolution is strictly 2D and the boundary condition is symmetric --> if kernel is normalized, preserves the sum.
    smth1pass = xr.DataArray(smth1pass, dims=data.dims,
                             coords=data.coords)  # ~copy_metadata
    # repeat smoothing many times:
    smthNpass = smth1pass.values.copy()
    for i in range(nsmooth):
        smthNpass = convolvePosNeg(smthNpass, kern, axnum, nzero)
    return xr.DataArray(smthNpass, dims=data.dims, coords=data.coords)


def resolveWavesHayashi(varfft: xr.DataArray, nDayWin: int, spd: int) -> xr.DataArray:
    """This is a direct translation from the NCL routine to python/xarray.
    input:
        varfft : expected to have rightmost dimensions of wavenumber and frequency.
        varfft : expected to be an xarray DataArray with coordinate variables.
        nDayWin : integer that is the length of the segments in days.
        spd : the sampling frequency in `timesteps` per day (I think).

    returns:
        a DataArray that is reordered to have correct westward & eastward propagation.

    """
    # -------------------------------------------------------------
    # Special reordering to resolve the Progressive and Retrogressive waves
    # Reference: Hayashi, Y.
    #    A Generalized Method of Resolving Disturbances into
    #    Progressive and Retrogressive Waves by Space and
    #    Fourier and Time Cross-Spectral Analysis
    #    J. Meteor. Soc. Japan, 1971, 49: 125-128.
    # -------------------------------------------------------------

    # in NCL varfft is dimensioned (2,mlon,nSampWin), but the first dim doesn't matter b/c python supports complex numbers.
    #
    # Create array PEE(NL+1,NT+1) which contains the (real) power spectrum.
    # all the following assume indexing starting with 0
    # In this array (PEE), the negative wavenumbers will be from pn=0 to NL/2-1 (left).
    # The positive wavenumbers will be for pn=NL/2+1 to NL (right).
    # Negative frequencies will be from pt=0 to NT/2-1 (left).
    # Positive frequencies will be from pt=NT/2+1 to NT  (right).
    # Information about zonal mean will be for pn=NL/2 (middle).
    # Information about time mean will be for pt=NT/2 (middle).
    # Information about the Nyquist Frequency is at pt=0 and pt=NT
    #

    # In PEE, define the
    # WESTWARD waves to be either
    #          positive frequency and negative wavenumber
    #          OR
    #          negative freq and positive wavenumber.
    # EASTWARD waves are either positive freq and positive wavenumber
    #          OR negative freq and negative wavenumber.

    # Note that frequencies are returned from fftpack are ordered like so
    #    input_time_pos [ 0    1   2    3     4      5    6   7  ]
    #    ouput_fft_coef [mean 1/7 2/7  3/7 nyquist -3/7 -2/7 -1/7]
    #                    mean,pos freq to nyq,neg freq hi to lo
    #
    # Rearrange the coef array to give you power array of freq and wave number east/west
    # Note east/west wave number *NOT* eq to fft wavenumber see Hayashi '71
    # Hence, NCL's 'cfftf_frq_reorder' can *not* be used.
    # BPM: This goes for np.fft.fftshift
    #
    # For ffts that return the coefficients as described above, here is the algorithm
    # coeff array varfft(2,n,t)   dimensioned (2,0:numlon-1,0:numtim-1)
    # new space/time pee(2,pn,pt) dimensioned (2,0:numlon  ,0:numtim  )
    #
    # NOTE: one larger in both freq/space dims
    # the initial index of 2 is for the real (indx 0) and imag (indx 1) parts of the array
    #
    #
    #    if  |  0 <= pn <= numlon/2-1    then    | numlon/2 <= n <= 1
    #        |  0 <= pt < numtim/2-1             | numtim/2 <= t <= numtim-1
    #
    #    if  |  0         <= pn <= numlon/2-1    then    | numlon/2 <= n <= 1
    #        |  numtime/2 <= pt <= numtim                | 0        <= t <= numtim/2
    #
    #    if  |  numlon/2  <= pn <= numlon    then    | 0  <= n <= numlon/2
    #        |  0         <= pt <= numtim/2          | numtim/2 <= t <= 0
    #
    #    if  |  numlon/2   <= pn <= numlon    then    | 0        <= n <= numlon/2
    #        |  numtim/2+1 <= pt <= numtim            | numtim-1 <= t <= numtim/2

    # local variables : dimvf, numlon, N, varspacetime, pee, wave, freq

    logging.debug(f"[Hayashi] nDayWin: {nDayWin}, spd: {spd}")
    dimnames = varfft.dims
    dimvf = varfft.shape
    # number of longitudes = numer of wavenumbers
    mlon = len(varfft['wavenumber'])
    N = len(varfft['frequency'])
    k_dim_index = dimnames.index('wavenumber')
    f_dim_index = dimnames.index('frequency')
    logging.info(
        f"[Hayashi] input dims is {dimnames}, {dimvf} || Input dtype: {varfft.dtype=}")
    logging.info(f"[Hayashi] input coords is {varfft.coords}")
    logging.debug(
        f"[Hayashi] wavenumber axis is {k_dim_index}, frequency axis is {f_dim_index}")
    if len(dimnames) != len(varfft.coords):
        logging.error("The size of varfft.coords is incorrect.")
        raise ValueError("STOP")

    nshape = list(dimvf)
    nshape[k_dim_index] += 1
    nshape[f_dim_index] += 1
    logging.debug(f"[Hayashi] The nshape ends up being {nshape}")
    # this is a reordering, use Ellipsis to allow arbitrary number of dimensions,
    # but we insist that the wavenumber and frequency dims are rightmost.
    # we will fill the new array in increasing order (arbitrary choice)
    varspacetime = np.full(nshape, np.nan, dtype=type(varfft))
    # first two are the negative wavenumbers (westward),
    # second two are the positive wavenumbers (eastward)
    logging.debug(
        f"[Hayashi] Assign values into array. Notable numbers: mlon//2={mlon//2}, N//2={N//2}")
    varspacetime[..., 0:mlon//2, 0:N//2] = varfft[...,
                                                  mlon//2:0:-1, N//2:]  # neg.k, pos.w
    varspacetime[..., 0:mlon//2, N//2:] = varfft[...,
                                                 mlon//2:0:-1, 0:N//2+1]   # neg.k,
    # assign eastward & neg.freq.
    varspacetime[..., mlon//2:, 0:N//2+1] = varfft[..., 0:mlon//2+1,  N//2::-1]
    # assign eastward & pos.freq.
    varspacetime[..., mlon//2:, N//2 +
                 1:] = varfft[..., 0:mlon//2+1, -1:N//2-1:-1]
    logging.debug(f"[Hayashi] Shape after reordering: {varspacetime.shape}")
    logging.debug(f"[Hayashi] Sum after reordering: {varspacetime.sum()}")
    #  Create the real power spectrum pee = sqrt(real^2+imag^2)^2
    logging.debug(
        f"[Hayashi] calculate power by absolute value (i.e. sqrt(real**2 + imag**2))and squaring.")
    pee = (np.abs(varspacetime))**2
    logging.debug(
        f"[Hayashi] sum of pee {pee.sum()}. Type of pee: {type(pee)} Dtype: {pee.dtype}")
    logging.debug(f"[Hayashi] put into DataArray")
    # add meta data for use upon return
    wave = np.arange(-mlon // 2, (mlon // 2) + 1, 1, dtype=int)
    freq = np.linspace(-1*nDayWin*spd/2, nDayWin*spd /
                       2, (nDayWin*spd)+1) / nDayWin

    logging.debug(f"[Hayashi] freq size is {freq.shape}.")
    odims = list(dimnames)
    odims[-2] = "wavenumber"
    odims[-1] = "frequency"
    ocoords = {}
    for c in varfft.coords:
        logging.debug(f"[hayashi] working on coordinate {c}")
        if (c != "wavenumber") and (c != "frequency"):
            ocoords[c] = varfft[c]
        elif c == "wavenumber":
            ocoords['wavenumber'] = wave
        elif c == "frequency":
            ocoords['frequency'] = freq
    pee = xr.DataArray(pee, dims=odims, coords=ocoords)
    z = pee.copy()
    z.loc[{'frequency': 0}] = np.nan
    logging.debug(
        f"[Hayashi] Sum at the end (removing zero freq): {z.sum().item()}")
    return pee


def split_hann_taper(series_length, fraction):
    """Implements `split cosine bell` taper of length `series_length`
       where only fraction of points are tapered (combined on both ends).

    This returns a function that tapers to zero on the ends. To taper to the mean of a series X:
    XTAPER = (X - X.mean())*series_taper + X.mean()
    """
    npts = int(np.rint(fraction * series_length))  # total size of taper
    taper = np.hanning(npts)
    series_taper = np.ones(series_length)
    series_taper[0:npts//2+1] = taper[0:npts//2+1]
    series_taper[-npts//2+1:] = taper[npts//2+1:]
    return series_taper


def spacetime_power(data, segsize=96, noverlap=60, spd=1, latitude_bounds=None, dosymmetries=False, rmvLowFrq=False):
    """Perform space-time spectral decomposition and return power spectrum following Wheeler-Kiladis approach.

    data: an xarray DataArray to be analyzed; needs to have (time, lat, lon) dimensions.
    segsize: integer denoting the size of time samples that will be decomposed (typically about 96)
    noverlap: integer denoting the number of days of overlap from one segment to the next
    spd: sampling rate, in "samples per day" (e.g. daily=1, 6-houry=4)

    latitude_bounds: a tuple of (southern_extent, northern_extent) to reduce data size.

    dosymmetries: if True, follow NCL convention of putting symmetric component in SH, antisymmetric in NH
                  If True, the function returns a DataArray with a `component` dimension.

    rmvLowFrq: if True, remove frequencies < 1/segsize from data.

    Method
    ------
        1. Subsample in latitude if latitude_bounds is specified.
        2. Detrend the data (but keeps the mean value, as in NCL)
        3. High-pass filter if rmvLowFrq is True
        4. Construct symmetric/antisymmetric array if dosymmetries is True.
        5. Construct overlapping window view of data.
        6. Detrend the segments (strange enough, removing mean).
        7. Apply taper in time dimension of windows (aka segments).
        8. Fourier transform
        9. Apply Hayashi reordering to get propagation direction & convert to power.
       10. return DataArray with power.

    Notes
    -----
        Upon returning power, this should be comparable to "raw" spectra.
        Next step would be be to smooth with `smooth_wavefreq`,
        and divide raw spectra by smooth background to obtain "significant" spectral power.

    """

    segsize = spd*segsize  # segment size in samples

    if latitude_bounds is not None:
        assert isinstance(latitude_bounds, tuple)
        # CAUTION: is this a mutable argument?
        data = data.sel(lat=slice(*latitude_bounds))
        logging.info(f"Data reduced by latitude bounds. Size is {data.sizes}")
        slat = latitude_bounds[0]
        nlat = latitude_bounds[1]
    else:
        slat = data['lat'].min().item()
        nlat = data['lat'].max().item()

    # "Remove dominant signals"

    # "detrend" the data, including removing the mean (uses scipy.signal.detrend):
    #  --> ncl version keeps the mean:
    xmean = data.mean(dim='time')
    xdetr = detrend(data.values, axis=0, type='linear')
    xdetr = xr.DataArray(xdetr, dims=data.dims, coords=data.coords)
    xdetr += xmean  # put the mean back in
    # --> Tested and confirmed that this approach gives same answer as NCL

    # field testing it: pass
    # if not(hasattr(xdetr, "name")) or (not isinstance(xdetr.name, str) ):
    #     xdetr.name = "detrended"
    # xdetr.to_netcdf("/Users/brianpm/Documents/pout_0_detrend.nc")

    # filter low-frequencies
    if rmvLowFrq:
        data = rmvAnnualCycle(xdetr, spd, 1/segsize)
    # --> Tested and confirmed that this function gives same answer as NCL
    # testing: pass -- indistinguisable from file produced by NCL
    # data.name = "filtered"
    # data.to_netcdf("/Users/brianpm/Documents/pout_1_filtered.nc")

    # NOTE: we have altered "data" to be detrended & filtered at this point

    dimsizes = data.sizes  # dict
    lon_size = dimsizes['lon']
    lat_size = dimsizes['lat']
    lat_dim = data.dims.index('lat')
    if dosymmetries:
        data = decompose2SymAsym(data)
    # testing: pass -- Gets the same result as NCL.
    logging.debug(
        f"[spacetime_power] data shape after removing low frequencies: {data.shape}")
    logging.debug(
        f"[spacetime_power] variance of data before windowing: {np.var(data).item()}")

    # 2. Windowing with the xarray "rolling" operation, and then limit overlap with `construct` to produce a new dataArray.
    # WK99 recommend "2-month" overlap
    # Shape of x_win: (_, lat, lon, segments: spd*segsize)
    # WK99 use 96-day window
    x_roll = data.rolling(time=segsize, min_periods=segsize)
    assert segsize-noverlap > 0, f"Error, inconsistent specification of segsize and noverlap results in stride of {segsize-noverlap}, but must be > 0."
    x_win = x_roll.construct("segments")
    x_win = x_win.isel(time=slice(segsize-1, None, segsize-noverlap))

    logging.debug(f"[spacetime_power] x_win shape is {x_win.shape}")
    # Additional detrend for each segment:
    if np.logical_not(np.any(np.isnan(x_win))):
        logging.info("No missing, so use simplest segment detrend.")
        # <-- missing data makes this not work
        x_win_detr = detrend(x_win.values, axis=-1, type='linear')
        x_win = xr.DataArray(x_win_detr, dims=x_win.dims, coords=x_win.coords)
    else:
        logging.warning(
            "EXTREME WARNING -- This method to detrend with missing values present does not quite work, probably need to do interpolation instead.")
        logging.warning(
            "There are missing data in x_win, so have to try to detrend around them.")
        x_win_cp = x_win.values.copy()
        logging.info(f"[spacetime_power] x_win_cp windowed data has shape {x_win_cp.shape} \n \t It is a numpy array, copied from x_win which has dims: {x_win.sizes} \n \t ** about to detrend this in the rightmost dimension.")
        x_win_cp[np.logical_not(np.isnan(x_win_cp))] = detrend(
            x_win_cp[np.logical_not(np.isnan(x_win_cp))])
        x_win = xr.DataArray(x_win_cp, dims=x_win.dims, coords=x_win.coords)
    logging.debug(
        f"[spacetime_power] x_win variance of segments: {np.var(x_win, axis=(1, 2, 3)).values}")
    # 3. Taper in time to make the signal periodic, as required for FFT.
    # taper = np.hanning(segsize)  # WK seem to use some kind of stretched out hanning window; unclear if it matters
    taper = split_hann_taper(segsize, 0.1)  # try to replicate NCL's
    # would do XTAPER = (X - X.mean())*series_taper + X.mean()
    x_wintap = x_win*taper
    # But since we have removed the mean, taper going to 0 is equivalent to taper going to the mean.
    logging.debug(
        f"[spacetime_power] x_wintap variance of segments: {np.var(x_wintap, axis=(1, 2, 3)).values}")

    # Do the transform using 2D FFT
    # - normalize by dimension sizes
    z = np.fft.fft2(x_wintap, axes=(2, 3)) / (lon_size * segsize)

    # NOTE: with this normalization, the power spectral density should
    #       be calculated as np.abs(z)**2 * dlon * dt * lon_size * segsize
    #       where dt = 1/spd, so dt*segsize=[length of segment in time]
    #       and dlon = lon[1]-lon[0] (= size of longitude dimension in degrees)
    #       _When only the positive frequencies are used, also multiply by 2._
    # AND: the integral of the power spectral density is then equal to the variance
    #      In this case, gets the variance of x_wintap; for suitably large segsize,
    #      the tapering shouldn't matter much, so VAR[x_wintap] ≃ VAR[x_win]
    # TEST OF INT[psd] = VAR[x_wintap]:
    # dlon = x_wintap['lon'][1].item()-x_wintap['lon'][0].item()
    # dt = 1/spd
    # Nlon = lon_size
    # Nt = segsize
    # print(f"{dlon = }, {dt = }, {Nlon = }, {Nt = }")
    # psd = (np.abs(z)**2) * dlon * dt * (Nlon * Nt)
    # logging.debug(f"{psd.shape = }")
    # variance = np.var(x_wintap, axis=(2,3))
    # logging.debug(f"VARIANCE ARRAY IS LEFT AS: {variance.shape = }")
    # kx = np.fft.fftfreq(Nlon, dlon)
    # ky = np.fft.fftfreq(Nt, dt)
    # psd_integral = np.sum(psd, axis=(2,3))* (kx[1]-kx[0]) * (ky[1]-ky[0])
    # logging.debug(f"PSD_INTEGRAL SHAPE: {psd_integral.shape}")
    # print(f"Variance of data (0,0): {variance[0,0]} -- Integral of spectrum: {psd_integral[0,0]}")

    # z has both positive & negative frequencies : usually you'd take the positive in each dimension and double it

    # Or do the transform with 2 steps (equivalent!)
    # z = np.fft.fft(x_wintap, axis=2) / lon_size  # note that np.fft.fft() produces same answers as NCL cfftf
    # z = np.fft.fft(z, axis=3) / segsize

    z = xr.DataArray(z, dims=("time", "lat", "wavenumber", "frequency"),
                     coords={"time": x_wintap["time"],
                             "lat": x_wintap["lat"],
                             "wavenumber": np.fft.fftfreq(lon_size, 1/lon_size),
                             "frequency": np.fft.fftfreq(segsize, 1/spd)})

    # The FFT is returned following ``standard order`` which has negative frequencies in second half of array.
    #
    # IMPORTANT:
    # If this were typical 2D FFT, we would do the following to get the frequencies and reorder:
    #         z_k = np.fft.fftfreq(x_wintap.shape[-2], 1/lon_size)
    #         z_v = np.fft.fftfreq(x_wintap.shape[-1], 1)  # Assumes 1/(1-day) timestep
    # reshape to get the frequencies centered
    #         z_centered = np.fft.fftshift(z, axes=(2,3))
    #         z_k_c = np.fft.fftshift(z_k)
    #         z_v_c = np.fft.fftshift(z_v)
    # and convert to DataArray as this:
    #         d1 = list(x_win.dims)
    #         d1[-2] = "wavenumber"
    #         d1[-1] = "frequency"
    #         c1 = {}
    #         for d in d1:
    #             if d in x_win.coords:
    #                 c1[d] = x_win[d]
    #             elif d == "wavenumber":
    #                 c1[d] = z_k_c
    #             elif d == "frequency":
    #                 c1[d] = z_v_c
    #         z_centered = xr.DataArray(z_centered, dims=d1, coords=c1)
    # BUT THAT IS INCORRECT TO GET THE PROPAGATION DIRECTION OF ZONAL WAVES
    # (in testing, it seems to end up opposite in wavenumber)
    # Apply reordering per Hayashi to get correct wave propagation convention
    #     this function is customized to expect z to be a DataArray
    z_pee = resolveWavesHayashi(z, segsize//spd, spd)
    # z_pee is spectral power already.
    # z_pee is a DataArray w/ coordinate vars for wavenumber & frequency

    # average over all available segments and sum over latitude
    # OUTPUT DEPENDS ON SYMMETRIES
    if dosymmetries:
        # multipy by 2 b/c we only used one hemisphere
        z_symmetric = 2.0 * \
            z_pee.isel(lat=z_pee.lat < 0).mean(
                dim='time').sum(dim='lat').squeeze()
        z_symmetric.name = "power"
        z_antisymmetric = 2.0 * \
            z_pee.isel(lat=z_pee.lat > 0).mean(
                dim='time').sum(dim='lat').squeeze()
        z_antisymmetric.name = "power"
        z_final = xr.concat([z_symmetric, z_antisymmetric], "component")
        z_final = z_final.assign_coords(
            {"component": ["symmetric", "antisymmetric"]})
    else:
        lat = z_pee['lat']
        lat_inds = np.argwhere(
            ((lat <= nlat) & (lat >= slat)).values).squeeze()
        z_final = z_pee.isel(lat=lat_inds).mean(
            dim='time').sum(dim='lat').squeeze()

    logging.getLogger("asyncio").setLevel(logging.WARNING)
    return z_final


def genDispersionCurves(nWaveType=6, nPlanetaryWave=50, rlat=0,
                        Ahe=[100, 50, 25, 12]):
    """
    Function to derive the shallow water dispersion curves. Closely follows NCL version.

    input:
        nWaveType : integer, number of wave types to do
        nPlanetaryWave: integer
        rlat: latitude in radians (just one latitude, usually 0.0)
        Ahe: [50.,25.,12.] equivalent depths
              ==> defines parameter: nEquivDepth ; integer, number of equivalent depths to do == len(Ahe)

    returns: tuple of size 2
        Afreq: Frequency, shape is (nWaveType, nEquivDepth, nPlanetaryWave)
        Apzwn: Zonal savenumber, shape is (nWaveType, nEquivDepth, nPlanetaryWave)

    notes:
        The outputs contain both symmetric and antisymmetric waves. In the case of
        nWaveType == 6:
        0,1,2 are (ASYMMETRIC) "MRG", "IG", "EIG" (mixed rossby gravity, inertial gravity, equatorial inertial gravity)
        3,4,5 are (SYMMETRIC) "Kelvin", "ER", "IG" (Kelvin, equatorial rossby, inertial gravity)
    """
    nEquivDepth = len(
        Ahe)  # this was an input originally, but I don't know why.
    pi = np.pi
    radius = 6.37122e06    # [m]   average radius of earth
    g = 9.80665        # [m/s] gravity at 45 deg lat used by the WMO
    omega = 7.292e-05      # [1/s] earth's angular vel
    # U     = 0.0   # NOT USED, so Commented
    # Un    = 0.0   # since Un = U*T/L  # NOT USED, so Commented
    ll = 2.*pi*radius*np.cos(np.abs(rlat))
    Beta = 2.*omega*np.cos(np.abs(rlat))/radius
    fillval = 1e20

    # NOTE: original code used a variable called del,
    #       I just replace that with `dell` because `del` is a python keyword.

    # Initialize the output arrays
    Afreq = np.empty((nWaveType, nEquivDepth, nPlanetaryWave))
    Apzwn = np.empty((nWaveType, nEquivDepth, nPlanetaryWave))

    for ww in range(1, nWaveType+1):
        for ed, he in enumerate(Ahe):
            # this loops through the specified equivalent depths
            # ed provides index to fill in output array, while
            # he is the current equivalent depth
            # T = 1./np.sqrt(Beta)*(g*he)**(0.25) This is close to pre-factor of the dispersion relation, but is not used.
            c = np.sqrt(g * he)  # phase speed
            # was: (g*he)**(0.25)/np.sqrt(Beta), this is Rossby radius of deformation
            L = np.sqrt(c/Beta)

            for wn in range(1, nPlanetaryWave+1):
                s = -20.*(wn-1)*2./(nPlanetaryWave-1) + 20.
                k = 2.0 * pi * s / ll
                kn = k * L

                # Anti-symmetric curves
                if (ww == 1):       # MRG wave
                    if (k < 0):
                        dell = np.sqrt(1.0 + (4.0 * Beta)/(k**2 * c))
                        deif = k * c * (0.5 - 0.5 * dell)

                    if (k == 0):
                        deif = np.sqrt(c * Beta)

                    if (k > 0):
                        deif = fillval

                if (ww == 2):       # n=0 IG wave
                    if (k < 0):
                        deif = fillval

                    if (k == 0):
                        deif = np.sqrt(c * Beta)

                    if (k > 0):
                        dell = np.sqrt(1.+(4.0*Beta)/(k**2 * c))
                        deif = k * c * (0.5 + 0.5 * dell)

                if (ww == 3):       # n=2 IG wave
                    n = 2.
                    dell = (Beta*c)
                    deif = np.sqrt((2.*n+1.)*dell + (g*he) * k**2)
                    # do some corrections to the above calculated frequency.......
                    for i in range(1, 5+1):
                        deif = np.sqrt((2.*n+1.)*dell + (g*he)
                                       * k**2 + g*he*Beta*k/deif)

                # symmetric curves
                if (ww == 4):       # n=1 ER wave
                    n = 1.
                    if (k < 0.0):
                        dell = (Beta/c)*(2.*n+1.)
                        deif = -Beta*k/(k**2 + dell)
                    else:
                        deif = fillval

                if (ww == 5):       # Kelvin wave
                    deif = k*c

                if (ww == 6):       # n=1 IG wave
                    n = 1.
                    dell = (Beta*c)
                    deif = np.sqrt((2. * n+1.) * dell + (g*he)*k**2)
                    # do some corrections to the above calculated frequency
                    for i in range(1, 5+1):
                        deif = np.sqrt((2.*n+1.)*dell + (g*he)
                                       * k**2 + g*he*Beta*k/deif)

                eif = deif  # + k*U since  U=0.0
                P = 2.*pi/(eif*24.*60.*60.)  # => PERIOD
                # dps  = deif/k  # Does not seem to be used.
                # R    = L #<-- this seemed unnecessary, I just changed R to L in Rdeg
                # Rdeg = (180.*L)/(pi*6.37e6) # And it doesn't get used.

                Apzwn[ww-1, ed-1, wn-1] = s
                if (deif != fillval):
                    # P = 2.*pi/(eif*24.*60.*60.) # not sure why we would re-calculate now
                    Afreq[ww-1, ed-1, wn-1] = 1./P
                else:
                    Afreq[ww-1, ed-1, wn-1] = fillval

    Afreq = np.where(Afreq == 1e20, np.nan, Afreq)
    Apzwn = np.where(Apzwn == 1e20, np.nan, Apzwn)

    return Afreq, Apzwn


def kf_filter(data):
    """
    Follows Wheeler-Kiladis and replicates NCL's kf_filter.
    Uses the entire time instead of breaking into segments.

    data: xr.DataArray
        NCL VERSION USES ONLY (time, lon)
        so we to that here, too

    """
    # "detrend" the data, including removing the mean (uses scipy.signal.detrend):
    #  --> ncl version keeps the mean:
    xmean = data.mean(dim='time')
    time_axis_index = data.dims.index('time')
    xdetr = xr.DataArray(detrend(data.values, axis=time_axis_index,
                         type='linear'), dims=data.dims, coords=data.coords)
    # xdetr += xmean # put the mean back in (not included in NCL kf_filter)

    # taper to the mean
    xtapr = split_hann_taper(series_length, fraction)
    xtapr = (xdetr - xdetr.mean(dim='time'))*xtapr + xdetr.mean(dim='time')

    # fft
    z = np.fft.fft2(xtapr)


def wf_analysis(x, period=False, **kwargs):
    """Return normalized spectra of x using standard processing parameters."""
    # Get the "raw" spectral power
    # OPTIONAL kwargs:
    # segsize, noverlap, spd, latitude_bounds (tuple: (south, north)), dosymmetries, rmvLowFrq

    z2 = spacetime_power(x, **kwargs)
    z2avg = z2.mean(dim='component')
    z2.loc[{'frequency': 0}] = np.nan  # get rid of spurious power at \nu = 0
    # the background is supposed to be derived from both symmetric & antisymmetric
    background = smooth_wavefreq(
        z2avg, kern=simple_smooth_kernel(), nsmooth=50, freq_name='frequency')
    # separate components
    z2_sym = z2[0, ...]
    z2_asy = z2[1, ...]
    # normalize
    nspec_sym = z2_sym / background
    nspec_asy = z2_asy / background

    if period:
        nspec_sym.assign_coords(frequency=1./nspec_sym.frequency)
        nspec_sym = nspec_sym.rename({'frequency': 'period'})

        nspec_asy.assign_coords(frequency=1./nspec_asy.frequency)
        nspec_asy = nspec_asy.rename({'frequency': 'period'})

    return nspec_sym, nspec_asy
