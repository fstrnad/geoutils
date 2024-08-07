from scipy.signal import find_peaks
import statsmodels.stats as sm
from statsmodels.tsa.ar_model import AutoReg
import geoutils.utils.time_utils as tu
from sklearn.preprocessing import minmax_scale
from importlib import reload
from scipy.stats import norm, percentileofscore
from scipy.stats import skew
import scipy.special as special
import scipy.stats as st
import numpy as np
import xarray as xr
import copy
from tqdm import tqdm
from joblib import Parallel, delayed
import geoutils.utils.general_utils as gut

reload(gut)


def holm(pvals, alpha=0.05, corr_type="dunn"):
    """
    Returns indices of p-values using Holm's method for multiple testing.

    Args:
    ----
    pvals: list
        list of p-values
    alpha: float
    corr_type: str
    """
    n = len(pvals)
    sortidx = np.argsort(pvals)
    p_ = pvals[sortidx]
    j = np.arange(1, n + 1)
    if corr_type == "bonf":
        corr_factor = alpha / (n - j + 1)
    elif corr_type == "dunn":
        corr_factor = 1. - (1. - alpha) ** (1. / (n - j + 1))
    try:
        idx = np.where(p_ <= corr_factor)[0][-1]
        lst_idx = sortidx[:idx]
    except IndexError:
        lst_idx = []
    return lst_idx


def norths_rule_of_thumb(pca_eigval, n_samples):
    """Error bar for PCA components.

    Args:
        pca_eigval (np.ndarray): Eigenvalues of PCA
        n_samples (int): Number of datapoints.

    Returns:
        (np.ndarray): Error of each PCA component.
    """
    norths_rule = pca_eigval * np.sqrt(2 / n_samples)
    return norths_rule


def correct_p_values(pvals, alpha=0.05, method="fdr_bh"):
    if isinstance(pvals, xr.DataArray):
        pvals = pvals.data
    p_data_shape = pvals.shape
    if method == 'fdr_bh':
        bools, p_corrected = sm.multitest.fdrcorrection(pvals.flatten(),
                                                        alpha=alpha,
                                                        method='indep',
                                                        is_sorted=False)
    else:
        raise ValueError(f'Correction type {method} not implemented!')
    return p_corrected.reshape(p_data_shape)


def bin_ds(ds, vmin=None, vmax=None, **kwargs):
    data = ds.data
    if vmin is None:
        vmin = np.min(ds)
    if vmax is None:
        vmax = np.max(ds)
    density = kwargs.pop("density", False)
    nbins = kwargs.pop("nbins", None)
    bw = kwargs.pop("bw", None)
    round_bins = kwargs.pop('round_bins', True)
    if nbins is None and bw is None:
        nbins = __doane(data)
    hc, bc, be = hist(
        data,
        nbins=nbins,
        bw=bw,
        min_bw=vmin,
        max_bw=vmax,
        density=density,
    )

    new_ds = copy.deepcopy(ds)
    for idx, bmax in enumerate(be[1:]):
        bmin = be[idx]
        this_bc = int(bc[idx]) if round_bins else bc[idx]
        ts_tp = gut.get_range_ds(ds, bmin, bmax).time
        new_ds.loc[dict(time=ts_tp)] = this_bc
    new_ds = xr.where(new_ds > vmax, vmax, new_ds)
    new_ds = xr.where(new_ds < vmin, vmin, new_ds)
    return new_ds


def normalize(data, min=0, max=1):
    # norm = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
    norm = minmax_scale(data.data, feature_range=(min, max), )
    if type(data) == xr.DataArray:
        norm = xr.DataArray(
            data=norm,
            dims=data.dims,
            coords=data.coords,
            name=data.name,
        )
    return norm


def count_occ(occ_arr, count_arr, rel_freq=False, norm_fac=1.):
    """Counts the occurences of certain objects (eg. floats, ints, strs etc.)

    Args:
        occ_arr (arr): arr of objects
        count_arr (arr): objects to count
        rel_freq (bool, optional): Give relative frequency. Defaults to False.
        norm_fac (int, optional): Normalization factor. Defaults to 1.

    Returns:
        arr: counts of objects
    """
    res_c_occ_arr = np.zeros((len(occ_arr), len(count_arr)))

    for idx, occ in enumerate(occ_arr):
        m_c_occ = res_c_occ_arr[idx]
        tot_num = len(occ)
        u, count = np.unique(occ,  return_counts=True)
        u = np.array(u, dtype=int)
        for iu, u_val in enumerate(u):
            cnt_idx = np.where(u_val == count_arr)[0]
            # print(u_val, count_arr, cnt_idx)
            idx_cnt_arr = int(cnt_idx)
            m_c_occ[idx_cnt_arr] = count[iu]
        if rel_freq:
            if tot_num > 0:
                m_c_occ /= tot_num
                if np.abs(np.sum(m_c_occ) - 1) > 0.1:
                    gut.myprint(
                        f'WARNING, rel freq not summed to 1 {np.sum(m_c_occ)}')
            else:
                m_c_occ = 0

    if norm_fac == 'max':
        norm_fac = np.max(res_c_occ_arr, axis=1)
    res_c_occ = np.mean(res_c_occ_arr, axis=0)/norm_fac

    return res_c_occ


def rank_data(data, axis=0, method='min'):
    data_rk = st.rankdata(
        data, axis=axis, method=method)
    if type(data) == xr.DataArray:
        data_rk = xr.DataArray(
            data=data_rk,
            dims=data.dims,
            coords=data.coords,
            name=data.name,
        )
    return data_rk


def cdf_data(data, axis=0):
    data_rk = rank_data(data, axis=axis)
    q_data_rk = data_rk/len(data)
    return q_data_rk


def standardize(dataset, axis=0):
    std_ds = st.zscore(dataset, axis=axis)  # Much much faster!
    # if type(dataset) == xr.DataArray:
    #     reload(gut)
    #     std_ds = gut.create_xr_ds(data=std_ds,
    #                               dims=dataset.dims,
    #                               coords=dataset.coords,
    #                               name=dataset.name)
    return std_ds
    # return (dataset - np.nanmean(dataset, axis=axis)) / (np.nanstd(dataset, axis=axis))


def standardize_along_time(data, dim='time'):
    """
    Standardizes an xarray DataArray along the 'time' dimension.

    Parameters
    ----------
    data : xarray.DataArray
        The input data array with dimensions (time, lon, lat).

    Returns
    -------
    xarray.DataArray
        A new DataArray with the same dimensions, where the data is standardized
        along the 'time' dimension. Each point in the (lon, lat) grid will have a
        time series with a mean of 0 and a standard deviation of 1.

    """

    # Step 1: Calculate the mean along the 'time' dimension
    mean = data.mean(dim=dim)

    # Step 2: Calculate the standard deviation along the 'time' dimension
    std = data.std(dim=dim)

    # Step 3: Standardize the data by subtracting the mean and dividing by the standard deviation
    standardized_data = (data - mean) / std

    return standardized_data


def standardize_dataset(dataset, dim='time'):
    vars = gut.get_vars(dataset)
    x_stack_vars = []
    for var in vars:
        gut.myprint(f'Standardize {var}')
        x_stack_vars.append(standardize_along_time(dataset[var], dim=dim))
    x_stack_vars = xr.merge(x_stack_vars)

    return x_stack_vars


def normalize_along_time(data, dim='time'):
    """
    Normalizes an xarray DataArray along the 'time' dimension.

    Normalization rescales the data such that the minimum value along the time dimension
    is 0 and the maximum value is 1, for each (lon, lat) point.

    Parameters
    ----------
    data : xarray.DataArray
        The input data array with dimensions (time, lon, lat).

    Returns
    -------
    xarray.DataArray
        A new DataArray with the same dimensions, where the data is normalized
        along the 'time' dimension. Each point in the (lon, lat) grid will have a
        time series with values ranging from 0 to 1.

    Example
    -------
    >>> import xarray as xr
    >>> data = xr.DataArray(np.random.rand(100, 50, 50), dims=["time", "lon", "lat"])
    >>> normalized_data = normalize_along_time(data)
    """

    # Step 1: Calculate the minimum value along the 'time' dimension
    min_val = data.min(dim=dim)

    # Step 2: Calculate the maximum value along the 'time' dimension
    max_val = data.max(dim=dim)

    # Step 3: Normalize the data by rescaling to [0, 1] along the 'time' dimension
    normalized_data = (data - min_val) / (max_val - min_val)

    return normalized_data


def normalize_dataset(dataset, dim='time'):
    """
    Normalizes an xarray dataset along the 'time' dimension.

    Normalization rescales the data such that the minimum value along the time dimension
    is 0 and the maximum value is 1, for each (lon, lat) point.

    Parameters
    ----------
    dataset : xarray.Dataset
        The input dataset with multiple variables, each with dimensions (time, lon, lat).

    Returns
    -------
    xarray.Dataset
        A new Dataset with the same variables, where the data is normalized
        along the 'time' dimension. Each point in the (lon, lat) grid will have a
        time series with values ranging from 0 to 1.

    Example
    -------
    >>> import xarray as xr
    >>> data = xr.Dataset(
    ...     {
    ...         "var1": ("time", np.random.rand(100)),
    ...         "var2": ("time", np.random.rand(100)),
    ...     }
    ... )
    >>> normalized_data = normalize_dataset(data)
    """

    # Step 1: Get the variable names in the dataset
    vars = gut.get_vars(dataset)

    # Step 2: Normalize each variable along the 'time' dimension
    normalized_vars = []
    for var in vars:
        gut.myprint(f'Normalize {var}')
        normalized_vars.append(normalize_along_time(dataset[var], dim=dim))

    # Step 3: Merge the normalized variables into a new dataset
    normalized_dataset = xr.merge(normalized_vars)

    return normalized_dataset


def vmin_vmax_array(data, vmin, vmax):
    data = np.array(data)
    vmin_idx = np.argwhere(data <= vmin)
    data[vmin_idx] = vmin
    vmax_idx = np.argwhere(data >= vmax)
    data[vmax_idx] = vmax

    return data


def normalize_minmax(dataarray, axis=None):
    """Normalize dataarray between 0, 1

    Args:
        dataarray ([type]): [description]
    """
    if axis is None:
        flatten = dataarray.stack(z=dataarray.dims)
    else:
        flatten = dataarray.stack(z=dataarray.dims[axis])
    norm_data = (
        (flatten - flatten.min(skipna=True)) /
        (flatten.max(skipna=True) - flatten.min(skipna=True))
    )
    return norm_data.unstack('z')


def normalize_to_gaussian_arr(arr, mode="mean"):
    ts_gauss_arr = []
    for ts in arr:
        ts_gauss = normalize_to_gaussian(arr=ts, mode=mode)
        ts_gauss_arr.append(ts_gauss)

    return np.array(ts_gauss_arr)


def normalize_to_gaussian(arr, mode="mean"):
    """
    Normalizes the data to a Gaussian distribution using quantiles.
    """
    n = len(arr)
    perc = percentileofscore
    arr_ = arr.copy()[~np.isnan(arr)]
    out = np.zeros(n)
    for i in range(n):
        if not np.isnan(arr[i]):
            out[i] = norm.ppf(perc(arr_, arr[i], mode) / 100.)
        else:
            out[i] = np.nan
    return out


# Spearman's Correlation
def calc_spearman(data, test='twosided', verbose=False):
    """Spearman correlation of the flattened and without NaNs object.
    """

    corr, pvalue_twosided = st.spearmanr(
        data, axis=0, nan_policy='propagate')

    if test == 'onesided':
        pvalue, zscore = onesided_test(corr)
    elif test == 'twosided':
        pvalue = pvalue_twosided
    else:
        raise ValueError('Choosen test statisics does not exist. Choose "onesided" '
                         + 'or "twosided" test.')
    if verbose:
        print(f"Created spearman correlation matrix of shape {np.shape(corr)}")

    return corr, pvalue


# Pearson's Correlatio
def calc_pearson(data, verbose=False):
    """Compute the Pearson correlation of the flattened and without NaNs object.

    Args:
        data (np.array): array of shape (n, m) where n is the number of samples and m the number of features.
        verbose (bool, optional): Verbose notification. Defaults to False.

    Returns:
        np.ndarray: correlation and p-value matrix of shape (m, m)
    """
    # Pearson correlation
    corr = np.corrcoef(data.T)
    assert corr.shape[0] == data.shape[1]

    # get p-value matrix
    # https://stackoverflow.com/questions/24432101/correlation-coefficients-and-p-values-for-all-pairs-of-rows-of-a-matrix
    # TODO: Understand and check the implementation
    rf = corr[np.triu_indices(corr.shape[0], 1)]
    df = data.shape[1] - 2
    ts = rf * rf * (df / (1 - rf * rf))
    pf = special.betainc(0.5 * df, 0.5, df / (df + ts))
    p = np.zeros(shape=corr.shape)
    p[np.triu_indices(p.shape[0], 1)] = pf
    p[np.tril_indices(p.shape[0], -1)
      ] = p.T[np.tril_indices(p.shape[0], -1)]
    p[np.diag_indices(p.shape[0])] = np.ones(p.shape[0])

    if verbose:
        print(f"Created pearson correlation matrix of shape {np.shape(corr)}")

    return corr, p


def get_corr_function(corr_method):
    """Returns the correlation function based on the choosen correlation method.

    Args:
        corr_method (str): string of the choosen correlation method

    Raises:
        ValueError: string of the choosen correlation method does not exist

    Returns:
        function: function to calculate the correlation
    """
    # spearman correlation
    if corr_method == 'spearman':
        return calc_spearman
    # pearson correlation
    elif corr_method == 'pearson':
        return calc_pearson
    else:
        raise ValueError("Choosen correlation method does not exist!")


def corr_function(corr_type='pearson'):
    """
    Returns the correlation function based on the specified correlation type.

    Parameters:
        corr_type (str): The type of correlation. Default is 'pearson'.

    Returns:
        function: The correlation function.

    Raises:
        ValueError: If the specified correlation type is not implemented.
    """
    if corr_type == 'spearman':
        corr_func = st.stats.spearmanr
    elif corr_type == 'pearson':
        corr_func = st.stats.pearsonr
    else:
        raise ValueError(f'Correlation type {corr_type} not implemented!')
    return corr_func


def onesided_test(corr):
    """P-values of one sided t-test of spearman correlation.
    Following: https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
    """
    n = corr.shape[0]
    f = np.arctanh(corr)
    zscore = np.sqrt((n-3)/1.06) * f
    pvalue = 1 - st.norm.cdf(zscore)

    return pvalue, zscore


def loghist(arr, nbins=None, density=True):
    """
    Returns the histogram counts on a logarithmic binning.
    """
    if nbins is None:
        nbins = __doane(np.log(arr))
    bins = np.logspace(np.log10(arr.min()),
                       np.log10(arr.max() + 0.01),
                       nbins + 1)
    hc, be = np.histogram(arr, bins=bins, density=density)
    bc = 0.5 * (be[1:] + be[:-1])

    return hc, bc, be


def hist(arr, nbins=None, bw=None,
         min_bw=None, max_bw=None,
         density=True):
    if nbins is None:
        nbins = __doane(arr)

    if bw is None:
        bins = np.linspace(arr.min(),
                           arr.max() + 0.01,
                           nbins + 1
                           )
    else:
        minb = min_bw if min_bw is not None else arr.min()
        maxb = max_bw if max_bw is not None else arr.max()
        bins = np.arange(minb,
                         maxb + 0.01,
                         bw)
        # print(bw, min_bw, max_bw, bins)
    hc, be = np.histogram(arr, bins=bins, density=density)
    bc = 0.5 * (be[1:] + be[:-1])

    return hc, bc, be


def __doane(arr):
    """
    Returns the number of bins according to Doane's formula.

    More info:
    https://en.wikipedia.org/wiki/Histogram#Number_of_bins_and_width
    """
    n = float(len(arr))
    g1 = skew(arr)
    sig_g1 = np.sqrt((6. * (n - 2)) / ((n + 1) * (n + 3)))
    nbins = int(np.ceil(1. + np.log2(n) + np.log2(1 + np.abs(g1) / sig_g1)))

    return nbins


def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance
    between two probability distributions
    """

    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (st.entropy(p, m) + st.entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance


def KS_test(data1, data2=None, test='norm'):
    if data2 is not None:
        KS_st, p_val = st.ks_2samp(data1, data2)
    else:
        KS_st, p_val = st.kstest(data1, test)

    return KS_st, p_val


def effective_sample_size(X, zdim=('lat', 'lon')):
    """Compute effective sample size by fitting AR-process to each location in time-series.

        n_eff = n * (1-coeff) / (1+coeff)

    Args:
        X (xr.Dataarray): Spatio-temporal data.
        order (int, optional): Order of process. Defaults to 1.

    Returns:
        nobs_eff (xr.Dataarray): Effective sample size.
    """
    X = X.stack(z=zdim)

    print("Fit AR1-process to each location to obtain the effective sample size.")
    arcoeff = []
    for loc in tqdm(X['z']):
        x_loc = X.sel(z=loc)

        if np.isnan(x_loc[0]):
            arcoeff.append(np.nan)
        else:
            mod = AutoReg(x_loc.data, 1)
            res = mod.fit()
            arcoeff.append(res.params[0])

    arcoeff = xr.DataArray(data=np.array(arcoeff),
                           dims=['z'], coords=dict(z=X['z']))

    nobs_eff = (len(X['time']) * (1-arcoeff) / (1 + arcoeff))
    return nobs_eff.unstack()


def fit_ar1(X, i):
    x_loc = X.sel(z=X['z'][i])
    if np.isnan(x_loc[0]):
        arcoeff = np.nan
    else:
        mod = AutoReg(x_loc.data, 1)
        res = mod.fit()
        arcoeff = res.params[0]
    return arcoeff, i


def effective_sample_size_parallel(X, zdim=('lat', 'lon')):
    """Compute effective sample size by fitting AR-process to each location in time-series.

        n_eff = n * (1-coeff) / (1+coeff)

    Args:
        X (xr.Dataarray): Spatio-temporal data.
        order (int, optional): Order of process. Defaults to 1.

    Returns:
        nobs_eff (xr.Dataarray): Effective sample size.
    """
    X = X.stack(z=zdim)
    print("Fit AR1-process to each location to obtain the effective sample size.")
    # Run in parallel
    n_processes = len(X['z'])
    results = Parallel(n_jobs=8)(
        delayed(fit_ar1)(X, i)
        for i in tqdm(range(n_processes))
    )
    # Read results
    arcoeff = []
    ids = []
    for r in results:
        coef, i = r
        arcoeff.append(coef)
        ids.append(i)
    # Sort z dimension to avoid errors in reshape
    sort_idx = np.sort(ids)
    arcoeff = xr.DataArray(data=np.array(arcoeff)[sort_idx],
                           dims=['z'], coords=dict(z=X['z']))

    nobs_eff = (len(X['time']) * (1-arcoeff) / (1 + arcoeff))
    return nobs_eff.unstack()


def ttest_field(X, Y, serial_data=False,
                weights=None, weight_threshold=0.3,
                zdim=('lat', 'lon')):
    """Point-wise t-test between means of samples from two distributions to test against
    the null-hypothesis that their means are equal.

    Args:
        X (xr.Dataarray): Samples of first distribution.
        Y (xr.Dataarray): Samples of second distribution to test against.
        serial_data (bool): If data is serial use effective sample size. Defaults to False.
        weights (xr.Dataarray): Weights of each time point of X. Defaults to None.
        weight_threshold (float): Threshold on probability weights,
            only used if weights are set. Defaults to 0.3.

    Returns:
        statistics (xr.Dataarray): T-statistics
        pvalues (xr.Dataarray): P-values
    """
    zdim = tuple(zdim)
    if weights is not None:
        # Weighted mean
        X_weighted = X.weighted(weights)
        mean_x = X_weighted.mean(dim='time').stack(z=tuple(zdim))
        std_x = X_weighted.std(dim='time').stack(z=tuple(zdim))
        # Use threshold on weights to crop X for a realistic sample size
        ids = np.where(weights.data >= weight_threshold)[0]
        X = X.isel(time=ids)
    else:
        mean_x = X.mean(dim='time', skipna=True).stack(z=tuple(zdim))
        std_x = X.std(dim='time', skipna=True).stack(z=tuple(zdim))

    mean_y = Y.mean(dim='time', skipna=True).stack(z=tuple(zdim))
    std_y = Y.std(dim='time', skipna=True).stack(z=zdim)
    if serial_data:
        nobs_x = effective_sample_size_parallel(
            X, zdim=zdim).stack(z=zdim).data
        nobs_y = effective_sample_size_parallel(
            Y, zdim=zdim).stack(z=zdim).data
    else:
        nobs_x = len(X['time'])
        nobs_y = len(Y['time'])

    statistic, pvalues = st.ttest_ind_from_stats(
        mean_x.data, std_x.data, nobs_x,
        mean_y.data, std_y.data, nobs_y,
        equal_var=False,
        alternative='two-sided')
    # Convert to xarray
    pvalues = xr.DataArray(data=pvalues, coords=mean_x.coords)

    return mean_x.unstack(), pvalues.unstack()


def field_significance_mask(pvalues, alpha=0.05,
                            corr_type="dunn",
                            zdim=('lat', 'lon')):
    """Create mask field with 1, np.NaNs for significant values
    using a multiple test correction.

    Args:
        pvalues (xr.Dataarray): Pvalues
        alpha (float, optional): Alpha value. Defaults to 0.05.
        corr_type (str, optional): Multiple test correction type. Defaults to "dunn".

    Returns:
        mask (xr.Dataarray): Mask
    """
    if corr_type is not None:
        pval_dims = tuple(pvalues.dims)
        if pval_dims != zdim:
            zdim = tuple(pvalues.dims)
            print(zdim)
        pvals_flat = pvalues.stack(z=zdim)
        mask_flat = xr.DataArray(data=np.zeros(len(pvals_flat), dtype=bool),
                                 coords=pvals_flat.coords)
        ids = holm(pvals_flat.data, alpha=alpha, corr_type=corr_type)
        mask_flat[ids] = True
        mask = mask_flat.unstack()
    else:
        # mask 1 where significant
        mask = xr.where(pvalues < alpha, True, False)

    return mask


def sig_multiple_field(sig_mask_x, sig_mask_y, method='or'):
    """Combine two significance masks.

    Args:
        sig_mask_x (xr.Dataarray): Significance mask of first field.
        sig_mask_y (xr.Dataarray): Significance mask of second field.
        method (str, optional): Method to combine significance masks. Defaults to 'or'.

    Returns:
        sig_mask (xr.Dataarray): Combined significance mask.
    """
    if method == 'or':
        sig_mask = xr.where(sig_mask_x | sig_mask_y, True, False)
    elif method == 'and':
        sig_mask = xr.where(sig_mask_x & sig_mask_y, True, False)
    else:
        raise ValueError(f'Method {method} not implemented!')

    return sig_mask


def polyfit_regressor(data_array, predictor,  order=1):
    """
    Perform polynomial regression on a time series in an xarray DataArray with dimensions (time, lon, lat)
    using a single time series as a predictor.

    Parameters
    ----------
    data_array: xarray.DataArray
        A DataArray with dimensions (time, lon, lat) containing the time series data.
    order: int
        The order of the polynomial to fit.
    predictor: xarray.DataArray
        A DataArray with dimensions (time) containing the single time series predictor.

    Returns
    -------
    regressed: xarray.DataArray
        A DataArray with dimensions (time, lon, lat) containing the regressed time series data.

    Raises
    ------
    ValueError
        If the time dimension of `data_array` and `predictor` do not match.
    """
    # check if the time dimensions match
    if data_array['time'].shape[0] != predictor['time'].shape[0]:
        raise ValueError(
            'The time dimensions of `data_array` and `predictor` do not match.')

    regressed_arr = xr.full_like(data_array, np.nan)

    for lon in data_array['lon']:
        for lat in data_array['lat']:
            # Do something with the data_array.sel(lon=lon, lat=lat) time series
            this_ts = data_array.sel(lon=lon, lat=lat)
            coef = np.polyfit(predictor, this_ts, order)
            regressed_arr.loc[dict(lon=lon, lat=lat)
                              ] = np.polyval(coef, this_ts)

    return regressed_arr


def get_values_above_val(dataarray, val=None, q=None, dim='time'):
    """Return all values in the input xarray that are above a value.
    The median is taken if no value is given.

    Parameters:
    -----------
    dataarray : xarray.DataArray
        The input data array to get values from.

    Returns:
    --------
    xarray.DataArray
        An xarray object containing only the values that are above the median.

    Raises:
    -------
    ValueError:
        If the input data array does not have a time dimension.
    """
    if dim not in dataarray.dims:
        raise ValueError(f"Input data array must have a {dim} dimension")

    if val == 'mean':
        val = val_ = dataarray.mean(dim=dim)
    elif q is not None:
        if q < 1 and q > 0.5:
            val = dataarray.quantile(q=q, method='lower')
            val_ = dataarray.quantile(q=(1-q), method='higher')
        else:
            raise ValueError(f'q={q} has to be between 0.5 and 1')
    else:
        val = val_ = dataarray.median(dim=dim) if val is None else val
    above_val = dataarray.where(dataarray >= val).dropna(dim=dim)
    below_val = dataarray.where(dataarray <= val_).dropna(dim=dim)
    between_val = gut.diff_xarray(arr1=dataarray, arr2=above_val)
    between_val = gut.diff_xarray(arr1=between_val, arr2=below_val)

    return {
        'val': float(val),
        'val_': float(val_),
        'above': above_val,
        'below': below_val,
        'between': between_val
    }


def find_and_sort_peaks(time_series):
    """
    Find peaks in a time series and return the peak indices sorted by amplitude in descending order.

    Parameters:
    time_series (xarray.DataArray): The input time series as an xarray DataArray.

    Returns:
    np.ndarray: An array containing peak indices sorted by amplitude (highest to lowest).
    """
    # Ensure time_series is a 1D numpy array
    if not isinstance(time_series, np.ndarray):
        time_series = time_series.values

    # Find peaks using the find_peaks function from scipy
    peaks, _ = find_peaks(time_series)

    # Sort the peaks by amplitude in descending order
    sorted_peaks = sorted(peaks, key=lambda x: -time_series[x])

    return np.array(sorted_peaks)
