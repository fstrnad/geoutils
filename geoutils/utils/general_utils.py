"""General Util functions."""
from collections import Counter
import pickle
import cftime
from itertools import combinations_with_replacement
from scipy.signal import find_peaks
import contextlib
import os
import numpy as np
import xarray as xr
SEED = 42


def myprint(str, verbose=True):
    if verbose:
        print(str, flush=True)
    return


def get_dimensions(ds):
    dims = ds.dims
    dim_dict = {}
    for dim in dims:
        dim_dict[dim] = len(ds[dim])
    return dim_dict


def is_single_tp(tps):
    """Test whether an object of tps is an array of time points
    or if it is a single time point.

    Args:
        tps (np.array): array of time points, if xr.Dataarray it is casted to np.array

    Returns:
        bool: True if single time point
    """
    if isinstance(tps, xr.DataArray):
        tps = tps.time
        if len(np.array([tps.time.data]).shape) == 1:
            return True
    else:
        if len(np.array([tps]).shape) == 1:
            return True

    return False


def is_datetime360(time):
    if not is_single_tp(tps=time):
        time = time[0]
    return isinstance(time, cftime._cftime.Datetime360Day)


def compare_lists(*lists):
    """Checks if the lists are equal.

    Returns:
        bool: True if lists are equal
    """
    counters = map(Counter, lists)

    try:
        first_counter = next(counters)
    except StopIteration:
        return True

    return all(first_counter == counter for counter in counters)


def get_vars(ds):
    vars = list(ds.keys())
    return vars


def get_dims(ds):
    vars = list(ds.dims)  # works for xr.dataset and xr.dataarray
    return vars


@contextlib.contextmanager
def temp_seed(seed=SEED):
    """Set seed locally.
    Usage:
    ------
    with temp_seed(42):
        np.random.randn(3)

    Parameters:
    ----------
    seed: int
        Seed of function, Default: SEED
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_all_combs(arr):
    return list(combinations_with_replacement(arr, r=2))


def count_elements(arr):
    """Returns a dictionary of elements in array and their
    corresponding counts.py

    Args:
        arr (list): list of elements

    Returns:
        dict: dictionary (element: Counter)
    """
    unique, counts = np.unique(arr, return_counts=True)
    this_arr_count = dict(zip(unique, counts))
    return this_arr_count


def tryToCreateDir(d):
    dirname = os.path.dirname(d)
    try:
        os.makedirs(dirname)
    except FileExistsError:
        print("Directory ", d,  " already exists")

    return None


def get_index_array(arr, item):
    arr_tmp = np.array(arr)
    if item not in arr_tmp:
        myprint(f'{item} not in array {arr}!')
        return None
    else:
        idx_val = np.where(arr_tmp == item)[0]
        if len(idx_val) > 1:
            idx = int(idx_val[0])  # Always choose first array item
        else:
            idx = int(idx_val)
        return idx


def find_nearest(array, value, min=False, max=False):
    array = np.asarray(array)
    if min:
        array_min = np.where(array < value, array, np.nan)
        idx = np.nanargmin(np.abs(array_min - value))
    elif max:
        array_max = np.where(array > value, array, np.nan)
        idx = np.nanargmin(np.abs(array_max - value))
    else:
        idx = (np.abs(array - value)).argmin()

    return array[idx]


def check_range(arr_range, compare_arr_range):
    min_arr = min(arr_range)
    max_arr = max(arr_range)

    min_carr = min(compare_arr_range)
    max_carr = max(compare_arr_range)

    if min_arr <= min_carr or max_arr >= max_carr:
        return False
    else:
        return True


def check_range_val(val, arr_range):
    min_arr = min(arr_range)
    max_arr = max(arr_range)

    if min_arr < val and max_arr > val:
        return True
    else:
        return False


def get_range_ds(ds, min_val, max_val):
    new_da = ds.where((ds >= min_val) & (ds <= max_val), drop=True)
    return new_da


def find_local_max(data):

    peak_idx, _ = find_peaks(data)
    peak_idx = np.array(peak_idx, dtype=int)
    peak_val = data[peak_idx]
    if len(peak_idx) == 0:
        peak_idx = []
        peak_val = []
        max_val = np.max(data)
    else:
        max_val = np.max(peak_val)

    max_idx = get_index_array(arr=data, item=max_val)
    return {'idx': peak_idx,
            'val': peak_val,
            'max': max_val,
            'max_idx': max_idx}


def find_local_min(data):
    # same as local max but *-1
    peak_idx, _ = find_peaks(-1*data)
    peak_idx = np.array(peak_idx, dtype=int)
    peak_val = data[peak_idx]
    if len(peak_idx) == 0:
        peak_idx = []
        peak_val = []
        # Because we look for global minimum at the border
        max_val = np.min(data)
    else:
        max_val = np.max(peak_val)

    max_idx = get_index_array(arr=data, item=max_val)
    return {'idx': peak_idx,
            'val': peak_val,
            'min': max_val,
            'min_idx': max_idx}


def find_local_min_max_xy(data, x):
    if len(data) != len(x):
        raise ValueError(f'Error data and x not of the same length!')
    max_dict = find_local_extrema(arr=data)

    max_dict['x_local_max'] = x[max_dict['local_maxima_indices']]
    max_dict['x_max'] = x[max_dict['max_idx']]
    max_dict['x_local_min'] = x[max_dict['local_maxima_indices']]
    max_dict['x_min'] = x[max_dict['min_idx']]

    return max_dict


def find_local_extrema(arr):
    """
    Returns information about the local and global extrema of a given numpy array.

    Parameters:
    -----------
    arr : numpy array
        The input array for which the extrema should be found.

    Returns:
    --------
    A dictionary containing the following information:
    {
        "local_minima": list of local minima values,
        "local_maxima": list of local maxima values,
        "local_minima_indices": list of indices of local minima,
        "local_maxima_indices": list of indices of local maxima,
        "global_minimum": global minimum value,
        "global_maximum": global maximum value,
        "global_minimum_index": index of global minimum,
        "global_maximum_index": index of global maximum
    }
    """

    # Find local minima and maxima
    local_minima = np.where((arr[:-2] > arr[1:-1])
                            & (arr[1:-1] < arr[2:]))[0] + 1
    local_maxima = np.where((arr[:-2] < arr[1:-1])
                            & (arr[1:-1] > arr[2:]))[0] + 1

    # Find global minima and maxima
    global_minimum_index = np.argmin(arr)
    global_minimum = arr[global_minimum_index]
    global_maximum_index = np.argmax(arr)
    global_maximum = arr[global_maximum_index]

    # Return dictionary of results
    return {
        "local_minima": arr[local_minima],
        "local_maxima": arr[local_maxima],
        "local_minima_indices": local_minima,
        "local_maxima_indices": local_maxima,
        "min": global_minimum,
        "max": global_maximum,
        "min_idx": global_minimum_index,
        "max_idx": global_maximum_index
    }


def find_local_min_xy(data, x):
    if len(data) != len(x):
        raise ValueError(f'Error data and x not of the same length!')
    max_dict = find_local_min(data=data)
    max_dict['x'] = x[max_dict['idx']]
    max_dict['x_min'] = x[max_dict['min_idx']]

    return max_dict


def get_locmax_of_score(ts, q=0.95):
    q_value = np.quantile(ts, q)
    peak_idx, _ = find_peaks(ts, height=q_value, distance=1, prominence=1)
    peak_val = ts[peak_idx]
    return peak_val, peak_idx


def get_locmax_of_ts(ts, q=0.95):
    """Gets the maximum timepoints of xr.Dataarray time series for given quantile.

    Args:
        ts (xr.Dataarray): dataarray that contains the different time series.
        q (float, optional): qunatile value above the values are chosen. Defaults to 0.95.

    Returns:
        xr.Dataarray: returns the time points as np.datetime64
    """
    q_value = np.quantile(ts, q)
    peak_idx, _ = find_peaks(ts, height=q_value,
                             distance=1,
                             #  prominence=1
                             )
    peak_ts = ts[peak_idx]

    return peak_ts


def get_locmax_composite_tps(ts, q=0.95, distance=3):
    """Gets the maximum timepoints of xr.Dataarray time series for given quantile.

    Args:
        ts (xr.Dataarray): dataarray that contains the different time series.
        q (float, optional): qunatile value above the values are chosen. Defaults to 0.95.

    Returns:
        xr.Dataarray: returns the time points as np.datetime64
    """
    q_value = np.quantile(ts, q)
    peak_idx, _ = find_peaks(ts, height=q_value,
                             # Minimum distance of peaks (to avoid problems in composites)
                             distance=distance,
                             #  prominence=1
                             )
    roots_idx = find_roots(y=ts.data, x=None, y_0=q_value)
    # Get Peaks that are as well roots
    rp_idx = np.intersect1d(peak_idx, roots_idx)
    idx_sp = np.array([], dtype=int)
    idx_ep = np.array([], dtype=int)
    for rp in rp_idx:
        if ts[rp-1] < q_value:
            idx_sp = np.append(idx_sp, int(rp-1))
        else:
            idx_sp = np.append(idx_sp, int(
                find_nearest(roots_idx, rp, min=True)))
        if ts[rp+1] < q_value:
            idx_ep = np.append(idx_ep, int(rp+1))
        else:
            idx_ep = np.append(idx_ep, int(
                find_nearest(roots_idx, rp, max=True)))

    # Get Peaks that are not roots
    idx_peak_not_root = np.setxor1d(peak_idx, rp_idx)
    idx_root_not_peak = np.unique(np.setxor1d(roots_idx, rp_idx))
    # roots_idx_not_peak = np.unique(np.setxor1d(roots_idx, rp_idx))  # sort as well
    for pidx in idx_peak_not_root:
        rp_left = find_nearest(idx_root_not_peak, pidx, min=True)
        rp_right = find_nearest(idx_root_not_peak, pidx, max=True)
        idx_sp = np.append(idx_sp, rp_left)
        idx_ep = np.append(idx_ep, rp_right)

    idx_sp = np.unique(idx_sp)  # sorts as well!
    idx_ep = np.unique(idx_ep)

    # Account for roots being below the threshold
    for idx, rp in enumerate(idx_sp):
        this_rp = rp - 1 if ts[rp] > q_value else rp
        this_rp = rp if ts[this_rp] > ts[rp] else this_rp
        idx_sp[idx] = this_rp
    for idx, rp in enumerate(idx_ep):
        this_rp = rp + 1 if ts[rp] > q_value else rp
        this_rp = rp if ts[this_rp] > ts[rp] else this_rp
        idx_ep[idx] = this_rp

    peak_ts = ts[peak_idx]
    sp_ts = ts[idx_sp]
    ep_ts = ts[idx_ep]
    # print(roots_idx)
    # print(idx_sp)
    # print(peak_idx)
    # print(idx_ep)

    return {'peaks': peak_ts, 'sps': sp_ts, 'eps': ep_ts}


def find_roots(y, x=None, y_0=0):
    """Find roots in an array of values.


    Args:
        x (list): x values, if None set to [0, 1... len(y)]
        y (list): y values

    Returns:
        _type_: list of roots
    """
    x_ax = np.arange(len(y)) if x is None else x
    if y_0 != 0:
        y = y - y_0
    s = np.abs(np.diff(np.sign(y))).astype(bool)

    x_roots = x_ax[:-1][s] + np.diff(x_ax)[s]/(np.abs(y[1:][s]/y[:-1][s])+1)

    if x is None:
        x_roots = np.unique(np.array(np.round(x_roots), dtype=int))

    # Avoid consecutive roots, needed for integer based time series
    x_roots_up = x_roots + 1
    idx_up = np.argwhere(np.in1d(x_roots, x_roots_up)).flatten()
    x_roots[idx_up] = x_roots[idx_up] + 2  # Delete and add 1 further

    return np.unique(x_roots)


def get_exponent10(x, verbose=False):
    if not isinstance(x, int) and not isinstance(x, float):
        ValueError(f'Input {x} has to be int or float!')
    if np.abs(x) == 0.:
        if verbose:
            myprint(f'WARNING: input {x} is zero!')
        return 0
    exp = int(np.floor(np.log10(np.abs(x))))
    return exp


def get_quantile_of_ts(ts, q=0.9, verbose=False):
    q_value = np.quantile(ts, q)
    if q >= 0.5:
        q_ts = ts[np.where(ts >= q_value)[0]]
    else:
        q_ts = ts[np.where(ts <= q_value)[0]]

    myprint(f"Q-Value: {q_value}!", verbose=verbose)

    return q_ts


def get_varnames_ds(ds):
    return list(ds.keys())


def get_source_target_corr(corr, sids):
    source_corr = corr[0:len(sids), 0:len(sids)]
    target_corr = corr[np.ix_(np.arange(len(sids), len(corr)),
                              np.arange(len(sids), len(corr))
                              )
                       ]
    source_target_corr = corr[np.ix_(np.arange(0, len(sids)),
                                     np.arange(len(sids), len(corr))
                                     )
                              ]
    return {'source': source_corr,
            'target': target_corr,
            'source_target': source_target_corr}


def mk_grid_array(data, x_coords=None, y_coords=None,
                  x_coord_name='x', y_coord_name='y',
                  name='data',
                  **kwargs):

    if x_coords is None:
        x_coords = np.arange(0, data.shape[1])
    if y_coords is None:
        y_coords = np.arange(0, data.shape[0])

    xr_2d = xr.DataArray(
        data=np.array(data),
        dims=[y_coord_name, x_coord_name],  # x and y are in data.shape(y,x)
        coords={
            x_coord_name: x_coords,
            y_coord_name: y_coords,
        },
        name=name
    )

    return xr_2d


def remove_tids_sids(sids, tids):
    t_in_s = np.in1d(tids, sids)
    num_s_in_t = np.count_nonzero(t_in_s)
    if num_s_in_t > 0:
        print(f'Remove {num_s_in_t} targets that are in source!')
        # remove target links that are as well in source
        tids = tids[~np.in1d(tids, sids)]
    return tids

# These functions below are taken from
# https://stackoverflow.com/questions/50299172/python-range-or-numpy-arange-with-end-limit-include


def cust_range(*args, rtol=1e-05, atol=1e-08,
               include=[True, False],
               dtype=float):
    """
    Combines numpy.arange and numpy.isclose to mimic
    open, half-open and closed intervals.
    Avoids also floating point rounding errors as with
    >>> numpy.arange(1, 1.3, 0.1)
    array([1. , 1.1, 1.2, 1.3])

    args: [start, ]stop, [step, ]
        as in numpy.arange
    rtol, atol: floats
        floating point tolerance as in numpy.isclose
    include: boolean list-like, length 2
        if start and end point are included
    """
    # process arguments
    if len(args) == 1:
        start = 0
        stop = args[0]
        step = 1
    elif len(args) == 2:
        start, stop = args
        step = 1
    else:
        assert len(args) == 3
        start, stop, step = tuple(args)

    # determine number of segments
    n = (stop-start)/step + 1

    # do rounding for n
    if np.isclose(n, np.round(n), rtol=rtol, atol=atol):
        n = np.round(n)

    # correct for start/end is exluded
    if not include[0]:
        n -= 1
        start += step
    if not include[1]:
        n -= 1
        stop -= step
    return np.linspace(start, stop, int(n), dtype=dtype)


def crange(*args, **kwargs):
    return cust_range(*args, **kwargs, include=[True, True])


def orange(*args, **kwargs):
    return cust_range(*args, **kwargs, include=[True, False])


def get_random_numbers_no_neighboring_elems(min_num, max_num, amount):
    """Generates amount random numbers in [min_num,..,max_num] that do not
    include neighboring numbers."""

    # this is far from exact - it is best to have about 5+ times the amount
    # of numbers to choose from - if the margin is too small you might take
    # very long to get all your "fitting numbers" as only about 1/4 of the range
    # is a viable candidate (worst case):
    #   [1 2 3 4 5 6 7 8 9 10]: draw 2 then 5 then 8 and no more are possible
    if (max_num-min_num) // 5 < amount:
        raise ValueError(f"Range too small - increase given range.")

    rnd_set = set()
    while len(rnd_set) != amount:
        a = np.random.randint(min_num, max_num)
        if not {a-1, a, a+1} & rnd_set:  # set intersection: empty == False == no commons
            rnd_set.add(a)
    return np.array(sorted(rnd_set))


def get_a_not_b(a, b):
    """Returns the elements that are in a but not in b.

    Args:
        a (np.array): array of elements
        b (np.array): array of elements
    """
    a = np.array(a)
    b = np.array(b)
    return a[~np.in1d(a, b)]


def get_intersect_a_b(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.intersect1d(a, b)


def get_job_array_ids(def_id=0):
    # for job array on slurm cluster
    try:
        min_job_id = int(os.environ['SLURM_ARRAY_TASK_MIN'])
        max_job_id = int(os.environ['SLURM_ARRAY_TASK_MAX'])
        job_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
        num_jobs = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
        num_jobs = max_job_id
        print(
            f"job_id: {job_id}/{num_jobs}, Min Job ID: {min_job_id}, Max Job ID: {max_job_id}",
            flush=True)

    except KeyError:
        job_id = 0
        num_jobs = 1
        print("Not running with SLURM job arrays, but with manual id: ", job_id,
              flush=True)

    return job_id, num_jobs


def get_job_id(def_id=0):
    # for job array
    job_id, _ = get_job_array_ids(def_id=def_id)
    return job_id


def mk_dict_2_lists(key_lst, val_lst):
    dictionary = dict(zip(key_lst, val_lst))
    return dictionary


def zip_2_lists(list1, list2):
    return np.array(list(zip(list1, list2)))


def create_xr_ds(data, dims, coords, name=None):
    return xr.DataArray(data=data,
                        dims=dims,
                        coords=coords,
                        name=name
                        )


def nans_array(size):
    """Creates an array filled of np.nans.

    Args:
        size (int, tuple): int or tuple of int for dimension of array.

    Returns:
        np.ndarray: return array filled of np.nans of size size.
    """
    a = np.empty(size)
    a[:] = np.nan
    return a


def count_nans(x):
    return np.count_nonzero(np.isnan(x.data))


def contains_nan(arr):
    if np.isnan(arr).any():
        return True
    else:
        return False


def remove_nans(x):
    if isinstance(x, list):
        x = np.array(x)
    return x[~np.isnan(x)]


def remove_duplicates_arr(x):
    return np.unique(x)


def get_name_da(da):
    name = da.name
    return name


def rename_da(da, name):
    old_name = get_name_da(da)
    myprint(f'Rename {old_name} to {name}!')
    da = da.rename(name)
    return da


def merge_datasets(ds1, ds2):
    """
    Merge two xarray Dataset objects into a single Dataset object.

    Parameters:
    -----------
    ds1, ds2 : xarray.Dataset
        The two Dataset objects to be merged.

    Returns:
    --------
    merged_ds : xarray.Dataset
        The merged Dataset object.

    Raises:
    -------
    ValueError:
        If the dimensions (lat, lon, time) are not consistent between ds1 and ds2.
    """
    # Check if the dimensions are consistent between ds1 and ds2
    for dim in ["lat", "lon", "time"]:
        if not ds1[dim].equals(ds2[dim]):
            raise ValueError(
                f"Inconsistent dimension {dim} between datasets: {ds1[dim]} vs {ds2[dim]}!")

    # Merge the two Dataset objects into a single Dataset object
    merged_ds = xr.merge([ds1, ds2])

    return merged_ds


def flatten_array(dataarray, mask=None, time=True, check=False):
    """Flatten and remove NaNs.
    """

    if mask is not None:
        idx_land = np.where(mask.data.flatten() == 1)[0]
    else:
        idx_land = None
    if time is False:
        buff = dataarray.data.flatten()
        buff[np.isnan(buff)] = 0.0  # set missing data to climatology
        data = buff[idx_land] if idx_land is not None else buff[:]
    else:
        data = []
        for idx, t in enumerate(dataarray.time):
            buff = dataarray.sel(time=t.data).data.flatten()
            buff[np.isnan(buff)] = 0.0  # set missing data to climatology
            data_tmp = buff[idx_land] if idx_land is not None else buff[:]
            data.append(data_tmp)

    # check
    if check is True:
        num_nonzeros = np.count_nonzero(data[-1])
        num_landpoints = sum(~np.isnan(mask.data.flatten()))
        myprint(
            f"The number of non-zero datapoints {num_nonzeros} "
            + f"should approx. be {num_landpoints}."
        )

    return np.array(data)


def diff_xarray(arr1, arr2):
    """
    Returns an xarray object that contains all values that are in arr1 but not in arr2.

    Parameters:
    -----------
    arr1: xarray DataArray
        The first array to compare.
    arr2: xarray DataArray
        The second array to compare.

    Returns:
    --------
    xarray DataArray
        The difference of arr1 and arr2.
    """

    # Compute the difference
    unique_values = arr1[np.isin(arr1, arr2, invert=True)]

    return unique_values


def sort_by_frequency(arr: np.ndarray) -> np.ndarray:
    """
    Sorts the input NumPy array by the frequency of each element in descending order.

    Parameters:
    arr (np.ndarray): A NumPy array of objects (eg. integers)

    Returns:
    np.ndarray: A NumPy array of integers, sorted by the frequency of each element in descending order.

    Example:
    >>> arr = np.array([1, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 7, 7, 7, 7, 7, 8, 9, 9])
    >>> sorted_by_frequency = sort_by_frequency(arr)
    >>> print(sorted_by_frequency)
    [4 7 3 5 9 1 2 6 8 0 10]
    """
    # Get the unique elements and their counts using NumPy's unique function
    unique, counts = np.unique(arr, return_counts=True)

    # Sort the counts array in descending order using NumPy's argsort function
    sorted_indices = np.argsort(-counts)

    # Sort the unique elements array using the sorted indices
    return unique[sorted_indices]


def remove_non_dim_coords(ds):
    """
    Removes all coordinates that are not a dimension of an xarray dataset.

    Parameters:
    ds (xarray.Dataset): The input xarray dataset.

    Returns:
    xarray.Dataset: The output xarray dataset with only dimension coordinates.
    """
    dims = set(ds.dims)
    coords = set(ds.coords)

    non_dim_coords = coords - dims

    return ds.drop(non_dim_coords)


def add_compliment(arr):
    """
    Takes an array of integers and appends the negative of each integer (except 0)
    to the array, ensuring that the resulting array starts with the lowest integer,
    ends with the highest integer, and contains 0 in the middle.

    Args:
        arr (list): The input array of integers.

    Returns:
        list: The updated array with negative integers appended, sorted in ascending order,
              with 0 in the middle.

    """
    new_arr = []
    for num in arr:
        if num != 0:
            new_arr.append(num)
            new_arr.append(-1 * num)
    new_arr.sort()
    if 0 not in new_arr:
        new_arr.insert(len(new_arr) // 2, 0)
    return new_arr


def make_arr_negative(arr):
    """Gets for an array of integers for every item the negative compliment

    Args:
        arr (np.array): list or array

    Returns:
        np.array: sorted list of negative items, always add 0 at the end.
    """
    new_arr = -1*np.abs(arr)
    new_arr.sort()
    new_arr = np.append(new_arr, [0])
    new_arr = np.unique(new_arr)

    return new_arr


def split_array_by_half(arr, keyword):
    """
    Splits the input array into either the first or second half based on the keyword.

    Args:
        arr (list): The input array of items.
        keyword (str): The keyword specifying which half to return ('first' or 'second').

    Returns:
        list: The first or second half of the input array, based on the keyword.

    Examples:
        >>> my_array = [1, 2, 3, 4, 5, 6, 7, 8]
        >>> result = split_array_by_half(my_array, 'first')
        >>> print(result)
        [1, 2, 3, 4]

        >>> result = split_array_by_half(my_array, 'second')
        >>> print(result)
        [5, 6, 7, 8]
    """

    midpoint = len(arr) // 2
    print(midpoint)
    if keyword == 'first' or keyword == 1:
        return arr[:midpoint] if len(arr) % 2 == 0 else arr[:midpoint + 1]
    elif keyword == 'second' or keyword == -1:
        return arr[midpoint:] if len(arr) % 2 == 1 else arr[midpoint -1:]
    else:
        raise ValueError("Keyword must be either 'first' or 'second'.")
