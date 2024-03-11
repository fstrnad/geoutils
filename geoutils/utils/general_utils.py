"""General Util functions."""
from collections import Counter
from pprint import pprint
import cftime
from itertools import combinations_with_replacement
from scipy.signal import find_peaks
import contextlib
import os
import numpy as np
import xarray as xr
SEED = 42


def myprint(str, verbose=True, lines=False, bold=False, italic=False, color=None):
    # ANSI escape codes for styling text
    if verbose:
        style = "\033[1m" if bold else ""
        italic_code = "\033[3m" if italic else ""
        reset_style = "\033[0m" if bold or italic else ""

        # ANSI escape codes for colors
        color_code = ""
        if color:
            colors = {
                'red': "\033[91m",
                'green': "\033[92m",
                'yellow': "\033[93m",
                'blue': "\033[94m",
                'magenta': "\033[95m",
                'cyan': "\033[96m",
                'white': "\033[97m",
            }
            if color not in colors:
                raise ValueError(f"Invalid color: {color}! Available colors: {list(colors.keys())}")
                color_code = colors.get(color.lower(), "")  #

        styled_text = f"{style}{italic_code}{color_code}{str}{reset_style}"
        if lines:
            styled_text = pprint(styled_text)
        print(styled_text, flush=True)
        return styled_text
    else:
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


def get_locmax_composite_tps(ts, q=0.95, distance=3,
                             return_neighbors=False):
    """Gets the maximum timepoints of xr.Dataarray time series for given quantile.

    Args:
        ts (xr.Dataarray): dataarray that contains the different time series.
        q (float, optional): qunatile value above the values are chosen. Defaults to 0.95.

    Returns:
        xr.Dataarray: returns the time points as np.datetime64
    """
    q_value = np.quantile(ts, q)
    if np.isnan(q_value):
        raise ValueError(f'Quantile {q} is NaN!')
    peak_idx, _ = find_peaks(ts, height=q_value,
                             # Minimum distance of peaks (to avoid problems in composites)
                             distance=distance,
                             #  prominence=1
                             )
    peak_ts = ts[peak_idx]

    if return_neighbors:
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

        sp_ts = ts[idx_sp]
        ep_ts = ts[idx_ep]
    else:
        sp_ts = None
        ep_ts = None

    return {'peaks': peak_ts,
            'peak_idx': peak_idx,
            'sps': sp_ts,
            'eps': ep_ts,
            }


def invert_array(arr):
    """
    Inverts the values of a NumPy array containing only 0s and 1s.

    Args:
    arr (numpy.ndarray): The input NumPy array containing 0s and 1s.

    Returns:
    numpy.ndarray: A NumPy array with inverted values, where 0s are replaced by 1s
    and 1s are replaced by 0s.

    Raises:
    ValueError: If the input array contains values other than 0 and 1.

    Example:
    >>> input_array = np.array([[1, 0, 1], [0, 1, 0]])
    >>> inverted_array = invert_numpy_array(input_array)
    >>> print(inverted_array)
    [[0 1 0]
     [1 0 1]]
    """
    arr = np.array(arr)
    # Check if the input contains values other than 0 and 1
    if not np.all(np.logical_or(arr == 0, arr == 1)):
        raise ValueError("Input array must contain only 0s and 1s.")

    # Use NumPy's logical_not function to invert the values
    inverted_arr = np.logical_not(arr).astype(int)
    return inverted_arr


def reverse_array(input_array):
    """
    Reverses the order of elements in the input array.

    Args:
        input_array (list): The array to be reversed.

    Returns:
        list: The reversed array.
    """
    return input_array[::-1]


def set_neighbors_to_1(arr):
    """
    Sets the items to the left and right of each 1 in the input array to 1.

    Args:
    arr (numpy.ndarray): The input NumPy array containing 0s and 1s.

    Returns:
    numpy.ndarray: A modified array with items to the left and right of each 1 set to 1.

    Example:
    >>> input_array = np.array([[0, 1, 0], [1, 0, 1]])
    >>> modified_array = set_neighbors_to_1(input_array)
    >>> print(modified_array)
    [[1 1 1]
     [1 1 1]]
    """
    # Ensure that the input is a NumPy array
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    # Create a mask for 1s in the input array
    ones_mask = arr == 1

    # Create a new array with the same shape as the input
    modified_arr = np.zeros_like(arr)

    # Iterate through the dimensions and set neighbors to 1
    for dim in range(arr.ndim):
        padded = np.pad(ones_mask, [(0, 0) if i != dim else (
            1, 1) for i in range(arr.ndim)], mode='constant')
        modified_arr += np.logical_or.reduce(padded, axis=dim, keepdims=True)

    return modified_arr


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


def get_quantile_of_ts(ts, q=0.9,
                       max_quantile=False,
                       return_indices=False,
                       verbose=False):
    q_value = np.quantile(ts, q)
    if q >= 0.5 or max_quantile:
        indices = np.where(ts >= q_value)[0]
    else:
        indices = np.where(ts < q_value)[0]
    q_ts = ts[indices]

    myprint(f"Q-Value: {q_value}!", verbose=verbose)
    if return_indices:
        return q_ts, indices
    else:
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
    dtype = type(args[2]) if len(args) > 2 else type(args[1])
    return cust_range(*args, **kwargs, include=[True, True],
                      dtype=dtype)


def orange(*args, **kwargs):
    return cust_range(*args, **kwargs, include=[True, False])


def custom_arange(start, end, step, include_end=True):
    """
    Generate an array of evenly spaced values within a given range.

    Parameters:
        start (float): The starting value of the range.
        end (float): The ending value of the range.
        step (float): The step size between values.

    Returns:
        numpy.ndarray: An array of evenly spaced values within the specified range.
    """
    if np.isclose(start + step * np.floor((end - start) / step), end):
        return np.arange(start, end + step, step)
    else:
        return np.arange(start, end, step)

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


def has_duplicate(arr):
    """
    Checks if any item occurs more than once in the array.

    Args:
    arr (list or numpy.ndarray): The input array.

    Returns:
    bool: True if any item occurs more than once, False otherwise.

    Example:
    >>> has_duplicate([1, 2, 3, 4, 1])
    True
    >>> has_duplicate([1, 2, 3, 4, 5])
    False
    """
    seen = set()
    for item in arr:
        if item in seen:
            return True
        seen.add(item)
    return False


def get_name_da(da):
    name = da.name
    return name


def rename_dim(da, dim, name):
    dims = get_dims(da)
    if name in dims:
        myprint(f'Dimension {name} already in {dims}!')
        return da
    if dim not in dims:
        raise ValueError(f'Dimension {dim} not in {dims}!')
    if not isinstance(name, str):
        raise ValueError(f'New name {name} has to be string!')
    da = da.rename({dim: name})
    return da


def assign_new_coords(da, dim, coords):
    """
    Assigns new coordinates to a given dimension of a DataArray.

    Args:
        da (xarray.DataArray): The DataArray to modify.
        dim (str): The dimension to assign new coordinates to.
        coords (list): The list of new coordinates.

    Returns:
        xarray.DataArray: The modified DataArray with new coordinates assigned to the specified dimension.

    Raises:
        ValueError: If the specified dimension is not present in the DataArray.
        ValueError: If the coordinates are not provided as a list.
        ValueError: If the length of the coordinates list is not equal to the length of the specified dimension.
    """
    if dim not in get_dims(da):
        raise ValueError(f'Dimension {dim} not in {get_dims(da)}!')
    if not isinstance(coords, list) and not isinstance(coords, np.ndarray):
        raise ValueError(f'Coordinates {coords} has to be list!')
    if len(coords) != len(da[dim]):
        raise ValueError(
            f'Length of coordinates {len(coords)} has to be equal to length of dimension {len(da[dim])}!')
    da = da.assign_coords({dim: coords})
    return da


def delete_non_dimension_attribute(dataarray, attribute_name, verbose=True):
    """
    Delete the given attribute from the coordinates if it is not a real dimension of the xarray DataArray.

    Parameters:
        dataarray (xarray.DataArray): The input xarray DataArray.
        attribute_name (str): The name of the attribute to be deleted.

    Returns:
        xarray.DataArray: The modified xarray DataArray.
    """
    if attribute_name in dataarray.dims:
        # Skip deletion if the attribute is a real dimension
        myprint(f'The attribute {attribute_name} is a dimension and cannot be deleted!',
                verbose=verbose)
        return dataarray
    elif attribute_name in dataarray.coords:
        # Delete the attribute from the coordinates
        myprint(f'Delete attribute {attribute_name} from coordinates!',)
        del dataarray.coords[attribute_name]
        return dataarray
    else:
        # Attribute not found in dims or coords
        raise ValueError(f"The attribute '{attribute_name}' not found in dimensions or coordinates.")


def delete_all_non_dimension_attributes(dataarray):
    """
    Delete all attributes from the coordinates that are not real dimensions of the xarray DataArray.

    Parameters:
        dataarray (xarray.DataArray): The input xarray DataArray.

    Returns:
        xarray.DataArray: The modified xarray DataArray.
    """
    attribute_names = list(dataarray.coords)
    for attribute_name in attribute_names:
        delete_non_dimension_attribute(dataarray, attribute_name, verbose=False)
    return dataarray


def rename_da(da, name):
    old_name = get_name_da(da)
    myprint(f'Rename {old_name} to {name}!')
    da = da.rename(name)
    return da


def rename_var_era5(ds, verbose=True, **kwargs):
    names = get_vars(ds=ds)

    if "precipitation" in names:
        ds = ds.rename({"precipitation": "pr"})
        myprint("Rename precipitation: pr!")
    if "precip" in names:
        ds = ds.rename({"precip": "pr"})
        myprint("Rename precip: pr!")
    if "tp" in names:
        ds = ds.rename({"tp": "pr"})
        myprint("Rename tp to pr!")
        ds['pr'] = ds['pr']*1000*24  # convert m/h to mm/day
        myprint("Convert m/h to mm/day!")
        ds['pr'].attrs.update({'units': 'mm/day'})
        ds.attrs.update({'long_name': 'Precipitation'})

    if 'sp' in names:
        # PS is surface pressure but named according to CF convention
        ds = ds.rename({"sp": "PS"})
        myprint("Rename sp to PS!")
        if ds['PS'].units == 'Pa':
            myprint("Compute surface pressure from Pa to hPa!")
            ds['PS'] /= 100  # compute Pa to hPa
            ds.attrs.update({'units': 'hPa'})
            ds['PS'].attrs.update({'units': 'hPa'})

    if "p86.162" in names:
        ds = ds.rename({"p86.162": "vidtef"})
        myprint(
            "Rename vertical integral of divergence of total energy flux to: vidtef!"
        )
    if "p71.162" in names:
        ds = ds.rename({"p71.162": "ewvf"})
        myprint(
            "Rename vertical integral of eastward water vapour flux to: ewvf!")

    if "p72.162" in names:
        ds = ds.rename({"p72.162": "nwvf"})
        myprint(
            "Rename vertical integral of northward water vapour flux to: ewvf!")

    if "z" in names:
        pot2height = kwargs.pop('pot2height', False)
        if pot2height:
            ds['z'].attrs.update({'units': 'm'})
            ds['z'].attrs.update({'long_name': 'Geopotential Height'})
            ds.rename({'z': 'zh'})
            g = 9.80665  # earth's accelaration at equator m/s2
            ds['zh'] = ds['zh'] / g  # convert to m
            myprint(
                f'Compute geopotential height from z! \n Multiply by 1/{g}',
                verbose=verbose)
    if "w" in names:
        myprint("Rename w to OMEGA!")
        ds = ds.rename({"w": "OMEGA"})
        reverse_w = kwargs.pop('reverse_w', True)
        myprint(f'Multiply w by factor {-1}!', verbose=reverse_w)
        ds['OMEGA'] = -1*ds['OMEGA'] if reverse_w else ds['OMEGA']

    if "ttr" in names:
        ds = ds.rename({"ttr": "olr"})
        myprint(
            "Rename top net thermal radiation (ttr) to: olr [W/m2]!\n"
            "Multiply by -1/3600!")
        ds['olr'] *= -1./3600  # convert to W/m2
        ds['olr'].attrs.update({'units': 'W/m2'})
        ds.attrs.update({'long_name': 'Outgoing longwave radiation'})
        ds['olr'].attrs.update(
            {'long_name': 'Outgoing longwave radiation'})

    if "ar_binary_tag" in names:
        ds = ds.rename({"ar_binary_tag": "ar"})
        myprint(
            "Rename ar_binary_tag (atmospheric rivers) to: ar!",
            verbose=verbose)
    return ds


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
        if not np.array_equal(ds1[dim].data, ds2[dim].data):
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
    if keyword == 'first' or keyword == 1:
        return arr[:midpoint] if len(arr) % 2 == 0 else arr[:midpoint + 1]
    elif keyword == 'second' or keyword == -1:
        return arr[midpoint:] if len(arr) % 2 == 1 else arr[midpoint - 1:]
    else:
        raise ValueError("Keyword must be either 'first' or 'second'.")


def delete_ds_attr(ds, attr):
    """
    Delete an attribute from a dataset.

    Args:
        ds (xarray.Dataset): The dataset from which the attribute will be deleted.
        attr (str): The name of the attribute to be deleted.
    """
    all_attrs = list(ds.attrs.keys())
    if attr not in all_attrs:
        myprint(f'WARNING! {attr} not in {all_attrs}! No attribute deleted!')
    else:
        del ds.attrs[attr]
    return ds


def delete_element_at_index(arr, i, axis=0):
    """
    Deletes the item at the specified index from the input array.

    Args:
        arr (list or numpy.ndarray): The input array of items.
        i (int): The index of the item to be deleted.

    Returns:
        same_type_as_input: The modified array after deleting the item at index i.

    Examples:
        >>> my_array = [1, 2, 3, 4, 5]
        >>> new_array = delete_element_at_index(my_array, 2)
        >>> print(new_array)
        [1 2 4 5]

        >>> import numpy as np
        >>> my_array = np.array([10, 20, 30, 40, 50])
        >>> new_array = delete_element_at_index(my_array, 0)
        >>> print(new_array)
        [20 30 40 50]
    """

    arr = np.array(arr)  # Convert input to NumPy array
    if i < len(arr):
        return np.delete(arr, i, axis=axis)
    else:
        raise IndexError(
            "Index out of range. The index must be less than the length of the array.")


def delete_element_from_arr(arr, item):
    """
    Deletes the specified item from the input array.

    Args:
        arr (list or numpy.ndarray): The input array of items.
        item (int or str): The item to be deleted.

    Returns:
        same_type_as_input: The modified array after deleting the specified item.

    Examples:
        >>> my_array = [1, 2, 3, 4, 5]
        >>> new_array = delete_element_from_arr(my_array, 2)
        >>> print(new_array)
        [1 3 4 5]

        >>> import numpy as np
        >>> my_array = np.array([10, 20, 30, 40, 50])
        >>> new_array = delete_element_from_arr(my_array, 20)
        >>> print(new_array)
        [10 30 40 50]
    """

    arr = np.array(arr)  # Convert input to NumPy array
    if item in arr:
        return np.delete(arr, np.where(arr == item))
    else:
        myprint(
            "Item not found in array. The item must be present in the array.")
        return arr


def convert_to_integers(arr):
    """
    Recursively checks if all elements in the input array (including NumPy arrays)
    and its nested arrays are integers and converts them if they are not.

    Args:
        arr (list, np.ndarray, int): The input array, which can be nested, and can be either a regular list or a NumPy array.

    Returns:
        list, np.ndarray, int: The input array (or nested arrays) with all elements converted to integers.
                              If the input is a NumPy array, the output will also be a NumPy array.

    Example:
        >>> input_array = ["1", [2.5, "3"], [4, ["5", 6]]]
        >>> convert_to_integers(input_array)
        [1, [2, 3], [4, [5, 6]]]
    """
    if isinstance(arr, list) or isinstance(arr, np.ndarray):
        if isinstance(arr, np.ndarray):
            arr = arr.tolist()
        return np.array([convert_to_integers(item) for item in arr])
    elif isinstance(arr, int):
        return arr
    else:
        try:
            return int(arr)
        except (ValueError, TypeError):
            return arr


def check_contains_substring(main_string, sub_string):
    """
    Check if a main string contains a given sub string.

    Args:
        main_string (str): The main string to check.
        sub_string (str): The sub string to search for.

    Returns:
        bool: True if the sub string is found in the main string, False otherwise.
    """
    if not isinstance(sub_string, list):
        sub_string = [sub_string]

    if not isinstance(main_string, str):
        raise ValueError("Main string must be a string. But is {main_string}!")

    for sub in sub_string:
        if not isinstance(sub, str):
            raise ValueError("Sub string must be a string. But is {sub}!")
        if sub in main_string:
            return True
    return False


def set_first_element(arr, item):
    """
    Sets the specified item as the first element of the array.

    Parameters:
    arr (list or numpy.ndarray): The input array.
    item: The item to be set as the first element of the array.

    Returns:
    list or numpy.ndarray: The modified array with the specified item as the first element.
    """

    if isinstance(arr, list):
        if item in arr:
            arr.remove(item)
            arr.insert(0, item)
        return arr
    elif isinstance(arr, np.ndarray):
        if item in arr:
            arr = np.delete(arr, np.where(arr == item))
            arr = np.insert(arr, 0, item)
        return arr
    else:
        raise ValueError(
            "Unsupported array type. Please provide a list or numpy.ndarray.")


def replicate_object(obj, n):
    """
    Replicates the given object 'n' times and returns an array.

    Parameters:
    obj: any
        The object to be replicated.
    n: int
        The number of times the object should be replicated.

    Returns:
    list
        An array containing 'n' copies of the object.
    """
    return [obj] * n


def has_non_none_objects(arr):
    """
    Checks whether an array contains any objects other than None.

    Parameters:
    arr: list
        The array to be checked.

    Returns:
    bool
        True if the array contains objects other than None, False otherwise.
    """
    return any(item is not None for item in arr)


def move_item_to_first(arr, item):
    """
    Moves the specified item to the first position of the array if it exists.

    Args:
    arr (list or numpy.ndarray): The input array.
    item: The item to be moved to the first position.

    Returns:
    list or numpy.ndarray: The modified array with the specified item at the first position,
    or the original array if the item is not present.

    Example:
    >>> move_item_to_first([1, 2, 3, 4], 3)
    [3, 1, 2, 4]
    >>> move_item_to_first(np.array(['apple', 'banana', 'orange']), 'banana')
    array(['banana', 'apple', 'orange'], dtype='<U6')
    >>> move_item_to_first([1, 2, 3, 4], 5)
    [1, 2, 3, 4]
    """
    if item in arr:
        if isinstance(arr, list):
            arr.remove(item)
            arr.insert(0, item)
        elif isinstance(arr, np.ndarray):
            arr = np.concatenate(([item], np.setdiff1d(arr, [item])))
    return arr


def round2int(x, round=None):
    """
    Rounds a number to the nearest integer.

    Parameters:
        x (float): The number to be rounded.
        round (str, optional): The rounding method to be used. Defaults to None.

    Returns:
        int: The rounded integer value of x.
    """
    if round is None:
        x = int(np.round(x))
    elif round == 'up':
        x = int(np.ceil(x))
    elif round == 'down':
        x = int(np.floor(x))
    else:
        raise ValueError(f"Unsupported rounding method: {round}")

    return x


