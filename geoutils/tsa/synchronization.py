#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:09:03 2020

@author: Felix Strnad
"""
import random
import geoutils.utils.general_utils as gut
from importlib import reload
import numpy as np
import multiprocessing as mpi
import time
from joblib import Parallel, delayed
from tqdm import tqdm
import geoutils.tsa.time_series_analysis as tsa
import geoutils.utils.time_utils as tu
import dask

reload(tu)


def generate_random_events(x, T):
    """
    Generates an array of 0s and 1s with 'x' 1s randomly distributed within an array of length 'T'.

    Args:
        x (int): The number of 1s to generate.
        T (int): The length of the output array.

    Returns:
        list: A list containing 0s and 1s, with 'x' 1s randomly distributed within the array.

    Raises:
        ValueError: If 'x' is greater than 'T'.

    Example:
        >>> generate_random_events(3, 10)
        [0, 1, 0, 1, 0, 0, 0, 0, 1, 0]
    """
    if x > T:
        raise ValueError(
            "Number of events (x) cannot exceed the array length (T).")

    result = [0] * T
    indices = random.sample(range(T), x)

    for idx in indices:
        result[idx] = 1

    return result


def generate_random_event_series(input_array):
    """
    Generates an output array of random time series with the same number of 1s as in the input time series.

    Args:
        input_array (list): A list of time series where each time series is represented as a list of 0s and 1s.

    Returns:
        list: A list of random time series, each having the same number of 1s as the corresponding input time series.

    Example:
        >>> input_array = [[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]]
        >>> generate_random_time_series(input_array)
        [[0, 0, 0, 1, 0], [1, 1, 0, 0, 1]]
    """
    output_array = np.zeros_like(input_array)
    input_array = gut.convert_to_integers(input_array)
    for idx, time_series in enumerate(input_array):
        num_events = int(sum(time_series))
        T = len(time_series)

        if num_events == 0:
            continue
        else:
            output_array[idx] = generate_random_events(num_events, T)
    return np.array(output_array)


def null_model_one_surrogate(lag, times, es_array1, es_array2):
    rnd_es_array1 = generate_random_event_series(es_array1)
    if es_array2 is None:
        rnd_es_array2 = rnd_es_array1
    else:
        rnd_es_array2 = generate_random_event_series(es_array2)
    rnd_sync_ts = lagged_synchronization(lag, times,
                                         rnd_es_array1,
                                         rnd_es_array2,
                                         return_xr=False,
                                         verbose=False)
    return rnd_sync_ts


def null_model_lagged_synchronization(lag, times, es_array1, es_array2=None,
                                      surrogates=1000, return_xr=True,
                                      q=None, num_cpus=None, return_surr=False,):
    # For parallel Programming
    num_cpus_avail = mpi.cpu_count() if num_cpus is None else num_cpus
    backend = 'multiprocessing'

    gut.myprint(f"Use {num_cpus_avail} CPUs in parallel!")
    coll_ts = (Parallel(n_jobs=num_cpus_avail, backend=backend)
               (delayed(null_model_one_surrogate)
                (lag, times, es_array1, es_array2)
                for sample in tqdm(range(surrogates))
                )
               )
    coll_ts = np.array(coll_ts)
    if return_surr:
        return coll_ts
    else:
        if q is not None:
            rnd_sync_ts = np.quantile(coll_ts, q=q, axis=0)
        else:
            rnd_sync_ts = np.mean(coll_ts, axis=0)
        if return_xr:
            rnd_sync_ts = tu.create_xr_ts(data=rnd_sync_ts,
                                          times=times,)
            return rnd_sync_ts
        else:
            return rnd_sync_ts


def lagged_synchronization(lag, times, es_array1, es_array2=None,
                           return_xr=True,
                           exclude_lag0=True,
                           exclude_lags=0, verbose=True):
    if es_array2 is None:
        gut.myprint(
            'No second event series given. Using first event series for both!',
            verbose=verbose)
        es_array2 = es_array1
    arr_eq = np.array_equal(es_array1, es_array2)
    if es_array1.shape[1] != es_array2.shape[1]:
        raise ValueError('Event series must have same length in time!')
    if es_array1.shape[1] != len(times):
        raise ValueError(
            'Event series and times must have same length in time!')
    sync_ts = np.zeros(len(times))

    # Excludes syncs with lag 0
    if exclude_lag0 and lag != 0:
        e2lag0_compliment = gut.invert_array(es_array2)
        gut.myprint(f'Excluding lag 0 to {exclude_lags} array2 sync. events',
                    verbose=verbose)
        # Excludes all events until exclude_lags
        if exclude_lags != 0:
            sign_excl_lags = np.sign(exclude_lags)
            for this_lag_excl in np.arange(1, abs(exclude_lags)+1, 1):
                e2lag0_compliment, this_ts_exclude_compl_lag = tu.get_lagged_ts_arr(
                    e2lag0_compliment,
                    e2lag0_compliment,
                    lag=sign_excl_lags*1)
                e2lag0_compliment *= this_ts_exclude_compl_lag
    else:
        e2lag0_compliment = np.ones_like(es_array2)

    if isinstance(lag, int):
        lag = [lag]
    for this_lag in lag:
        es_array, es_array_lag = tu.get_lagged_ts_arr(
            es_array1, es_array2, lag=this_lag)
        e2lag0_compliment_lag, _ = tu.get_lagged_ts_arr(
            e2lag0_compliment,
            e2lag0_compliment,
            lag=this_lag - abs(exclude_lags))
        for i, e1 in enumerate(es_array):
            for j, e2 in enumerate(es_array_lag):
                # Only if same array not to count same events at lag 0
                if arr_eq and i == j and lag == 0:
                    continue
                else:
                    # print(j, np.count_nonzero(e2), np.count_nonzero(1-e2lag0_compliment_lag[j][12:]))
                    sync_e1e2 = e1*e2*e2lag0_compliment_lag[j]
                    sync_evs = np.where(sync_e1e2 == 1)[0]
                    sync_ts[sync_evs] += 1
    if return_xr:
        sync_ts = tu.create_xr_ts(data=sync_ts,
                                  times=times,)
        return sync_ts
    else:
        return sync_ts


def lagged_synchronization_exclude_ts(lag, es_array1,
                                      es_array2=None,
                                      ts_exclude=None,
                                      exclude_lags=None):
    if es_array2 is None:
        gut.myprint(
            'No second event series given. Using first event series for both!')
        es_array2 = es_array1
    arr_eq = np.array_equal(es_array1, es_array2)
    if es_array1.shape[1] != es_array2.shape[1]:
        raise ValueError('Event series must have same length in time!')
    if es_array1.shape[1] != len(ts_exclude):
        raise ValueError(
            'Event series and ts_exclude must have same length in time!')

    times = ts_exclude.time.values
    sync_ts = np.zeros(len(ts_exclude))

    if isinstance(lag, int):
        lag = [lag]

    if ts_exclude is None:
        raise ValueError('No ts_exclude given. Please provide one!')

    ts_excl_compl = tsa.complement_evs_series(ts_exclude).values

    for this_lag in lag:
        if exclude_lags is not None and exclude_lags < this_lag and exclude_lags > 0:
            gut.myprint(f'Excluding events until lag {exclude_lags} steps')
            sign_excl_lags = np.sign(exclude_lags)
            # Excludes all events until exclude_lags
            for this_lag_excl in np.arange(1, abs(exclude_lags)+1, 1):
                ts_excl_compl, ts_excl_compl_lag = tu.get_lagged_ts(
                    ts_excl_compl,
                    ts_excl_compl,
                    lag=sign_excl_lags*1)
                ts_excl_compl *= ts_excl_compl_lag
        else:
            exclude_lags = 0

        # This is only to bring it to the same length as the other event series
        this_ts_excl_compl, _ = tu.get_lagged_ts(
            ts_excl_compl,
            ts_excl_compl,
            lag=this_lag - abs(exclude_lags))
        es_array, es_array_lag = tu.get_lagged_ts_arr(
            es_array1, es_array2, lag=this_lag)

        for i, e1 in enumerate(es_array):
            for j, e2 in enumerate(es_array_lag):
                if arr_eq and i == j and lag == 0:
                    continue
                else:
                    # only if there is no 0 in ts_exclude the sync is counted
                    sync_e1e2 = e1*e2*this_ts_excl_compl
                    sync_evs = np.where(sync_e1e2 == 1)[0]
                    sync_ts[sync_evs] += 1
    # print(np.count_nonzero(sync_ts))

    sync_ts = tu.create_xr_ts(data=sync_ts,
                              times=times,)
    return sync_ts


def count_lags(es_array1, es_array2=None, num_lags=10, lag_step=1, lag_start=0):
    if es_array2 is None:
        gut.myprint(
            'No second event series given. Using first event series for both!')
        es_array2 = es_array1
    if es_array1.shape[1] != es_array2.shape[1]:
        raise ValueError('Event series must have same length in time!')

    lag_arr = np.arange(lag_start, lag_start + num_lags+1, lag_step)
    arr_eq = np.array_equal(es_array1, es_array2)

    tau_dict = {}
    for this_lag in tqdm(lag_arr):
        sync_ts = np.zeros(es_array1.shape[1])

        es_array, es_array_lag = tu.get_lagged_ts_arr(
            es_array1, es_array2, lag=this_lag)

        for i, e1 in enumerate(es_array):
            for j, e2 in enumerate(es_array_lag):
                if arr_eq and i == j:
                    continue
                else:
                    sync_e1e2 = e1*e2
                    sync_evs = np.where(sync_e1e2 == 1)[0]
                    sync_ts[sync_evs] += 1
        tau_dict[this_lag] = np.sum(sync_ts)

    return tau_dict
