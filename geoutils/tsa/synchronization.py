#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:09:03 2020

@author: Felix Strnad
"""
import geoutils.utils.general_utils as gut
from importlib import reload
import numpy as np
import multiprocessing as mpi
import time
from joblib import Parallel, delayed
from tqdm import tqdm
import geoutils.tsa.time_series_analysis as tsa
import geoutils.utils.time_utils as tu

reload(tu)


def lagged_synchronization(lag, times, es_array1, es_array2=None):
    print(lag)
    if es_array2 is None:
        gut.myprint(
            'No second event series given. Using first event series for both!')
        es_array2 = es_array1
    arr_eq = np.array_equal(es_array1, es_array2)
    if es_array1.shape[1] != es_array2.shape[1]:
        raise ValueError('Event series must have same length in time!')
    if es_array1.shape[1] != len(times):
        raise ValueError(
            'Event series and times must have same length in time!')
    sync_ts = np.zeros(len(times))

    if isinstance(lag, int):
        lag = [lag]
    for this_lag in lag:
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

    sync_ts = tu.create_xr_ts(data=sync_ts,
                              times=times,)
    return sync_ts


def lagged_synchronization_exclude_ts(lag, es_array1,
                                      es_array2=None,
                                      ts_exclude=None,
                                      lag_exclude=None):
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

    ts_exclude_compliment = tsa.complement_evs_series(ts_exclude).values

    for this_lag in lag:
        print(this_lag)
        if lag_exclude is not None and lag_exclude < this_lag and lag_exclude > 0:
            gut.myprint(f'Excluding events until lag {lag_exclude} steps')
            # Excludes all events until lag_exclude
            for this_lag_excl in np.arange(0, lag_exclude+1, 1):
                this_ts_exclude_compl, this_ts_exclude_compl_lag = tu.get_lagged_ts(
                    ts_exclude_compliment,
                    ts_exclude_compliment,
                    lag=this_lag_excl)
                this_ts_exclude_compl = this_ts_exclude_compl * this_ts_exclude_compl_lag
            this_ts_exclude_compl = this_ts_exclude_compl[:-np.abs(
                this_lag-lag_exclude)]
        else:
            # This is only to bring it to the same length as the other event series
            if this_lag > 0:
                this_ts_exclude_compl = ts_exclude_compliment[:-np.abs(
                    this_lag)]
            elif this_lag < 0:
                this_ts_exclude_compl = ts_exclude_compliment[np.abs(
                    this_lag):]
            else:
                this_ts_exclude_compl = ts_exclude_compliment
        es_array, es_array_lag = tu.get_lagged_ts_arr(
            es_array1, es_array2, lag=this_lag)
        for i, e1 in enumerate(es_array):
            for j, e2 in enumerate(es_array_lag):
                if arr_eq and i == j:
                    continue
                else:
                    # only if there is no 0 in ts_exclude the sync is counted
                    sync_e1e2 = e1*e2*this_ts_exclude_compl
                    sync_evs = np.where(sync_e1e2 == 1)[0]
                    sync_ts[sync_evs] += 1

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
