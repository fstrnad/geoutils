#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:09:03 2020

@author: Felix Strnad
"""
import geoutils.utils.file_utils as fut
import geoutils.utils.general_utils as gut
from importlib import reload
import os
import numpy as np
import multiprocessing as mpi
import time
from joblib import Parallel, delayed
from tqdm import tqdm
import scipy.stats as st
from itertools import product
import geoutils.tsa.time_series_analysis as tsa
import geoutils.utils.time_utils as tu

reload(tu)


def instant_synchronization(lag, times, es_array1, es_array2=None):
    if es_array2 is None:
        gut.myprint('No second event series given. Using first event series for both!')
        es_array2 = es_array1
    if es_array1.shape[1] != es_array2.shape[1]:
        raise ValueError('Event series must have same length in time!')

    es_array, es_array_lag = tu.get_lagged_ts_arr(es_array1, es_array2, lag=lag)

    sync_ts = np.zeros(len(times))

    for i, e1 in enumerate(es_array):
        for j, e2 in enumerate(es_array_lag):
            if i != j:
                sync_e1e2 = e1*e2
                sync_evs = np.where(sync_e1e2 == 1)[0]
                sync_ts[sync_evs] += 1

    sync_ts = tu.create_xr_ts(data=sync_ts,
                              times=times,)
    return sync_ts