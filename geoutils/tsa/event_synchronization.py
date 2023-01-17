#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:09:03 2020

@author: Felix Strnad
"""
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


def event_synchronization(event_data, taumax=10, min_num_sync_events=2):
    num_time_series = len(event_data)
    adj_matrix = np.zeros((num_time_series, num_time_series), dtype=int)
    double_taumax = 2*taumax

    extreme_event_index_matrix = tsa.get_evs_index_array(
        event_data=event_data, th=0, rcevs=True)

    start = time.time()
    print("Start computing event synchronization!")

    for i, ind_list_e1 in enumerate(extreme_event_index_matrix):
        # Get indices of event series 1
        # ind_list_e1= np.where(e1>0)[0]
        for j, ind_list_e2 in enumerate(extreme_event_index_matrix):
            if i == j:
                continue

            sync_event = 0

            for m, e1_ind in enumerate(ind_list_e1[1:-1], start=1):
                d_11_past = e1_ind-ind_list_e1[m-1]
                d_11_next = ind_list_e1[m+1]-e1_ind

                for n, e2_ind in enumerate(ind_list_e2[1:-1], start=1):
                    d_12_now = (e1_ind-e2_ind)
                    if d_12_now > taumax:
                        continue

                    d_22_past = e2_ind-ind_list_e2[n-1]
                    d_22_next = ind_list_e2[n+1]-e2_ind

                    tau = min(d_11_past, d_11_next, d_22_past,
                              d_22_next, double_taumax) / 2
                    # print(tau, d_11_past, d_11_next, d_22_past, d_22_next, double_taumax)
                    if d_12_now <= tau and d_12_now >= 0:
                        sync_event += 1
                        # print("Sync: ", d_12_now, e1_ind, e2_ind, sync_event,n)

                    if d_12_now < -taumax:
                        # print('break!',  d_12_now, e1_ind, e2_ind, )
                        break

            # Createria if number of synchron events is relevant
            if sync_event >= min_num_sync_events:
                # print(i,j, sync_event)
                adj_matrix[i, j] = 1
    end = time.time()
    print(end - start)
    np.save('adj_matrix_gpcp.npy', adj_matrix)
    print(adj_matrix)

    return adj_matrix


def event_synchronization_one_series(extreme_event_index_matrix,
                                     ind_list_e1,
                                     i,
                                     taumax=10,
                                     min_num_sync_events=2):
    double_taumax = 2*taumax
    sync_time_series_indicies = []
    # Get indices of event series 1
    # ind_list_e1= np.where(e1>0)[0]

    for j, ind_list_e2 in enumerate(extreme_event_index_matrix):
        if i == j:
            continue

        sync_events = event_sync(
            ind_list_e1, ind_list_e2, taumax, double_taumax)

        # Createria if number of synchron events is relevant
        if sync_events > min_num_sync_events:
            # print(i,j, sync_event)
            num_events_i = len(ind_list_e1)
            num_events_j = len(ind_list_e2)
            sync_time_series_indicies.append(
                (j, num_events_i, num_events_j, sync_events))

    return (i, sync_time_series_indicies)


def event_sync(ind_list_e1, ind_list_e2, taumax, double_taumax):
    # Get indices of event series 2
    sync_events = 0
    # print(ind_list_e1)
    # print(ind_list_e2)
    for m, e1_ind in enumerate(ind_list_e1[1:-1], start=1):
        d_11_past = e1_ind-ind_list_e1[m-1]
        d_11_next = ind_list_e1[m+1]-e1_ind

        for n, e2_ind in enumerate(ind_list_e2[1:-1], start=1):
            d_12_now = (e1_ind-e2_ind)
            if d_12_now > taumax:
                continue

            d_22_past = e2_ind-ind_list_e2[n-1]
            d_22_next = ind_list_e2[n+1]-e2_ind

            tau = min(d_11_past, d_11_next, d_22_past,
                      d_22_next, double_taumax) / 2
            if d_12_now <= tau and d_12_now >= 0:  # directed
                # if abs(d_12_now) <= tau:  # undirected
                sync_events += 1  # Todo Use weight of an event!

            if d_12_now < -taumax:
                break

    return sync_events


def parallel_event_synchronization(event_data,
                                   taumax=10,
                                   min_num_sync_events=2,
                                   job_id=0,
                                   num_jobs=1,
                                   savepath=None,
                                   q_dict=None,
                                   q_min=0.5,  # minimum value that i-j is considered significant
                                   num_cpus=None
                                   ):
    # Load and check null model:
    q_dict_keys = list(q_dict.keys())
    if q_min not in q_dict_keys:
        raise ValueError(f'This q {q_min} is not in dictionary!')
    null_model = q_dict[q_min]
    max_num_evs = null_model.shape[0]
    num_time_series = len(event_data)
    one_array_length = int(num_time_series/num_jobs) + 1

    extreme_event_index_matrix = tsa.get_evs_index_array(event_data=event_data,
                                                         max_num_evs=max_num_evs,
                                                         rcevs=True)

    start_arr_idx = job_id*one_array_length
    end_arr_idx = (job_id+1)*one_array_length
    # start_arr_idx = 0
    # end_arr_idx = 30
    print(
        f"Start computing event synchronization for event data from {start_arr_idx} to {end_arr_idx}!")

    # For parallel Programming
    num_cpus_avail = mpi.cpu_count() if num_cpus is None else num_cpus
    print(f"Use {num_cpus_avail} CPUs in parallel!")

    parallelArray = []

    start = time.time()

    # Parallelizing by using joblib
    backend = 'multiprocessing'
    # backend='loky'
    # backend='threading'
    parallelArray = (Parallel(n_jobs=num_cpus_avail, backend=backend)
                     (delayed(event_synchronization_one_series)
                      (extreme_event_index_matrix, idx_lst_e1,
                       start_arr_idx + i, taumax, min_num_sync_events)
                      for i, idx_lst_e1 in enumerate(tqdm(extreme_event_index_matrix[start_arr_idx:end_arr_idx]))
                      )
                     )

    # Store output of parallel processes in adjecency matrix
    adj_matrix_edge_list = []

    print(f"Now store results in numpy array to {savepath}!", flush=True)
    for process in tqdm(parallelArray):
        i, list_sync_event_series = process
        for sync_event in list_sync_event_series:
            j, num_events_i, num_events_j, num_sync_events_ij = sync_event
            thresh_null_model = null_model[num_events_i, num_events_j]

            if max_num_evs < num_events_i or max_num_evs < num_events_j:
                raise ValueError(
                    f'Allowed are max. {max_num_evs}! Too many events in i {num_events_i} or j {num_events_j}')

            # Threshold needs to be larger (non >= !)
            if num_sync_events_ij > thresh_null_model:
                adj_matrix_edge_list.append((int(i),
                                             int(j),
                                             int(num_events_i),
                                             int(num_events_j),
                                             int(num_sync_events_ij)
                                             )
                                            )

        # print(i, list_sync_event_series)
    end = time.time()
    print(f'{end-start}', flush=True)
    # sys.exit(0)
    gut.save_np_dict(arr_dict=adj_matrix_edge_list, sp=savepath)
    gut.myprint(f'Finished for job ID {job_id}')

    return adj_matrix_edge_list


def event_sync_reg(ind_list_e1, ind_list_e2, idx, taumax, double_taumax):
    """
    ES for regional analysis that delivers specific timings.
    It returns the
    """
    sync_events = 0
    t12_lst = []
    t21_lst = []
    t_lst = []
    dyn_delay_lst_12 = []
    dyn_delay_lst_21 = []

    for m, e1_ind in enumerate(ind_list_e1[1:-1], start=1):
        d_11_past = e1_ind-ind_list_e1[m-1]
        d_11_next = ind_list_e1[m+1]-e1_ind

        for n, e2_ind in enumerate(ind_list_e2[1:-1], start=1):
            d_12_now = (e1_ind-e2_ind)
            if d_12_now > taumax:
                continue

            d_22_past = e2_ind-ind_list_e2[n-1]
            d_22_next = ind_list_e2[n+1]-e2_ind

            tau = min(d_11_past, d_11_next, d_22_past,
                      d_22_next, double_taumax) / 2
            if abs(d_12_now) <= tau:
                sync_events += 1
                t_lst.append(e2_ind)
                if d_12_now < 0:  # First event in 1 then in 2
                    t12_lst.append(e1_ind)
                    dyn_delay_lst_12.append(d_12_now)
                elif d_12_now > 0:  # First event in 2 then in 1
                    t21_lst.append(e2_ind)
                    dyn_delay_lst_21.append(-1*d_12_now)
                else:
                    t_lst.append(e2_ind)
                    t12_lst.append(e1_ind)
                    t21_lst.append(e2_ind)
                    dyn_delay_lst_12.append(d_12_now)
                    dyn_delay_lst_21.append(d_12_now)

            if d_12_now < -taumax:
                break

    ret_dict = dict(
        t=t_lst,
        t12=t12_lst,
        t21=t21_lst,
        dyn_delay_12=dyn_delay_lst_12,
        dyn_delay_21=dyn_delay_lst_21,
        idx=idx
    )

    return ret_dict


def es_reg(comb_e12, taumax, num_tp):
    """
    """
    backend = 'multiprocessing'
    # backend='loky'
    # backend='threading'
    num_cpus_avail = mpi.cpu_count()
    print(f"Number of available CPUs: {num_cpus_avail}")
    parallelArray = (Parallel(n_jobs=num_cpus_avail, backend=backend)
                     (delayed(event_sync_reg)
                      (e1,  e2, idx, taumax, 2*taumax)
                      for idx, (e1, e2) in enumerate(tqdm(comb_e12))
                      )
                     )

    t12 = np.zeros(num_tp, dtype=int)
    t21 = np.zeros(num_tp, dtype=int)
    t = np.zeros(num_tp)
    dyn_delay_arr_12 = []
    dyn_delay_arr_21 = []
    for ret_dict in tqdm(parallelArray):
        t_e = ret_dict['t'],
        t12_e = ret_dict['t12']
        t21_e = ret_dict['t21']
        dyn_delay_12 = ret_dict['dyn_delay_12']
        dyn_delay_21 = ret_dict['dyn_delay_21']
        t[t_e] += 1
        t12[t12_e] += 1
        t21[t21_e] += 1
        dyn_delay_arr_12.append(dyn_delay_12)
        dyn_delay_arr_21.append(dyn_delay_21)

    return t, t12, t21, dyn_delay_arr_12, dyn_delay_arr_21


def get_network_comb(c_indices1, c_indices2, adjacency=None):
    comb_c12 = np.array(list(product(c_indices1, c_indices2)), dtype=object)

    if adjacency is None:
        return comb_c12
    else:
        comb_c12_in_network = []
        for (c1, c2) in comb_c12:
            if adjacency[c1][c2] == 1 or adjacency[c2][c1] == 1:
                comb_c12_in_network.append([c1, c2])
        if len(comb_c12) == len(comb_c12_in_network):
            print("WARNING! All links in network seem to be connected!")
        return np.array(comb_c12_in_network, dtype=object)


def get_network_comb_es(c_indices1, c_indices2, ind_ts_dict1, ind_ts_dict2, adjacency=None):
    print("Get combinations of Event Series!")

    comb_c12_in_network = get_network_comb(
        c_indices1, c_indices2, adjacency=adjacency)
    comb_e12 = []
    for (c1, c2) in comb_c12_in_network:
        e1 = ind_ts_dict1[c1]
        e2 = ind_ts_dict2[c2]
        comb_e12.append([e1, e2])
    comb_e12 = np.array(comb_e12, dtype=object)
    return comb_e12


def es_reg_network(ind_ts_dict1, ind_ts_dict2, taumax,
                   adjacency=None, num_tp=0):
    """
    ES between 2 regions.
    However, only links are considered that are found to be statistically significant
    """
    if adjacency is None:
        raise ValueError("Please give network structure via Adjacency!")
    if num_tp == 0:
        raise ValueError("ERROR! Please specify correct number of timepoints!")
    c_indices1 = ind_ts_dict1.keys()
    c_indices2 = ind_ts_dict2.keys()
    comb_e12 = get_network_comb_es(
        c_indices1, c_indices2,
        ind_ts_dict1=ind_ts_dict1,
        ind_ts_dict2=ind_ts_dict2,
        adjacency=adjacency)
    backend = 'multiprocessing'
    num_cpus_avail = mpi.cpu_count()
    print(f"Number of available CPUs: {num_cpus_avail}")
    parallelArray = (
        Parallel(n_jobs=num_cpus_avail, backend=backend)
        (delayed(event_sync_reg)
         (e1, e2, idx, taumax, 2*taumax)
         for idx, (e1, e2) in enumerate(tqdm(comb_e12))
         )
    )
    t12 = np.zeros(num_tp, dtype=int)
    t21 = np.zeros(num_tp, dtype=int)
    t = np.zeros(num_tp)
    dyn_delay_arr = []
    for ret_dict in tqdm(parallelArray):
        t = ret_dict['t'],
        t12 = ret_dict['t12']
        t21 = ret_dict['t21']
        dyn_delay = ret_dict['dyn_delay']
        dyn_delay_arr.append(dyn_delay)
    if np.array_equal(t12, t21):
        print('Warning: Arrays are exactly equal!')
    # dyn_delay_arr = np.concatenate(dyn_delay_arr, axis=0)
    return t, t12, t21, dyn_delay_arr


# %%  Adjacency Matrix from Event Synchronization files

def get_adj_from_E(E_matrix_folder, num_time_series,
                   savepath=None,
                   weighted=False,
                   q_dict=None,
                   q_med=0.5,
                   lq=0.25,
                   hq=0.75,
                   q_sig=0.95,
                   ):
    # Load and check null model:
    q_dict_keys = list(q_dict.keys())
    for q in [lq, q_med, hq, q_sig]:
        if q not in q_dict_keys:
            raise ValueError(f'This q {q} is not in dictionary!')
    null_model = q_dict[q_sig]
    PLq = q_dict[lq]
    PMed = q_dict[q_med]
    PHq = q_dict[hq]

    if os.path.exists(E_matrix_folder):
        path = E_matrix_folder
        E_matrix_files = [os.path.join(path, fn)
                          for fn in next(os.walk(path))[2]]
    else:
        raise ValueError(f"E_matrix Folder {E_matrix_folder} does not exist!")
    adj_matrix = np.zeros((num_time_series, num_time_series), dtype=int)
    weight_matrix = np.zeros((num_time_series, num_time_series))

    for filename in tqdm(E_matrix_files):
        print(f"Read Matrix with name {filename}")
        if os.path.isfile(filename):
            this_E_matrix = np.load(filename)
        else:
            raise ValueError(f"WARNING! File does not exist {filename}!")

        for adj_list in this_E_matrix:
            i, j, num_events_i, num_events_j, num_sync_events_ij = adj_list
            i = int(i)
            j = int(j)
            thresh_null_model = null_model[num_events_i, num_events_j]
            lq = PLq[num_events_i, num_events_j]
            med = PMed[num_events_i, num_events_j]
            hq = PHq[num_events_i, num_events_j]

            if thresh_null_model > 1:  # Null Model should have at least 2 events
                if num_sync_events_ij > thresh_null_model and np.abs(hq - lq) > 0.001:
                    if weighted is True:
                        weight = (num_sync_events_ij - med) / (hq - lq)
                    else:
                        weight = 1  # All weights are set to 1
                    # adj_matrix[j, i] = adj_matrix[i, j] = 1
                    # weight_matrix[j, i] = weight_matrix[i, j] = weight
                    # events in i are followed by events in j
                    adj_matrix[j, i] = 1
                    weight_matrix[j, i] = weight

    if savepath is not None:
        np.save(savepath, adj_matrix)
    print(
        f'Finished computing Adjency Matrix for Null model with {num_time_series} time series!')

    return adj_matrix, weight_matrix


def null_model_one_series(i,
                          min_num_events,
                          le,
                          num_permutations,
                          taumax,
                          double_taumax,
                          q=[0.25, 0.5, 0.75, 0.95, 0.98, 0.99, 0.995, 0.999],
                          nnelems=True):
    reload(gut)
    list_thresholds_i = []
    for j in range(min_num_events, i + 1):
        season1 = np.zeros(le, dtype="bool")
        season2 = np.zeros(le, dtype="bool")
        season1[:i] = 1
        season2[:j] = 1
        cor = np.zeros(num_permutations)
        # rng = np.random.default_rng()
        for k in range(num_permutations):

            if nnelems:
                # ind_list_e1 = np.sort(rng.choice(le, size=i, replace=False))
                # ind_list_e2 = np.sort(rng.choice(le, size=j, replace=False))
                ind_list_e1 = gut.get_random_numbers_no_neighboring_elems(
                    0, le, i)
                ind_list_e2 = gut.get_random_numbers_no_neighboring_elems(
                    0, le, j)

                cor[k] = event_sync(ind_list_e1, ind_list_e2,
                                    taumax, double_taumax)
            else:
                dat_rnd = np.random.permutation(season1)
                ind_list_e1 = tsa.get_evs_index(evs=dat_rnd, rcevs=True)
                dat_rnd = np.random.permutation(season2)
                ind_list_e2 = tsa.get_evs_index(evs=dat_rnd, rcevs=True)

        res_q = np.quantile(
            cor,
            q=q
        )

        list_thresholds_i.append(
            [j, res_q])

    return i, list_thresholds_i


def null_model_distribution(length_time_series, taumax=10,
                            min_num_events=1, max_num_events=1000,
                            num_permutations=3000,
                            q=[0.25, 0.5, 0.75, 0.95, 0.98, 0.99, 0.995, 0.999],
                            savepath=None,
                            nnelems=True):
    print("Start creating Null model of Event time series!")
    print(f"Model distribution size: {num_permutations}")
    le = length_time_series
    double_taumax = 2*taumax

    size = max_num_events-min_num_events
    # num_ij_pairs = ceil(size*(size + 1) / 2) #  "Kleiner Gauss"
    print(f"Size of Null_model Matrix: {size}")

    size = max_num_events
    num_q_vals = len(q)  # lq, med, hq, th05, th02, th01, th005, th001
    null_model_arr_q = np.empty((num_q_vals, size, size))
    null_model_arr_q[:] = np.nan

    # For parallel Programming
    num_cpus_avail = mpi.cpu_count()
    # num_cpus_avail=1
    print(f"Number of available CPUs: {num_cpus_avail}")
    backend = 'multiprocessing'
    # backend='loky'
    # backend='threading'

    # Parallelizing by using joblib
    pb_fmt = "{desc:<5.5}{percentage:3.0f}%|{bar:30}{r_bar}"
    pb_desc = "Computing Null Model in time..."
    parallelArray = (Parallel(n_jobs=num_cpus_avail, backend=backend)
                     (delayed(null_model_one_series)
                      (i, min_num_events, le, num_permutations,
                       taumax, double_taumax, q, nnelems)
                      for i in tqdm(range(min_num_events, max_num_events),
                                    bar_format=pb_fmt, desc=pb_desc)
                      )
                     )

    print(f"Now store results in numpy array to {savepath}!")
    for process in tqdm(parallelArray):
        i, list_thresholds_i = process
        for j_thresholds in list_thresholds_i:
            j, res_q = j_thresholds
            if len(res_q) != num_q_vals:
                raise ValueError(
                    f'Number of returned q-values {len(res_q)} != to predefined num q-vals {num_q_vals}!')
            for q_idx, P_arr in enumerate(null_model_arr_q):
                P_arr[i, j] = P_arr[j, i] = res_q[q_idx]

    q_dict = {q[idx_q]: P_arr for idx_q, P_arr in enumerate(null_model_arr_q)}
    if savepath is not None:
        print(f'Save null model to {savepath}')
        # save and load a dictionary to a file using NumPy, pickle would work as well
        np.save(savepath, q_dict, allow_pickle=True)
    return q_dict


def null_model_cdf_one_series(i, min_num_events,
                              li,
                              num_permutations,
                              taumax,
                              double_taumax):
    list_thresholds_i = []
    for j in range(min_num_events, i + 1):
        season1 = np.zeros(li, dtype="bool")
        season2 = np.zeros(li, dtype="bool")
        season1[:i] = 1
        season2[:j] = 1
        dat = np.zeros((2, li), dtype="bool")
        cor = np.zeros(num_permutations)
        for k in range(num_permutations):
            dat[0] = np.random.permutation(season1)
            dat[1] = np.random.permutation(season2)
            ind_list_e1, ind_list_e2 = tsa.get_evs_index_array(event_data=dat,
                                                               rcevs=True)
            cor[k] = event_sync(ind_list_e1, ind_list_e2,
                                taumax, double_taumax)

        norm_cdf = st.norm.cdf(cor)

        list_thresholds_i.append([j, norm_cdf])

    return i, list_thresholds_i


# %% Past processing
def construct_full_E(num_jobs, filename, savepath=None):
    # Load matrix for jobid 0
    print(f"Read data from {filename}")

    if os.path.exists(savepath):
        full_adj_matrix = np.load(savepath)
    else:
        full_adj_matrix = np.load(filename+'0.npy')
        for job_id in tqdm(range(1, num_jobs)):
            print(f"Read Matrix with ID {job_id}")
            this_filename = filename+str(job_id) + '.npy'
            if os.path.isfile(this_filename):
                this_adj_matrix = np.load(this_filename)
            else:
                continue
            full_adj_matrix = np.concatenate(
                (full_adj_matrix, this_adj_matrix), axis=0)
            del this_adj_matrix

        print("Full length E_matrix: ", len(full_adj_matrix))
        if savepath is not None:
            np.save(savepath, full_adj_matrix)
    return full_adj_matrix
