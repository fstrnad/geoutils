"""
File that contains files for the analysis of different time series.
Often using event synchronization
"""

import geoutils.utils.statistic_utils as sut
import multiprocessing as mpi
from joblib import Parallel, delayed
import scipy.stats as st
from itertools import product
import pandas as pd
import geoutils.tsa.event_synchronization as es
import numpy as np
import xarray as xr
from importlib import reload
import copy
import geoutils.utils.time_utils as tu
import geoutils.utils.general_utils as gut
from tqdm import tqdm
reload(tu)
reload(gut)
reload(sut)


def get_yearly_ts(data_t, times,
                  sm='Jan',
                  em='Dec',
                  name='data'):
    """Gets a yearly seperated time series where the columns are the years and the
    indices the time points in the yearly data

    Args:
        data_t (np.array): time series of dataset
        times (np.datetime): array of dates to
        sm (str, optional): Start Months. Defaults to 'Jan'.
        em (str, optional): end months. Defaults to 'Dec'.

    Returns:
        pd.DataFrame: DataFrame that contains the years seperated as columns
    """
    xr_ts = xr.DataArray(data=data_t,
                         name='data',
                         coords={'time': np.array(times)},
                         dims=['time'])

    sy, ey = tu.get_sy_ey_time(xr_ts.time, sm=sm, em=em)

    for idx_y, year in enumerate(range(sy, ey+1, 1)):
        start_date, end_date = tu.get_start_end_date(sm, em, year, year)
        data = xr_ts.sel(time=slice(start_date,
                                    end_date)
                         )

        if idx_y == 0:
            ts_yearly = pd.DataFrame(data.data,
                                     columns=[year],
                                     )
        elif idx_y > 0:
            ts_yearly_new = pd.DataFrame(data.data,
                                         columns=[year],
                                         )
            ts_yearly = pd.concat([ts_yearly, ts_yearly_new], axis=1)

    return ts_yearly


# ############################## Event Series ###########################
# def remove_consecutive_evs(es):
#     this_series_idx = np.where(es > 0)[0]
#     this_series_idx_1nb = this_series_idx + 1
#     # this_series_idx_2nb = event_data_idx[i] + 2

#     intersect_1nb = np.intersect1d(this_series_idx, this_series_idx_1nb)

#     es[intersect_1nb] = 0

#     return es


# def remove_consec_evs_array(event_data):
#     """
#     Consecutive days with rainfall above the threshold are considered as single events
#     and placed on the first day of occurrence.

#     Parameters
#     ----------
#     event_data : Array
#         Array containing event_data
#     Returns
#     -------
#     event_data : Array
#         Corrected array of event data.
#     """
#     for es in event_data:
#         es = remove_consecutive_evs(es=es)

#     return event_data


def remove_consec_idx(idx_lst):
    idx_lst = np.array(idx_lst)
    idx_lst_nn = idx_lst + 1
    return idx_lst[~np.in1d(idx_lst, idx_lst_nn)]


def get_evs_index(evs, th=0, rcevs=True):
    """Gets the index series of an event series. Can optionally remove consecutive events.

    Args:
        evs (list): list of events (0 and 1 )
        th (float, optional): threshold for 1 event. Defaults to 0.
        rcevs (bool, optional): Remove consecutive events. Defaults to True.

    Returns:
        list: list of indices
    """
    idx_lst = np.where(evs > th)[0]
    if rcevs:
        idx_lst = remove_consec_idx(idx_lst=idx_lst)
    return idx_lst


def get_evs_index_array(event_data, th=0, max_num_evs=None, rcevs=True, verbose=False):
    """Get an array of indices for time event time series.

    Args:
        event_data (list): list of event series
        th (int, optional): threshold for num events. Defaults to 0.
        rcevs (bool, optional): recompute event Series. Defaults to True.

    Returns:
        np.array: array of event series.
    """
    extreme_event_index_matrix = []
    gut.myprint('Compute Index Array', verbose=verbose)
    for i, e in enumerate(tqdm(event_data, disable=~verbose)):
        idx_lst = get_evs_index(evs=e, th=th, rcevs=rcevs)
        num_evs = len(idx_lst)
        # if num_evs < 1:
        #     raise ValueError(f'Index {i} does not contain any events!')
        if max_num_evs is not None:
            if num_evs >= max_num_evs:
                raise ValueError(
                    f'Index {i} contains too many events ({num_evs}) than allowed ({max_num_evs})!')
        extreme_event_index_matrix.append(idx_lst)
    return np.array(extreme_event_index_matrix, dtype=object)


def get_tps_evs(evs):
    """Returns for an event series xr.Dataarray a list with for each location
    the time points when an event occured.

    Args:
        evs (xr.Dataarray): dataarray with the variable 'evs'.

    Returns:
        list: list xr.Dataarray times.
    """
    times = evs.time
    extreme_event_index_matrix = get_evs_index_array(
        event_data=evs.data.T, rcevs=True)
    tps_arr = []
    for idx_ts in extreme_event_index_matrix:
        tps_arr.append(times[idx_ts])

    return tps_arr


def count_all_events_series(ds, evs_ind_arr, plot=True, savepath=None, label_arr=None):
    """
    Get all events in array of cluster indices.
    """
    tps_arr = []

    for ts_evs in evs_ind_arr:
        tps_arr_ts = []
        for evs in ts_evs:
            ts = np.where(evs == 1)[0]
            tps_arr_ts.append(ts)

        tps = np.concatenate(np.array(tps_arr_ts, dtype=object), axis=0)
        # print(tps)
        tps_arr.append(np.array(tps, dtype=int))

    return tps_arr





def get_ee_ts(evs, rcevs=False, norm=False):
    """Compute a time series that counts per time step the number of Extremes for
    a event series time set of multiple points.

    Args:
        evs (xr.Dataarray): dataarray of time, points
        rcevs (bool, optional): recompute event times series. Defaults to False.
        norm (bool, optional): Normalize time series by number of points. Defaults to False.

    Returns:
        xr.dataarray: array of 1-d time series.
    """
    reload(tu)
    reload(gut)
    evs_ts = tu.get_ts_arr_ds(da=evs)
    ts_idx_arr = get_evs_index_array(event_data=evs_ts, rcevs=rcevs)
    times = evs.time
    dims = gut.get_dimensions(evs)
    num_ee_ts = np.zeros(dims['time'])
    for ts in ts_idx_arr:
        num_ee_ts[ts] += 1

    # Normalize by number of data points
    if norm:
        num_ee_ts /= dims['points']
    num_ee_xr = xr.DataArray(data=num_ee_ts,
                             name='num_ee',
                             coords={'time': np.array(times)},
                             dims=['time'])
    return num_ee_xr


def get_most_sync_days_evs(ts, q=0.95):
    times = ts.time
    num_tp = len(times)
    if q > 0:
        sync_days_ts = np.zeros(num_tp)
        _, indices = gut.get_quantile_of_ts(ts, q=q,
                                            max_quantile=True,
                                            return_indices=True)
        # indices = gut.get_locmax_composite_tps(ts=ts, q=q)['peak_idx']
        sync_days_ts[indices] = 1
    elif q == 0:
        sync_days_ts = np.ones(num_tp)
    else:
        raise ValueError(f'Quantile {q} not defined')
    sync_days_ts = tu.create_xr_ts(data=sync_days_ts, times=times)

    return sync_days_ts


def complement_evs_series(dataarray):
    """
    Create the complement time series of an xarray DataArray containing 0 and 1 values.

    Parameters:
        dataarray (xarray.DataArray): An xarray DataArray containing 0 and 1 values.

    Returns:
        xarray.DataArray: The complement time series with every 0 replaced by 1 and every 1 replaced by 0.

    Raises:
        ValueError: If the input time series contains values other than 0 and 1.
    """
    # Check if the input time series contains only 0 and 1 values
    unique_values = set(dataarray.values.flatten())
    if not set([0, 1]).issuperset(unique_values):
        raise ValueError(
            "The input time series should contain only 0 and 1 values.")

    complement_dataarray = 1 - dataarray
    return complement_dataarray


def get_evs_idx_ds(ds, ids):
    pids = ds.get_points_for_idx(ids)
    evs_s = ds.ds['evs'].sel(points=pids)
    idx_array = get_evs_index_array(evs_s.T)

    return idx_array


def get_ratio_ee_ds(ds, p_tps, q=0.95, start_month='Jan', end_month='Dec'):
    """Ratio of EEs defined by
    R_p = f_p / f_0

    Args:
        ds (xr.DataArray): dataset containing the EEs.
        p_tps (list): list of tps that are examined.
    """
    var_names = gut.get_varnames_ds(ds=ds.ds)
    if start_month != 'Jan' or end_month != 'Dec':
        ds_tmp = tu.get_month_range_data(
            dataset=ds.ds, start_month=start_month, end_month=end_month)
    else:
        ds_tmp = ds.ds

    if 'evs' not in var_names:
        print('Need first to compute Evs!')
        ds_tmp['evs'], _ = tu.compute_evs(dataarray=ds_tmp[ds.var_name], q=q,)
    ee_cnts = tu.get_ee_count_ds(ds=ds_tmp)
    num_days = tu.get_num_tps(ds_tmp)
    f_0 = ee_cnts / num_days

    p_tps_ds = tu.get_sel_tps_ds(ds=ds_tmp, tps=p_tps)
    num_p_tps = len(p_tps)
    ee_cnts_p_tps = tu.get_ee_count_ds(ds=p_tps_ds)
    f_p = ee_cnts_p_tps / num_p_tps

    return f_p / f_0


def get_comb_ij(cnx, ed):
    pids = cnx.ds.get_points_for_idx(ed)
    sid, tid = pids
    evs_s = cnx.ds.ds['evs'].sel(points=sid)
    evs_t = cnx.ds.ds['evs'].sel(points=tid)
    if np.count_nonzero(evs_s) < 1:
        raise ValueError(f"Error zeros in series {ed}!")
    if np.count_nonzero(evs_t) < 1:
        raise ValueError(f"Error zeros in series {ed}!")
    return get_evs_index_array([evs_s, evs_t])


def get_el_evs_idx(cnx, el, parallel=True):
    """Creates an array of event index series for each pair of sids-tids
    in a climnet.

    Args:
        cnx (climNetworX): climate Networkx containing the network
        el (np.ndarray): 2d array of sids-tids.
        parallel (bool): apply parallel computing, faster but might be buggy
    """
    if parallel:
        print(f'Get Event Series of edge list {len(el)}!')
        all_sids = el[:, 0]
        all_tids = el[:, 1]
        s_pids = cnx.ds.get_points_for_idx(all_sids)
        t_pids = cnx.ds.get_points_for_idx(all_tids)
        evs_s = cnx.ds.ds['evs'].sel(points=s_pids).T
        evs_t = cnx.ds.ds['evs'].sel(points=t_pids).T

        print(f'Get Evs Index lists of source list {len(el)}!')
        idx_s = get_evs_index_array(evs_s.data)
        print(f'Get Evs Index lists of target list {len(el)}!')
        idx_t = get_evs_index_array(evs_t.data)

        all_combs_idx = np.array(list(zip(idx_s, idx_t)), dtype=object)

    else:
        all_combs_idx = []

        for ed in tqdm(el):
            all_combs_idx.append(get_comb_ij(cnx, ed))
    return all_combs_idx


def get_sync_times_ES(cnx,
                      idx_1,
                      idx_2=None,
                      taumax=10,
                      use_adj=True,
                      within=False,
                      ):
    """Function that computes the time series of synchronous events for
    a given Network (via adjacency) and certain nodes (nodes can be the same!)


    """
    reload(es)

    if idx_2 is None:
        idx_2 = idx_1

    ts_1 = get_evs_idx_ds(ds=cnx.ds, ids=idx_1)
    ts_2 = get_evs_idx_ds(ds=cnx.ds, ids=idx_2)
    if use_adj:
        # compare only events along existing links
        print('WARNING! Use only time series that are significantly correlated.')
        # Get all edges within idx_1 and idx_2 and between idx_1 and idx_2
        if within:
            idx_1 = idx_2 = np.unique(np.concatenate([idx_1, idx_2]))
        el_12 = cnx.get_edges_between_nodes(ids1=idx_1, ids2=idx_2)
        comb_e12 = get_el_evs_idx(cnx, el_12)
    else:
        comb_e12 = np.array(list(product(ts_1, ts_2)), dtype=object)
    print("prepared time series!")

    xr_ts = compute_sync_times(cnx=cnx, comb_e12=comb_e12, taumax=taumax)

    return xr_ts


def get_sync_times_el(cnx, el, taumax=10):

    comb_e12 = get_el_evs_idx(cnx, el)

    return compute_sync_times(cnx=cnx, comb_e12=comb_e12, taumax=taumax)


def compute_sync_times(cnx, comb_e12, taumax):
    time = cnx.ds.ds.time.data
    num_tp = len(time)

    # Take compare all event series to all other event series
    t, t12, t21, dyn_delay_12, dyn_delay_21 = es.es_reg(
        comb_e12=comb_e12,
        taumax=taumax,
        num_tp=num_tp)

    ts_pd = pd.DataFrame(index=pd.DatetimeIndex(time),
                         columns=['t', 't12', 't21'])
    if len(t12) != len(time) or len(t) != len(time):
        raise ValueError(f"Times are not of same length {len(t12)}!")

    ts_pd.loc[time, 't'] = t
    ts_pd.loc[time, 't12'] = t12
    ts_pd.loc[time, 't21'] = t21

    xr_ts = xr.Dataset.from_dataframe(ts_pd)
    xr_ts = xr_ts.rename({'index': 'time'})
    return xr_ts


def get_sync_times_ES_yearly(ds,
                             net,
                             region_1_dict,
                             region_2_dict=None,
                             sy=None,
                             ey=None,
                             sm=None,
                             em=None,
                             taumax=10,
                             use_adj=True,
                             ):

    reload(es)
    if region_2_dict is None:
        region_2_dict = region_1_dict
    times = ds.ds['time']
    start_year, end_year = tu.get_sy_ey_time(times, sy=sy, ey=ey, sm=sm, em=em)

    for idx_y, year in enumerate(np.arange(start_year, end_year)):
        print(f"Year {year}")
        ts_idx_1 = es.prepare_es_input_data(region_1_dict[year]['ts'])
        ts_idx_2 = es.prepare_es_input_data(region_2_dict[year]['ts'])
        num_tp = len(region_1_dict[year]['ts'][0])
        times = region_1_dict[year]['times']
        if num_tp != len(times):
            raise ValueError(f'lenght of time series {num_tp} not equal!')

        ind_ts_dict1 = dict(zip(region_1_dict['ids'], ts_idx_1))
        ind_ts_dict2 = dict(zip(region_2_dict['ids'], ts_idx_2))
        if use_adj is True:
            t, t12, t21, dyn_delay = es.es_reg_network(
                ind_ts_dict1, ind_ts_dict2, taumax,
                adjacency=net.adjacency,
                num_tp=num_tp)
        else:
            t, t12, t21, dyn_delay = es.es_reg(
                ts_idx_1, ts_idx_2, taumax,
                num_tp=num_tp)
        if len(times) != len(t12):
            raise ValueError(
                f"Time not of same length {len(times)} vs {len(t12)}!")
        pd_array_new = pd.DataFrame(np.vstack([t12, t21]).transpose(),
                                    columns=['t12', 't21'],
                                    index=times.data)
        if idx_y == 0:
            pd_array = copy.deepcopy(pd_array_new)
            ts12_yearly = pd.DataFrame(t12,
                                       columns=[year],
                                       )
            ts21_yearly = pd.DataFrame(t21,
                                       columns=[year],
                                       )
        elif idx_y > 0:
            pd_array = pd_array.append(pd_array_new)
            ts12_yearly_new = pd.DataFrame(t12,
                                           columns=[year],
                                           )
            ts12_yearly = pd.concat([ts12_yearly, ts12_yearly_new], axis=1)

            ts21_yearly_new = pd.DataFrame(t21,
                                           columns=[year],
                                           )
            ts21_yearly = pd.concat([ts21_yearly, ts21_yearly_new], axis=1)

    return pd_array, ts12_yearly, ts21_yearly


def get_quantile_yearly_ts(ts_yearly, q_arr=0):
    """Yearly pandas array of time series.

    Args:
        ts_yearly (pd.DataFrame):  Rows are time steps of year, columns
    are specific year.
        q_arr ([type]): [description]
        label_arr ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    pd_res = ts_yearly.quantile()

    return pd_res


def rank_corr_ts(ts1, ts2, av_type='mean'):
    # transpose data to get array of time series
    ts_rk1 = ts1.data.T
    ts_rk2 = ts2.data.T
    comb_t12 = np.array(list(product(ts_rk1, ts_rk2)), dtype=object)
    t12_arr = []
    for t1, t2 in comb_t12:
        t12 = t1*t2  # point product of both ts
        t12_arr.append(t12)

    t12_res = np.mean(t12_arr, axis=0)

    time = ts1.time.data
    ts_pd = pd.DataFrame(index=pd.DatetimeIndex(time),
                         columns=['t12'])

    ts_pd.loc[time, 't12'] = t12_res

    xr_ts = xr.Dataset.from_dataframe(ts_pd)
    xr_ts = xr_ts.rename({'index': 'time'})

    return xr_ts


def get_tps4val(ts, val):
    idx_tps = np.where(ts == val)[0]

    return ts[idx_tps].time


def get_expt_ees(evs, tps, timemean='year'):

    sy, ey, num_years = tu.get_time_range_years(evs)

    # Get selected number of EEs
    sel_tps_data = tu.get_sel_tps_ds(ds=evs, tps=tps)
    # First fill time series with 0 to ensure continous time series results
    sel_tps_data = tu.fill_time_series_val(sel_tps_data)
    # Get the EE time series from the evs dataset
    t_ee_sel = get_ee_ts(evs=sel_tps_data)
    t_tm_sel = tu.apply_timesum(t_ee_sel, timemean=timemean)

    # Get total number of EEs
    t_ee_tot = get_ee_ts(evs=evs)
    # t_tm_tot = tu.apply_timesum(t_ee_tot, timemean=timemean)
    # average number of EREs per year
    t_tm_tot = np.count_nonzero(evs) / num_years
    # Fraction of EREs
    frac_ees = t_tm_sel/t_tm_tot
    # Normalize by number of days over whole time period
    norm = len(tps.time) / len(evs.time)
    # print(len(t_ee_sel), len(t_ee_tot), norm)

    return frac_ees / norm


def get_cond_occ(tps, cond, counter,
                 small_sample_corection=True,
                 min_num_samples=20):

    if isinstance(cond, list):
        if len(cond) < 1:
            gut.myprint('WARNING: Condition empty, return 0')
            return np.zeros(len(counter))
    # Joint count, eg. sync, active, phase
    phase_sync_act = tu.get_sel_tps_ds(ds=cond, tps=tps)
    # Get Counts of conditional intersection, eg. Sync + Active/Break
    if len(phase_sync_act) > 0:
        count_phase_act_sync = sut.count_occ(occ_arr=[phase_sync_act.data],
                                             count_arr=counter,
                                             rel_freq=False)
    else:
        gut.myprint(f'WARNNING: No events in intersection, return 0')
        return np.zeros(len(counter))

    # Determine time points counts for denominator
    # Tps active/break of phase = joint probabilites (P(p,a))
    count_phase_act = sut.count_occ(occ_arr=[cond.data],
                                    count_arr=counter,
                                    rel_freq=False)

    # print(count_phase_act_sync, count_phase_act)

    # print(count_phase_act)
    if small_sample_corection:
        count_phase_act = np.where(count_phase_act <= min_num_samples,
                                   count_phase_act*2.22,
                                   count_phase_act)
    # correct for 0 counts
    count_phase_act = np.where(count_phase_act == 0, np.inf, count_phase_act)
    # print(count_phase_act)
    p_s_1_act = count_phase_act_sync/count_phase_act

    return p_s_1_act
