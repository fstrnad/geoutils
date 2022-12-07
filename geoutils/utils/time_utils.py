import math
import scipy as sp
import geoutils.utils.spatial_utils as sput
import geoutils.tsa.filters as flt
import scipy.stats as st
from tqdm import tqdm
import pandas as pd
import xarray as xr
import numpy as np
import copy
import cftime

from importlib import reload
import geoutils.utils.statistic_utils as sut
import geoutils.utils.general_utils as gut
reload(gut)
reload(sut)

months = np.array([
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
])

jjas_months = np.array([
    "Jun",
    "Jul",
    "Aug",
    "Sep",
])


def get_index_of_month(month):
    idx = int(np.argwhere(months == month)[0])
    if idx not in np.arange(0, len(months)):
        raise ValueError(f"This month does not exist:{month}!")
    # Attention, this idx starts with 0, so Jan=0, Feb=1, ... Dec=11

    return idx


# def get_month_number(month):
#     return int(get_index_of_month(month) + 1)

def get_month_number(*month_list):
    idx_lst = []
    for month in month_list:
        idx = get_index_of_month(month) + 1
        idx_lst.append(idx)
    if len(idx_lst) == 1:
        return int(idx_lst[0])
    return idx_lst


def get_ts_arr_ds(da):
    """Get an array of time series for multiple data points.

    Args:
        da (xr.Dataarray): Xr.dataarray of type points, time

    Returns:
        np.array: 2d np.array of time series.
    """
    if da.dims == ('time', 'points'):
        time_series_arr = da.data.T
    elif da.dims == ('points', 'time'):
        time_series_arr = da.data
    else:
        raise ValueError(f'this dimension is unknown: {da.dims}!')
    return time_series_arr


def get_num_tps(ds):
    return len(ds.time)


def get_max_num_tps(ds, q=None):
    time_steps = get_num_tps(ds)
    if q is not None:
        max_num_events = int(np.ceil(time_steps * (1 - q)))

    return max_num_events


def get_sel_tps_ds(ds, tps, remove_tp=False, verbose=False):
    if isinstance(tps, xr.DataArray):
        tps = tps.time.data

    if gut.is_single_tp(tps=tps):
        tps = [tps]

    if len(tps) == 0:
        gut.myprint(f'Empty list of time points')
        return []

    if gut.is_datetime360(tps):
        tps_sel = tps
        ds_sel = ds.sel(time=tps_sel, method='nearest')
    else:
        if remove_tp:
            ds_sel = ds.sel(time=tps, method='nearest')
        else:
            tps_sel = np.intersect1d(ds.time, tps)  # Build always intersection
            if len(tps_sel) != len(tps):
                gut.myprint('WARNING! Not all tps in ds', verbose=verbose)
                if verbose:
                    for x in tps.data:
                        if x not in ds.time.data:
                            gut.myprint(f'WARNING! tp not in ds: {x}')
            if len(tps_sel) == 0:
                print('No tps not in dataset!')
                return []
            ds_sel = ds.sel(time=tps_sel, method='nearest')

    return ds_sel


def remove_consecutive_tps(tps, steps=1):
    """Removes consecutive steps in a set of time points until steps after.

    Args:
        tps (xr.DataArray): time points
        steps (int): number of steps to have no consecutive time points.
    """
    num_init_tps = len(tps)
    rem_tps = copy.deepcopy(tps)
    for step in range(1, steps + 1):
        tps_step = add_time_step_tps(tps=rem_tps, time_step=step)
        common_tps = get_common_tps(tps1=rem_tps, tps2=tps_step)
        if len(common_tps) > 0:
            rem_tps = rem_tps.drop_sel(time=common_tps)

    gut.myprint(f'Removed {num_init_tps - len(rem_tps)} time points!')

    return rem_tps


def get_common_tps(tps1, tps2, offset=0, delay=0, step=1):
    """Get the common tps between two set of tps. Can also be used to get all tps that
    are a certain offset later until some max delay.

    Args:
        tps1 (xr.DataArray): dataarray containing time points.
        tps2 (xr.DataArray): dataarray containing time points.
        offset (int, optional): offset. Defaults to 0.
        delay (int, optional): maximum delay. Defaults to 0.
        step (int, optional): stepsize of time points later. Defaults to 1.

    Raises:
        ValueError:

    Returns:
        xr.DataArray: Dataarray of time points.
    """
    if isinstance(tps1, xr.DataArray):
        tps1 = tps1.time
    if isinstance(tps2, xr.DataArray):
        tps2 = tps2.time
    if delay == 0:
        tps = np.sort(np.intersect1d(tps1, tps2))
    elif delay < 0:
        raise ValueError('Only positive delay allowed!')
    else:
        common_tps = []
        for offset in np.arange(offset, delay+1, step):
            # tps2 that have a delay are shifted by this delay backward
            tps2_tmp = add_time_step_tps(tps2.time, time_step=-1*offset)
            tps = np.sort(np.intersect1d(tps1, tps2_tmp))
            common_tps.append(tps)

        tps = np.sort(np.unique(np.concatenate(common_tps, axis=0)))

    tps = gut.create_xr_ds(data=tps, dims=['time'], coords={'time': tps})

    return tps


def get_tps_month(ds, month, ):

    times = ds.time
    tps_month = get_month_range_data(times, start_month=month, end_month=month)

    if len(tps_month) == 0:
        raise ValueError('Tps not in dataset!')

    return tps_month


def get_sel_time_range(ds, time_range,
                       start_month='Jan',
                       end_month='Dec',
                       freq='D',
                       verbose=True):
    if time_range is not None:
        sd, ed = get_time_range(ds)
        time_range_0 = str2datetime(time_range[0], verbose=False)
        time_range_1 = str2datetime(time_range[-1], verbose=False)

        if time_range_0 < sd:
            gut.myprint(
                f'WARNING: Selected {time_range_0} smaller dataset {sd}',
                verbose=verbose)
            time_range_0 = sd
        if time_range_1 > ed:
            gut.myprint(
                f'WARNING: Selected {time_range_1} larger dataset {ed}',
                verbose=verbose)
            time_range_1 = ed

        if gut.is_datetime360(time_range_0):
            time_range_1 = add_time_window(time_range_1, time_unit=freq)

        tps = get_dates_of_time_range(
            time_range=[time_range_0, time_range_1], freq=freq)
        ds_sel = ds.sel(time=slice(tps[0], tps[-1]))
        # ds_sel = ds.sel(time=tps, method='bfill')  # always that one that is closest to last

        if start_month != 'Jan' or end_month != 'Dec':
            ds_sel = get_month_range_data(dataset=ds_sel,
                                          start_month=start_month,
                                          end_month=end_month)
    else:
        ds_sel = ds
    return ds_sel


def is_in_month_range(month, start_month, end_month):
    start_month_idx = get_month_number(start_month)
    end_month_idx = get_month_number(end_month)

    if start_month_idx <= end_month_idx:
        mask = (month >= start_month_idx) & (month <= end_month_idx)
    else:
        mask = (month >= start_month_idx) | (month <= end_month_idx)
    return mask


def get_month_range_data(dataset, start_month="Jan", end_month="Dec"):
    """
    This function generates data within a given month range.
    It can be from smaller month to higher (eg. Jul-Sep) but as well from higher month
    to smaller month (eg. Dec-Feb)

    Parameters
    ----------
    start_month : string, optional
        Start month. The default is 'Jan'.
    end_month : string, optional
        End Month. The default is 'Dec'.

    Returns
    -------
    seasonal_data : xr.dataarray
        array that contains only data within month-range.

    """
    seasonal_data = dataset.sel(
        time=is_in_month_range(dataset["time.month"], start_month, end_month)
    )

    return seasonal_data


def get_idx_months(times, start_month, end_month):
    month_dates = get_month_range_data(
        dataset=times, start_month=start_month, end_month=end_month
    ).data
    fulltime = times.data
    idx_months = np.where(np.in1d(fulltime, month_dates))[0]
    idx_non_months = np.where(~np.in1d(fulltime, month_dates))[0]

    return idx_months, idx_non_months


def get_month_range_zero(dataarray, start_month, end_month):
    times = dataarray.time
    idx_in_months, idx_out_months = get_idx_months(
        times=times, start_month=start_month, end_month=end_month
    )
    non_dates = times.data[idx_out_months]

    dataarray.loc[dict(time=non_dates)] = 0

    return dataarray


def get_idx_tps_times(tps, times):
    """Get the indices of time points in another time series.

    Args:
        tps (list): list of time points
        times (xr.dataarray): dataarray of time series

    Returns:
        list: list of indices of tps in time series.
    """
    tps_idx = np.where(np.in1d(times, tps))[0]
    return tps_idx


def get_ym_date(date):
    if isinstance(date, xr.DataArray):
        date = date.data
    else:
        date = np.datetime64(date)
    y, m, d = get_ymd_date(date)
    mi = m.astype(int) % 12
    mname = months[mi]
    return f"{mname} {y}"


def get_sy_ey_time(times, sy=None, ey=None, sm=None, em=None):
    """Returns the start and end year of a xr Dataarray
    datetime object

    Args:
        times (xr.Datetime): xr.Datetime object that contains time
        sy (int, optional): other startyear if specified. Defaults to None.
        ey (int, optional): end endyear. Defaults to None.

    Returns:
        int, int: start and end year
    """
    if sy is None:
        start_year = int(times[0].time.dt.year)
    else:
        start_year = sy
    if ey is None:
        end_year = int(times[-1].time.dt.year) + 1
    else:
        end_year = ey

    if sm is not None and em is not None:
        smi = get_month_number(sm)
        emi = get_month_number(em)
        if emi < smi:
            end_year -= 1

    return start_year, end_year


def get_start_end_date(data):
    base_period = np.array(
        [data.time.data.min(), data.time.data.max()])
    return base_period


# def get_start_end_date(times, sm="Jan", em="Dec"):
#     sy, ey = get_sy_ey_time(times=times)
#     smi = get_month_number(sm)
#     emi = get_month_number(em)
#     start_date = np.datetime64(f"{int(sy)}-{int(smi):02}-{int(1):02}", "D")
#     if em == "Feb":
#         end_day = 28
#     elif em in ["Jan", "Mar", "May", "Jul", "Aug", "Oct", "Dec"]:
#         end_day = 31
#     else:
#         end_day = 30

#     ey = copy.deepcopy(ey)
#     if emi < smi:
#         ey = ey + 1
#     end_date = np.datetime64(f"{int(ey)}-{int(emi):02}-{int(end_day):02}", "D")

#     return np.array([start_date, end_date])


def get_start_end_date_shift(time, sm, em, shift=0):
    """Same as normal get_start_date_year but the start and end shifted by shift days

    Args:
        sm (str): Start Month
        em (str): End Month
        sy (int): start Year
        ey (int): end Year
        shift (int, optional): shift by days. Defaults to 0.

    Returns:
        str: start and end date
    """
    smi = get_month_number(sm)
    emi = get_month_number(em)
    sy, ey = get_sy_ey_time(time=time)

    start_date = np.datetime64(f"{int(sy)}-{int(smi):02}-{int(1):02}", "D")
    if em == "Feb":
        end_day = 28
    elif em in ["Jan", "Mar", "May", "Jul", "Aug", "Oct", "Dec"]:
        end_day = 31
    else:
        end_day = 30

    ey = copy.deepcopy(ey)
    if emi < smi:
        ey = ey + 1
    end_date = np.datetime64(f"{int(ey)}-{int(emi):02}-{int(end_day):02}", "D")

    if shift > 0:
        start_date -= np.timedelta64(int(shift), "D")
        end_date += np.timedelta64(int(shift), "D")
    return start_date, end_date


def get_time_range(ds):
    time = ds.time
    if gut.is_datetime360(time=time.data[0]):
        sd = time.data[0]
        ed = time.data[-1]
    else:
        sd = np.datetime64(time.data[0], "D")
        ed = np.datetime64(time.data[-1], "D")
    return sd, ed


def get_dates_of_ds(ds):
    sd, ed = get_time_range(ds=ds)
    tps = get_dates_of_time_range(time_range=[sd, ed])
    return tps


def str2datetime(string, dtype='D', verbose=False):
    if type(string) is str:
        date = np.datetime64(string, dtype)
    else:
        date = string
        gut.myprint(f'WARNING {string} is not string!', verbose=verbose)
    return date


def is_full_year(ds, get_missing_dates=False):
    sd, ed = get_time_range(ds)
    ds_time = ds.time.data
    all_days = np.arange(sd, ed, np.timedelta64(1, "D"))

    if len(ds_time) < len(all_days):
        if get_missing_dates:
            return np.setdiff1d(all_days, ds_time)
        else:
            return False
    else:
        return True


def get_tm_name(timemean):
    if timemean == "day":
        tm = "1D"
    elif timemean == "week":
        tm = "1W"
    elif timemean == "month":
        tm = "1MS"
    elif timemean == "season":  # This is not just 3MS!  found here: https://github.com/pydata/xarray/issues/810
        tm = "Q-FEB"
    elif timemean == "year":
        tm = "1Y"
    elif timemean == 'pentad':
        tm = "5D"
    else:
        raise ValueError(
            f"This time mean {timemean} does not exist! Please choose week, month, season or year!"
        )

    return tm


def get_mean_time_series(da, lon_range, lat_range, time_roll=0):
    """Get mean time series of selected area.

    Parameters:
    -----------
    da: xr.DataArray
        Data
    lon_range: list
        [min, max] of longitudinal range
    lat_range: list
        [min, max] of latiduninal range
    """
    da_area = sput.cut_map(da, lon_range=lon_range,
                           lat_range=lat_range)
    ts_mean = da_area.mean(dim=('lon', 'lat'), skipna=True)
    ts_std = da_area.std(dim=('lon', 'lat'), skipna=True)
    if time_roll > 0:
        ts_mean = ts_mean.rolling(time=time_roll, center=True).mean()
        ts_std = ts_std.rolling(time=time_roll, center=True).mean()

    return ts_mean, ts_std


def compute_timemean(ds, timemean, sm=None, em=None, dropna=True):
    """Computes the monmean average on a given xr.dataset

    Args:
        ds (xr.dataset): xr dataset for the dataset

    Returns:
        xr.dataset: monthly average dataset
    """
    tm = get_tm_name(timemean)

    gut.myprint(f"Compute {timemean}ly means of all variables!")
    if dropna:
        ds = ds.resample(time=tm).mean(
            dim="time", skipna=True).dropna(dim="time")
    else:
        ds = ds.resample(time=tm).mean(dim="time", skipna=True)
    if sm is not None or em is not None:
        ds = get_month_range_data(ds, start_month=sm, end_month=em)
    return ds


def apply_timemax(ds, timemean, sm=None, em=None):
    """Computes the monmax on a given xr.dataset

    Args:
        ds (xr.dataset): xr dataset for the dataset

    Returns:
        xr.dataset: monthly average dataset
    """
    tm = get_tm_name(timemean)

    gut.myprint(f"Compute {timemean}ly means of all variables!")
    ds = ds.resample(time=tm).max(dim="time", skipna=True)
    if sm is not None or em is not None:
        ds = get_month_range_data(ds, start_month=sm, end_month=em)
    return ds


def apply_timesum(ds, timemean, sm=None, em=None, dropna=True):
    """Computes the monmean average on a given xr.dataset

    Args:
        ds (xr.dataset): xr dataset for the dataset

    Returns:
        xr.dataset: monthly average dataset
    """
    tm = get_tm_name(timemean)

    gut.myprint(f"Compute {timemean}ly means of all variables!")
    if dropna:
        ds = ds.resample(time=tm).sum(
            dim="time", skipna=True).dropna(dim="time")
    else:
        ds = ds.resample(time=tm).sum(dim="time", skipna=True)
    if sm is not None or em is not None:
        ds = get_month_range_data(ds, start_month=sm, end_month=em)
    return ds


def averge_out_nans_ts(ts, av_range=1):

    num_nans = np.count_nonzero(np.isnan(ts) == True)
    if num_nans == 0:
        return ts
    else:
        len_ts = len(ts)
        if num_nans / len_ts > 0.1:
            gut.myprint("Warning! More than 10 % nans in ts")
        idx_nans = np.where(np.isnan(ts) == True)[0]

        for idx in idx_nans:
            if idx == 0:
                ts[0] = np.nanmean(ts[:10])
            else:
                ts[idx] = np.mean(ts[idx - av_range: idx])

        num_nans = np.count_nonzero(np.isnan(ts) == True)
        if num_nans > 0:
            idx_nans = np.where(np.isnan(ts) == True)[0]
            raise ValueError(f"Still nans in ts {idx_nans}")

        return ts


def compute_anomalies(dataarray, climatology_array=None,
                      group="dayofyear", base_period=None,
                      verbose=True):
    """Calculate anomalies.

    Parameters:
    -----
    dataarray: xr.DataArray
        Dataarray to compute anomalies from.
    climatology_array: xr.Dataarray array from which the climatology is computed.
        Set to dataarray if not provided explicitly.
    group: str
        time group the anomalies are calculated over, i.e. 'month', 'day', 'dayofyear'
    base_period (list, None): period to calculate climatology over. Default None.

    Return:
    -------
    anomalies: xr.dataarray
    """

    if climatology_array is None:
        climatology_array = dataarray

    if base_period is None:
        base_period = np.array(
            [dataarray.time.data.min(), dataarray.time.data.max()])

    if group in ["dayofyear", "month", "season"]:
        climatology = (
            climatology_array.sel(time=slice(base_period[0], base_period[1]))
            .groupby(f"time.{group}")
            .mean(dim="time")
        )
        anomalies = dataarray.groupby(f"time.{group}") - climatology
    else:
        month_ids = []
        monthly_groups = climatology_array.groupby("time.month")

        if group == "JJAS":
            for month in ["Jun", "Jul", "Aug", "Sep"]:
                month_ids.append(get_month_number(month))
        elif group == 'DJFM':  # here it is important to include the December of the previous year
            for month in ["Dec", "Jan", "Feb", "Mar"]:
                month_ids.append(get_month_number(month))
        climatology = (
            monthly_groups.mean(dim="time").sel(
                month=month_ids).mean(dim="month")
        )
        anomalies = dataarray - climatology
    if verbose:
        gut.myprint(f"Created {group}ly anomalies!")

    var_name = anomalies.name
    anomalies = gut.rename_da(da=anomalies, name=f'{var_name}_an_{group}')

    return anomalies


def get_ee_ds(dataarray, q=0.95, th=1, th_eev=15):
    # Remove days without rain
    data_above_th = dataarray.where(dataarray > th)
    # Gives the quanile value for each cell
    q_val_map = data_above_th.quantile(q, dim="time")
    # Set values below quantile to 0
    data_above_quantile = xr.where(
        data_above_th > q_val_map[:], data_above_th, np.nan)
    # Set values to 0 that have not at least the value th_eev
    data_above_quantile = xr.where(
        data_above_quantile > th_eev, data_above_quantile, np.nan
    )
    ee_map = data_above_quantile.count(dim="time")

    rel_frac_q_map = data_above_quantile.sum(
        dim="time") / dataarray.sum(dim="time")

    return q_val_map, ee_map, data_above_quantile, rel_frac_q_map


def get_ee_count_ds(ds, q=0.95):
    varnames = gut.get_varnames_ds(ds)
    if 'evs' in varnames:
        evs = ds['evs']
        evs_cnt = evs.sum(dim='time')
    else:
        raise ValueError('No Evs in dataset found!')
    return evs_cnt


def compute_evs(dataarray, q=0.9, th=1, th_eev=5, min_evs=3):
    """Creates an event series from an input time series.

    Args:
        dataarray (xr.dataarray): The time series of a variable of
        q (float, optional): Quantile for defining an extreme event. Defaults to 0.95.
        th (int, optional): Threshold. Removes all values in time series.
            Eg. important to get wet days. Defaults to 1.
        th_eev (int, optional): Minimum value of an extreme event. Defaults to 15.
        min_evs (int, optional): Minimum number of extreme event within 1 time series. Defaults to 20.

    Raises:
        ValueError: [description]
        ValueError: [description]

    Returns:
        event_series (xr.Dataarray): Event time series of 0 and 1 (1=event).
        mask (xr.dataarray): Gives out which values are masked out.
    """
    if q > 1 or q < 0:
        raise ValueError(f"ERROR! q = {q} has to be in range [0, 1]!")
    if th <= 0:
        raise ValueError(
            f"ERROR! Threshold for values th = {th} has to be > 0!")

    # Compute percentile data, remove all values below percentile, but with a minimum of threshold q
    gut.myprint(
        f"Start remove values below q={q} and at least with q_value >= {th_eev} ...")
    _, ee_map, data_above_quantile, _ = get_ee_ds(
        dataarray=dataarray, q=q, th=th, th_eev=th_eev
    )
    # Create mask for which cells are left out
    gut.myprint(f"Remove cells without min number of events: {min_evs}")
    mask = ee_map > min_evs
    final_data = data_above_quantile.where(mask, np.nan)

    gut.myprint("Now create binary event series!")
    event_series = xr.where(final_data[:] > 0, 1, 0)
    gut.myprint("Done!")
    event_series = event_series.rename("evs")

    # Create new mask for dataset: Masked values are areas with no events!
    mask = xr.where(ee_map > min_evs, 1, 0)

    return event_series, mask


def detrend_dim(da, dim="time", deg=1, startyear=None, freq='D'):
    if startyear is None:
        p = da.polyfit(dim=dim, deg=deg)
        fit = xr.polyval(da[dim], p.polyfit_coefficients)
        start_val = fit[0]
        detrended_da = da - fit + start_val
    else:
        start_date, end_date = get_start_end_date(data=da)
        if gut.is_datetime360(time=da.time.data[0]):
            date_before_detrend = cftime.Datetime360Day(startyear-1, 12, 30)
            date_start_detrend = cftime.Datetime360Day(startyear, 1, 1)
        else:
            date_before_detrend = np.datetime64(f'{startyear-1}-12-31')
            date_start_detrend = np.datetime64(f'{startyear}-01-01')
        gut.myprint(f'Start detrending from {date_start_detrend}...')
        da_no_detrend = get_sel_time_range(ds=da,
                                           time_range=[start_date,
                                                       date_before_detrend],
                                           freq=freq,
                                           verbose=False)
        da_detrend = get_sel_time_range(ds=da,
                                        time_range=[
                                            date_start_detrend, end_date],
                                        freq=freq,
                                        verbose=False)
        p = da_detrend.polyfit(dim=dim, deg=deg)
        fit = xr.polyval(da_detrend[dim], p.polyfit_coefficients)
        start_val = fit[0]
        detrended_da = da_detrend - fit + start_val
        detrended_da = xr.concat([da_no_detrend, detrended_da], dim='time')

    return detrended_da


def correlation_per_timeperiod(x, y, time_period):
    """Correlation per time period.

    Args:
        x ([type]): [description]
        y ([type]): [description]
        time_period ([type]): [description]

    Returns:
        [type]: [description]
    """
    corr = []
    for tp in time_period:
        corr.append(
            np.corrcoef(
                x.sel(time=slice(tp[0], tp[1])), y.sel(
                    time=slice(tp[0], tp[1]))
            )[0, 1]
        )

    return xr.DataArray(corr, dims=["time"], coords={"time": time_period[:, 0]})


def tp2str(tp, m=True, d=True):
    """Returns the string for np.datetime(64) object.

    Args:
        tp (np.datetime): time point
        m (bool, optional): Return month as well. Defaults to False.
        d (bool, optional): Return day as well. Defaults to False.

    Returns:
        str: string of the date
    """
    if isinstance(tp, xr.DataArray):
        tp = tp.time.data

    if gut.is_datetime360(tp):
        date = f'{tp.year}'
    else:
        ts = pd.to_datetime(str(tp))
        date = ts.strftime('%Y')
        if m:
            date = ts.strftime('%Y-%m')
        if d:
            date = ts.strftime('%Y-%m-%d')
    return date


def get_ymd_date(date):
    if gut.is_datetime360(date):
        d = int(date.day)
        m = int(date.month)
        y = int(date.year)
    else:
        if isinstance(date, xr.DataArray):
            date = np.datetime64(date.time.data)
        else:
            date = np.datetime64(date)

        d = date.astype("M8[D]")
        m = date.astype("M8[M]")
        y = date.astype("M8[Y]")

    return y, m, d


def add_time_window(date, time_step=1, time_unit="D"):

    y, m, d = get_ymd_date(date)
    ad = time_step if time_unit == "D" else 0
    am = time_step if time_unit == "M" else 0
    ay = time_step if time_unit == "Y" else 0

    if gut.is_datetime360(date):
        if d + ad > 30:
            nd = (d + ad) % 30
            am = 1
        else:
            nd = d + ad
        if m + am > 12:
            nm = (m + am) % 12
            ay = 1
        else:
            nm = m + am
        ny = y + ay
        next_date = cftime.Datetime360Day(ny, nm, nd)
    else:
        if isinstance(date, xr.DataArray):
            date = np.datetime64(date.time.data)
        if time_unit == "D":
            next_date = (d + ad) + (date - d)
        elif time_unit == "M":
            next_date = (m + am) + (date - m)
        elif time_unit == "Y":
            next_date = (y + ay) + (date - y)
        next_date = np.datetime64(next_date, "D")

    return next_date


def get_day_progression_arr(ds, tps, start,
                            sps=None, eps=None,
                            var=None, end=None,
                            q=None, step=1,
                            average_ts=False,
                            verbose=False):
    """Gets a day progression for a xr.Dataset for specific time points.

    Args:
        ds (xr.DataSet): dataset of variables
        tps (xr.dataarray): dataarray that contains time points
        start (int): how many time points before to start
        end (int, optional): how many time points to end. Defaults to None.
        step (int, optional): step of progression. Defaults to 1.
        average_ts(bool, optional): Takes all days between two steps into account and averages over .

    Returns:
        dict: dictionary that contains xr.Dataarrays of the means
    """
    composite_arrs = dict()
    s_step = 1
    e_step = -1
    if end is None:
        end = start
    if sps is None:
        sps = tps
        s_step = 0
    if eps is None:
        eps = tps
        e_step = 0

    days = np.arange(-start, end+step, step)
    composite_arrs = []
    for thisstep in days:
        if thisstep < 0:
            this_tps = add_time_step_tps(sps.time, time_step=thisstep + s_step)
        elif thisstep > 0:
            this_tps = add_time_step_tps(eps.time, time_step=thisstep + e_step)
        else:
            this_tps = tps.time  # day 0

        if average_ts and thisstep != 0:
            # the sign is because average is for the preceeding periode
            # signum of thisstep
            av_step = step * -1 * math.copysign(1, thisstep)
            this_tps = get_periods_tps(tps=this_tps, step=av_step)

        this_comp_ts = get_sel_tps_ds(ds=ds, tps=this_tps)
        if var == 'evs':
            this_comp_ts = xr.where(
                this_comp_ts[var] == 1, this_comp_ts[var], np.nan
            )
            mean_ts = this_comp_ts.sum(dim='time')
            # print(mean_ts)
        elif var is not None:
            if q is None:
                mean_ts = this_comp_ts[var].mean(dim='time')
            else:
                mean_ts = this_comp_ts[var].quantile(q=q,
                                                     dim='time')
                # this_comp_ts = xr.where(
                #     this_comp_ts < 1, this_comp_ts, np.nan
                # )
                # mean_ts = this_comp_ts[var].count(dim='time')
        else:
            mean_ts = this_comp_ts.mean(dim='time')

        mean_ts = mean_ts.expand_dims(
            {'day': 1}).assign_coords({'day': [thisstep]})

        composite_arrs.append(mean_ts)

    gut.myprint(
        'Merge selected composite days into 1 xr.DataSet...', verbose=verbose)
    composite_arrs = xr.merge(composite_arrs)

    return composite_arrs


def get_day_arr(ds, tps):

    composite_arrs = []
    for day, tp in enumerate(tps):
        tp_ds = get_sel_tps_ds(ds, tps=[tp]).mean(dim='time')
        tp_ds = tp_ds.expand_dims(
            {'day': 1}).assign_coords({'day': [day]})
        composite_arrs.append(tp_ds)

    gut.myprint('Merge selected composite days into 1 xr.DataSet...')
    composite_arrs = xr.merge(composite_arrs)

    return composite_arrs


def get_box_propagation(ds, loc_dict, tps,
                        sps=None, eps=None,
                        num_days=1, regions=None,
                        normalize=True,
                        var='evs', step=1, q=0.9,
                        norm_grid_fac=2):  # four borders
    reload(sput)
    coll_data = dict()
    if regions is None:
        regions = list(loc_dict.keys())
    for region in tqdm(regions):
        # EE TS
        pids = loc_dict[region]['pids']
        pr_data = ds.sel(points=pids)
        # pr_data = loc_dict[region]['data']
        composite_arrs = get_day_progression_arr(ds=pr_data,
                                                 tps=tps,
                                                 sps=sps, eps=eps,
                                                 start=num_days,
                                                 end=num_days,
                                                 step=step,
                                                 var=var,
                                                 )
        coll_data[region] = composite_arrs[var]

    days = composite_arrs.day.data
    box_data = np.zeros((len(regions), len(days)))
    for i, region in enumerate(regions):
        this_data = coll_data[region]
        pids = this_data.points
        if normalize:
            evs_data = loc_dict[region]['data']['evs']
            tot_num_days = len(tps)
            num_cells = len(evs_data.points)
            # num_cells = 10  # Get Results per 100 cells
            # norm = num_cells * tot_num_days
            norm = tot_num_days*num_cells / \
                (norm_grid_fac*100)  # Get Results per 100 cells
            print(region, norm)
        else:
            norm = 1
        for j, day in enumerate(days):
            if var != 'evs':
                box_data[i][j] = float(this_data.sel(day=day).quantile(
                    dim='points',
                    q=q).data)
            else:
                this_box_data = float(this_data.sel(
                    day=day).sum(dim='points').data) / norm
                box_data[i][j] = this_box_data
    box_data = gut.mk_grid_array(data=box_data,
                                 x_coords=days,
                                 y_coords=regions,)

    return box_data


def get_quantile_progression_arr(ds, tps, start,
                                 sps=None, eps=None,
                                 var=None, end=None,
                                 q=None, step=1,
                                 average_ts=False,
                                 verbose=False,
                                 q_th=0.05,
                                 th=None):
    progression_arr = get_day_progression_arr(
        ds=ds, tps=tps,
        start=start,
        sps=sps, eps=eps,
        var=var, end=end,
        q=q, step=step,
        average_ts=average_ts,
        verbose=verbose)
    day_arr = xr.zeros_like(progression_arr.sel(day=0))
    day_arr = xr.where(day_arr == 1, 0, np.nan)
    for idx, (day) in enumerate(progression_arr.day):
        day = int(day)
        mean_ts = progression_arr.sel(day=day)
        th_mask = xr.ones_like(mean_ts)
        if th is not None:
            th_mask = xr.where(mean_ts <= th, 1, 0) if q_th < 0.5 else xr.where(
                mean_ts >= th, 1, 0)
        q_val = mean_ts.quantile(q=q_th)
        q_mask = xr.where(mean_ts <= q_val, 1, 0)
        mask = q_mask * th_mask

        # This overwrites old values
        day_arr = xr.where(mask, day, day_arr)

    return day_arr


def get_hovmoeller(ds, tps, sps=None, eps=None, num_days=0,
                   start=1,
                   var=None, step=1,
                   lat_range=None, lon_range=None,
                   zonal=True,
                   dateline=False):
    reload(sput)
    if num_days > 0:
        composite_arrs = get_day_progression_arr(ds=ds,
                                                 tps=tps,
                                                 sps=sps, eps=eps,
                                                 start=start,
                                                 end=num_days,
                                                 step=step,
                                                 var=var,
                                                 )
    else:
        composite_arrs = get_day_arr(ds=ds, tps=tps)
    composite_arrs = sput.cut_map(ds=composite_arrs,
                                  lon_range=lon_range,
                                  lat_range=lat_range,
                                  dateline=dateline)
    if zonal:
        hov_means = sput.compute_zonal_mean(ds=composite_arrs)
    else:
        hov_means = sput.compute_meridional_mean(ds=composite_arrs)

    return hov_means


def get_hovmoeller_single_tps(ds, tps, num_days,
                              start=1,
                              var=None, step=1,
                              lat_range=None, lon_range=None,
                              zonal=True,
                              gf=(0, 0),
                              dateline=False):
    hov_data = []
    if gf[0] != 0 or gf[1] != 0:
        gut.myprint(f'Apply Gaussian Filter with sigma = {gf}!')
        sigma = [gf[1], gf[0]]  # sigma_y, sigma_x

    for tp in tps:
        this_hov_data = get_hovmoeller(ds=ds, tps=tp,
                                       num_days=num_days,
                                       start=start,
                                       var=var, step=step,
                                       lat_range=lat_range,
                                       lon_range=lon_range,
                                       zonal=zonal, dateline=dateline)
        if gf[0] != 0 or gf[1] != 0:
            tmp_data = sp.ndimage.filters.gaussian_filter(
                this_hov_data[var].data, sigma, mode='constant')
            this_hov_data = xr.DataArray(data=tmp_data,
                                         dims=this_hov_data.dims,
                                         coords=this_hov_data.coords)
        hov_data.append(this_hov_data)
    single_hov_dates = xr.concat(hov_data, tps)

    return single_hov_dates


def add_time_step_tps(tps, time_step=1, time_unit="D", ):
    ntps = []
    if isinstance(tps, xr.DataArray):
        tps = tps.time
    if len(np.array([tps.time.data]).shape) == 1:
        tps = [tps]
    for tp in tps:
        ntp = add_time_window(
            date=tp, time_step=time_step, time_unit=time_unit)
        ntps.append(ntp)
    ntps = np.array(ntps)

    return xr.DataArray(ntps, dims=["time"], coords={"time": ntps})


def get_tw_periods(
    sd, ed, tw_length=1, tw_unit="Y", sliding_length=1, sliding_unit="M"
):

    ep = sd
    all_time_periods = []
    all_tps = []
    while ep < ed:
        ep = add_time_window(sd, time_step=tw_length, time_unit=tw_unit)
        if ep < ed:
            tw_range = get_dates_of_time_range([sd, ep])
            all_time_periods.append(tw_range)
            all_tps.append(ep)
            sd = add_time_window(
                sd, time_step=sliding_length, time_unit=sliding_unit)

    return {"range": all_time_periods, "tps": np.array(all_tps)}


def get_periods_tps(tps, start=0, step=1, time_unit="D", include_start=True):
    if step == 0:
        return tps
    else:
        if not include_start and start == 0:
            sign = math.copysign(step)
            tps = add_time_step_tps(tps=tps, step=sign*1)
            step += sign  # because we have shifted the step
        if start > 0:
            stps = add_time_step_tps(
                tps=tps, time_step=start, time_unit=time_unit)
        else:
            stps = tps
        etps = add_time_step_tps(tps=tps, time_step=step, time_unit=time_unit)
        all_time_periods = []
        for idx, stp in enumerate(stps):
            etp = etps[idx]
            tw_range = get_dates_of_time_range([stp, etp], freq=time_unit)
            all_time_periods.append(tw_range)

        all_time_periods = np.concatenate(all_time_periods, axis=0)
        # Removes duplicates of time points
        all_time_periods = np.unique(all_time_periods)
        return all_time_periods


def get_dates_of_time_range(time_range, freq='D'):
    dtype = f"datetime64[{freq}]"
    if type(time_range[0]) is str:
        gut.myprint('Convert String', verbose=False)
        time_range = np.array(time_range, dtype=dtype)
    if gut.is_datetime360(time=time_range[0]):
        date_arr = xr.cftime_range(start=time_range[0],
                                   end=time_range[-1],
                                   normalize=True,   # Normalize to Midnight
                                   freq=freq)
    else:
        sp, ep = time_range[0], time_range[-1]
        if isinstance(sp, xr.DataArray):
            sp = sp.time.data
            ep = ep.time.data

        sp, ep = np.sort([sp, ep])  # Order in time

        date_arr = get_dates_in_range(start_date=sp,
                                      end_date=ep,
                                      time_unit=freq)
        # Include as well last time point
        date_arr = np.concatenate(
            [date_arr, [date_arr[-1] + np.timedelta64(1, freq)]], axis=0
        )
    return date_arr


def get_dates_of_time_ranges(time_ranges, freq='D'):
    dtype = f"datetime64[{freq}]"
    arr = np.array([], dtype=dtype)
    for time_range in time_ranges:
        arr = np.concatenate(
            [arr, get_dates_of_time_range(time_range, freq=freq)], axis=0
        )
    # Also sort the array
    arr = np.sort(arr)
    return arr


def get_dates_in_range(start_date, end_date, time_unit='D'):
    tps = np.arange(start_date, end_date, dtype=f'datetime64[{time_unit}]')
    return tps


def sliding_time_window(
    da,
    corr_method="spearman",
    tw_length=1,
    tw_unit="Y",
    sliding_length=1,
    sliding_unit="M",
    source_ids=None,
    target_ids=None,
):
    """Computes a sliding time window approach for a given dataset.

    Args:
        da (xr.dataarray): dataarray that contains the time series of
        points (list, optional): list of spatial points to applay the method.
                                 Defaults to None.
    """
    reload(sut)
    corr_function = sut.get_corr_function(corr_method=corr_method)
    sids = source_ids
    tids = target_ids
    tids = gut.remove_tids_sids(sids=sids, tids=tids)
    comb_ids = np.concatenate([sids, tids])
    num_nodes = len(comb_ids)

    # Get all different time periods
    sd, ed = get_time_range(ds=da)

    tw_periods = get_tw_periods(
        sd,
        ed,
        tw_length=tw_length,
        tw_unit=tw_unit,
        sliding_length=sliding_length,
        sliding_unit=sliding_unit,
    )

    corr_time_dict = dict(sids=sids, tids=tids, tps=tw_periods["tps"])

    # Apply sliding window over all time periods
    for idx, tw_period in enumerate(tqdm(tw_periods["range"])):
        # Select data only for specified source - target points
        this_tp_data = da.sel(time=tw_period, points=comb_ids)
        corr, pvalue = corr_function(data=this_tp_data)
        if corr.shape != (num_nodes, num_nodes):
            raise ValueError(
                f"Wrong dimension of corr matrix {corr.shape} != {(num_nodes, num_nodes)}!"
            )

        # Define source - correlations, target correlations and source-target correlations
        # in correlation matrix
        st_dict = gut.get_source_target_corr(corr=corr, sids=sids)

        corr_time_dict[idx] = dict(
            corr=corr,
            source_corr=st_dict["source"],
            target_corr=st_dict["target"],
            st_corr=st_dict["source_target"],
        )

    return corr_time_dict


def mean_slw(corr_time_dict, corr_key="st_corr"):
    """Computes the mean correlation time of a dictionary of different times
    to get the time evolution of a correlation.

    Args:
        corr_time_dict (dict): dict that contains the time points and cross correlations
        corr_key (str, optional): Which correlation to use: source correlations, targetcorr
                                  or st_corr. Defaults to 'st_corr'.

    Returns:
        [type]: [description]
    """
    tps = corr_time_dict["tps"]
    mean_arr = []
    std_arr = []
    ts_pd = pd.DataFrame(index=pd.DatetimeIndex(tps), columns=["mean", "std"])

    for idx, tp in enumerate(tps):
        st_corr = corr_time_dict[idx][corr_key]
        mean_arr.append(np.mean(st_corr))
        std_arr.append(np.std(st_corr))

    ts_pd.loc[tps, "mean"] = mean_arr
    ts_pd.loc[tps, "std"] = std_arr
    xr_ts = xr.Dataset.from_dataframe(ts_pd)

    return xr_ts.rename({"index": "time"})


def local_cross_degree_slw(corr_time_dict, corr_key="st_corr", th=0.1):

    tps = corr_time_dict["tps"]
    mean_arr = []
    ts_pd = pd.DataFrame(index=pd.DatetimeIndex(tps), columns=["lcd"])

    for idx, tp in enumerate(tps):
        st_corr = corr_time_dict[idx][corr_key]
        adj_st = np.where(np.abs(st_corr) > th, 1, 0)
        mean_arr.append(np.sum(adj_st))

    ts_pd.loc[tps, "lcd"] = mean_arr
    xr_ts = xr.Dataset.from_dataframe(ts_pd)

    return xr_ts.rename({"index": "time"})


def get_tp_corr(corr_time_dict, tps, corr_method="st_corr"):
    dict_tps = corr_time_dict["tps"]

    corr_array = []
    for tp in tps:
        if tp not in dict_tps:
            raise ValueError(f"This tp does not exist {tp}")

        idx = np.where(tp == dict_tps)[0][0]
        corr = corr_time_dict[idx][corr_method]
        corr_array.append(corr)

    return np.array(corr_array)


def corr_distribution_2_region(corr_arr):
    all_corrs = np.array(corr_arr).flatten()

    return all_corrs


def get_corr_full_ts(
    ds,
    time_periods,
    source_ids=None,
    target_ids=None,
    var_name="anomalies",
    corr_type="spearman",
):
    reload(gut)
    da = ds.ds[var_name]
    corr_function = sut.get_corr_function(corr_method=corr_type)
    sids = source_ids
    tids = target_ids
    tids = gut.remove_tids_sids(sids, tids)
    comb_ids = np.concatenate([sids, tids])
    num_nodes = len(comb_ids)
    tps = get_dates_of_time_ranges(time_ranges=time_periods)
    corr_time_dict = dict(sids=sids, tids=tids, tps=time_periods)

    this_tp_data = da.sel(time=tps, points=comb_ids)
    corr, pvalue = corr_function(data=this_tp_data)
    if corr.shape != (num_nodes, num_nodes):
        raise ValueError(
            f"Wrong dimension of corr matrix {corr.shape} != {(num_nodes, num_nodes)}!"
        )

    # Define source - correlations, target correlations and source-target correlations
    # in correlation matrix
    st_dict = gut.get_source_target_corr(corr=corr, sids=sids)

    corr_time_dict.update(
        dict(
            corr=corr,
            source_corr=st_dict["source"],
            target_corr=st_dict["target"],
            st_corr=st_dict["source_target"],
        )
    )

    return corr_time_dict


def get_rank_ts(ts, tps=None, q=None):
    if tps is not None:
        ts_source_tps = get_sel_tps_ds(ds=ts, tps=tps)
    else:
        ts_source_tps = ts

    # rank along the time dimension
    if q is None:
        ts_source_rk = st.rankdata(ts_source_tps.data, axis=0)
        xr_ts = xr.DataArray(
            data=ts_source_rk, dims=ts_source_tps.dims, coords=ts_source_tps.coords,
        )
    else:
        q_val_map = ts_source_tps.quantile(q, dim="time")
        # Set values below quantile to 0
        xr_ts = xr.where(ts_source_tps > q_val_map[:], 1, 0)

    return xr_ts


def arr_lagged_ts(ts_arr, lag):
    times = ts_arr[0].time
    ntimes = len(times)
    num_tps = ntimes-lag
    df = pd.DataFrame(index=times[0:num_tps])
    # df = pd.DataFrame(index=times)
    for idx, ts in enumerate(ts_arr):
        data = ts.data
        t_data = np.vstack(data)
        for tlag in range(0, lag+1, 1):
            # t_data = np.roll(t_data, tlag)  # wrong but somehow predictive...
            # df[f'{idx}_{tlag}'] = t_data
            if lag == tlag:
                df[f'{idx}_{tlag}'] = t_data[tlag:]
            else:
                df[f'{idx}_{tlag}'] = t_data[tlag:-(lag-tlag)]

    return df


def lead_lag_corr(ts1, ts2,
                  maxlags=20,
                  corr_method='spearman',
                  cutoff=1,
                  cutoff_ts=1):
    reload(gut)
    reload(flt)
    Nx = len(ts1)
    if Nx != len(ts2):
        raise ValueError('ts1 and ts2 must be equal length')
    nts1 = sut.standardize(ts1)
    nts2 = sut.standardize(ts2)

    if cutoff_ts != 1:
        nts1 = flt.apply_butter_filter(ts=nts1, cutoff=cutoff_ts)
        nts2 = flt.apply_butter_filter(ts=nts2, cutoff=cutoff_ts)

    corr_range = []
    p_val_arr = []

    if corr_method == 'spearman':
        corr_func = st.stats.spearmanr
    elif corr_method == 'pearson':
        corr_func = st.stats.pearsonr
    tau_arr = np.arange(-maxlags, maxlags+1, 1)
    for lag in tau_arr:
        if lag > 0:
            corr, p_val = corr_func(nts1[:-np.abs(lag)],
                                    nts2[np.abs(lag):])
        elif lag < 0:
            corr, p_val = corr_func(nts1[np.abs(lag):],
                                    nts2[:-np.abs(lag)])
        else:
            corr, p_val = corr_func(nts1, nts2)
        corr_range.append(corr)
        p_val_arr.append(p_val)
    corr_range = np.array(corr_range)
    # if cutoff > 1:
    #     corr_range = flt.apply_butter_filter(corr_range, cutoff=cutoff)

    max_dict = gut.find_local_max_xy(data=corr_range, x=tau_arr)
    min_dict = gut.find_local_min_xy(data=corr_range, x=tau_arr)

    ll_dict = {'corr': corr_range,
               'p_val': np.array(p_val_arr),
               'tau': tau_arr,
               'tau_max': max_dict['x_max'],
               'max': max_dict['max'],
               'all_max': max_dict['val'],
               'all_tau_max': max_dict['x'],
               'tau_min': min_dict['x_min'],
               'min': min_dict['min'],
               'all_min': min_dict['val'],
               'all_tau_min': min_dict['x']
               }

    return ll_dict


def find_idx_tp_in_array(times, tps):
    idx_list = []
    if type(times) == xr.DataArray:
        times = times.time.data
    for tp in tps:
        idx_list.append(np.where(times == tp)[0])
    return np.array(idx_list).flatten()


def create_xr_ts(data, times):
    return xr.DataArray(data=data,
                        dims=["time"],
                        coords={"time": times}
                        )


def create_zero_ts(times):
    return create_xr_ts(data=np.zeros(len(times)), times=times)


def set_tps_val(ts, tps, val, replace=True):
    times_ts = ts.time

    for x in tps.data:
        if x not in times_ts:
            gut.myprint(f'WARNING! tp not in ds: {x}')
    if replace:
        ts.loc[dict(time=tps)] = val
    else:
        ts.loc[dict(time=tps)] += val
    return ts


def get_time_derivative(ts, dx=1):
    dtdx = np.gradient(ts.data, dx)

    return gut.create_xr_ds(
        data=dtdx,
        dims=ts.dims,
        coords=ts.coords,
        name=f'{ts.name}/dt'
    )


def select_time_snippets(ds, time_snippets):
    """Cut time snippets from dataset and concatenate them.

    Args:
        ds (xr.Dataset): Dataset to snip.
        time_snippets (np.array): Array of n time snippets
            with dimension (n,2).

    Returns:
        (xr.Dataset): Dataset with concatenate times
    """
    ds_lst = []
    for time_range in time_snippets:
        # ds_lst.append(ds.sel(time=slice(time_range[0], time_range[1])))
        ds_lst.append(get_sel_time_range(
            ds=ds, time_range=time_range, verbose=False))

    ds_snip = xr.concat(ds_lst, dim='time')

    return ds_snip
