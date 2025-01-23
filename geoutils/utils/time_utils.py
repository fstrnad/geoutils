import re
import datetime
import math
import geoutils.utils.spatial_utils as sput
import geoutils.tsa.filters as flt
import scipy.stats as st
import pandas as pd
import xarray as xr
import numpy as np
import copy

from importlib import reload
import geoutils.utils.statistic_utils as sut
import geoutils.utils.general_utils as gut

reload(gut)
reload(sut)

months = np.array(
    [
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
    ]
)

jjas_months = np.array(
    [
        "Jun",
        "Jul",
        "Aug",
        "Sep",
    ]
)


def get_time_dim(da):
    dims = gut.get_dimensions(da)
    if "time" in dims:
        return "time"
    else:
        spatial_dims = ["lat", "lon"]
        if len(dims) == 3:
            time_dim = [dim for dim in dims if dim not in spatial_dims]
            if len(time_dim) == 1:
                return time_dim[0]
        else:
            raise ValueError(f"Time dimension not found in {dims}!")


def assert_has_time_dimension(da):
    """
    Assert that a given xarray DataArray has a time dimension.

    Parameters
    ----------
    da : xarray.DataArray
        The input DataArray to check.

    Raises
    ------
    ValueError
        If the input DataArray does not have a time dimension.
    """
    if not isinstance(da, xr.DataArray) and not isinstance(da, xr.Dataset):
        raise ValueError(
            f"Data has to be of xarray type but is type {type(da)}")

    if "time" not in da.dims:
        raise ValueError(
            f"The input DataArray '{da.name}' does not have a time dimension."
        )


def reset_time(dataset, time_dim="time"):
    """
    Resets the time dimension of an xarray dataset to always have 0 hours.

    Parameters:
    -----------
    dataset: xarray dataset
        The dataset to reset the time dimension for.

    Returns:
    --------
    xarray dataset
        The dataset with the time dimension reset.
    """
    # Set the hour of each time value to 0
    time_values = dataset.time.values.astype("M8[s]").astype(datetime.datetime)
    time_values = np.array([dt.replace(hour=0) for dt in time_values])
    time_values = time_values.astype("M8[s]")

    # Update the dataset with the new time dimension
    dataset = dataset.assign_coords({time_dim: time_values})

    return dataset


def check_timepoints_in_dataarray(dataarray, timepoints, verbose=True):
    """
    Check if all time points in a set exist in an xr.Dataarray with a time dimension.

    Parameters:
    -----------
    dataarray : xr.Dataarray
        The data array to check for the time points.
    timepoints : xr.Dataarray
        The set of time points to check for in the data array.

    Returns:
    --------
    bool
        True if all time points exist in the data array, False otherwise.
    """
    assert_has_time_dimension(dataarray)

    # Get the time values in the data array
    dataarray_times = dataarray.time

    # Get the time values in the set of time points
    if gut.is_single_tp(tps=timepoints):
        timepoints_times = [timepoints.time]
    else:
        timepoints_times = timepoints.time

    # Check if all time points in the set exist in the data array
    points_in_array = True
    for time in timepoints_times:
        if time.data not in dataarray_times.data:
            tpstr = tp2str(time)
            gut.myprint(f"{tpstr} not in dataset!", verbose=verbose)
            points_in_array = False

    return points_in_array


def get_index_of_month(month):
    idx = int(np.argwhere(months == month)[0])
    if idx not in np.arange(0, len(months)):
        raise ValueError(f"This month does not exist:{month}!")
    # Attention, this idx starts with 0, so Jan=0, Feb=1, ... Dec=11

    return idx


def days_in_month(month):
    if isinstance(month, str):
        month = get_index_of_month(month=month)
    if month == 2:
        return 28
    elif month in [4, 6, 9, 11]:
        return 30
    else:
        return 31


# def get_month_number(month): return int(get_index_of_month(month) + 1)


def get_month_number(*month_list):
    idx_lst = []
    for month in month_list:
        idx = get_index_of_month(month) + 1
        idx_lst.append(idx)
    if len(idx_lst) == 1:
        return int(idx_lst[0])
    return idx_lst


def num2str_list(num_list):
    str_list = []
    for num in num_list:
        str_list.append(num2str(num))
    return str_list


def num2str(mi):
    if isinstance(mi, int) or isinstance(mi, np.int64):
        mstr = f"{mi}" if mi > 9 else f"0{mi}"
    else:
        print(type(mi))
        raise ValueError(
            f"Month number should be integer but is of type {type(mi)}!")

    return mstr


def month2str(month):
    mi = get_month_number(month)
    mstr = num2str(mi)
    return mstr


def get_month_name(month_number):
    if not isinstance(month_number, int):
        raise ValueError(f"Month is not integer but is {type(month_number)}!")
    if month_number < 1 or month_number > 12:
        raise ValueError(
            f"Month should be in range 1-12 but is {month_number}!")
    return months[month_number - 1]


def get_netcdf_encoding(
    ds, calendar="gregorian",
    units="hours since 1900-01-01T00:00",
    verbose=True,
    hours_to_zero=True,
):

    time = ds.time
    if check_AR_strings(da=time):
        time = convert_AR_xarray_to_timepoints(da=time)

    time = time.convert_calendar(calendar=calendar)
    gut.myprint("Set time to np.datetime[ns] time format!", verbose=verbose)
    ds["time"] = time.data.astype("datetime64[ns]")
    freq = get_frequency(ds)
    if freq != "hour":
        gut.myprint("set hours to 0", verbose=hours_to_zero)
        # This avoids problems with time encoding at 0h and 11h!
        ds = set_hours_to_zero(x=ds) if hours_to_zero else ds

    # ds = ds.transpose('time', 'lat', 'lon
    ds.time.attrs.pop("calendar", None)
    # ds.time.attrs.update({'calendar': '365_day'})
    ds.time.encoding["calendar"] = calendar
    ds.time.encoding["units"] = units
    ds.time.attrs["standard_name"] = "time"
    ds.time.attrs["long_name"] = "time"
    ds.time.attrs["axis"] = "T"

    return ds


def get_ts_arr_ds(da):
    """Get an array of time series for multiple data points.

    Args:
        da (xr.Dataarray): Xr.dataarray of type points, time

    Returns:
        np.array: 2d np.array of time series.
    """
    if da.dims == ("time", "points"):
        time_series_arr = da.data.T
    elif da.dims == ("points", "time"):
        time_series_arr = da.data
    elif da.dims == ("lat", "lon", "time"):
        time_series_arr = da.data.reshape(
            (da.shape[0] * da.shape[1], da.shape[2]))
    else:
        raise ValueError(f"this dimension is unknown: {da.dims}!")
    return time_series_arr


def get_num_tps(ds):
    return len(ds.time)


def get_max_num_tps(ds, q=None):
    time_steps = get_num_tps(ds)
    if q is not None:
        max_num_events = int(np.ceil(time_steps * (1 - q)))

    return max_num_events


def get_sel_tps_ds(
    ds,
    tps,
    remove_tp=False,
    drop_dim=True,
    verbose=False,
):
    if not isinstance(ds, xr.DataArray) and not isinstance(ds, xr.Dataset):
        raise ValueError(
            f"Data has to be of xarray type but is type {type(ds)}")

    if isinstance(tps, xr.DataArray):
        tps = tps.time
    else:
        tps = create_xr_tps(times=tps)
    stp = gut.is_single_tp(tps=tps)
    if not stp:
        start_month, end_month = get_month_range(tps)
    if not stp:
        if len(tps) == 0:
            gut.myprint(f"Empty list of time points")
            return []
    if gut.is_datetime360(tps):
        tps_sel = tps
        ds_sel = ds.sel(time=tps_sel, method="nearest")
    else:
        if remove_tp:
            ds_sel = ds.sel(time=tps, method="nearest")
        else:
            # ds_max = compute_timemax(ds=ds, timemean=timemean) tps_max =
            # compute_timemax(ds=tps, timemean=timemean)

            # Build always intersection
            if not stp:
                tps_sel = np.intersect1d(ds.time, tps)
                if len(tps_sel) == 0:
                    gut.myprint("No tps in intersection of dataset!")
                    return []
            else:
                if not check_timepoints_in_dataarray(
                    dataarray=ds, timepoints=tps, verbose=verbose
                ):
                    gut.myprint(f"WARNING: Single {tps} not in dataset!")
                    return []
                else:
                    tps_sel = tps

            # Attention for time points out of range!
            ds_sel = ds.sel(time=tps_sel, method="nearest")
            # Remove duplicates
            if not stp:
                ds_sel = remove_duplicate_times(da=ds_sel)
                # restrict to month range
                ds_sel = get_month_range_data(
                    dataset=ds_sel,
                    start_month=start_month,
                    end_month=end_month,
                    verbose=False,
                )
            else:
                if drop_dim:
                    if "time" in list(ds_sel.dims):
                        ds_sel = ds_sel.mean(dim="time")

    return ds_sel


def get_mean_tps(
    da,
    tps,
    varname=None,
    sig_test=True,
    #  corr_type='dunn',
    corr_type=None,
    first_dim="lev",
    alpha=0.05,
):
    """Get mean of dataarray for specific time points."""

    if gut.is_single_tp(tps):
        gut.myprint(f"Single time point {tps} selected!")
        this_da = get_sel_tps_ds(ds=da, tps=tps)
        return this_da, xr.ones_like(this_da)

    this_comp = get_sel_tps_ds(ds=da, tps=tps)
    dims = gut.get_dims(this_comp)

    if sig_test:
        if varname is None and isinstance(this_comp, xr.Dataset):
            raise ValueError(
                """Significance test can only be applied on specific dataarray.
                Please specify varname!"""
            )
        dims = gut.delete_element_from_arr(dims, "time")
        mean, pvalues_ttest = sut.ttest_field(this_comp, da, zdim=dims)
        mask = sut.field_significance_mask(
            pvalues_ttest, alpha=alpha, corr_type=corr_type
        )

        if first_dim in dims:
            dims = gut.set_first_element(dims, first_dim)
            mean = mean.transpose(*dims)
            mask = mask.transpose(*dims)
        return mean, mask
    else:
        mean = this_comp.mean(dim="time")
        if first_dim in dims:
            dims = gut.set_first_element(dims, first_dim)
            mean = mean.transpose(*dims)
        return mean


def normlize_time_slides(data, min=0, max=1):
    mean_data_arr = []
    times = data.time
    for i, (tp) in enumerate(times):
        mean_data = get_sel_tps_ds(ds=data, tps=[tp.data]).mean(
            dim="time"
        )  # get single time steps
        res_ds = sut.normalize(data=mean_data, min=min, max=max)
        mean_data_arr.append(res_ds.to_dataset(name="norm"))
    xr_new = xr.concat(mean_data_arr, times)

    return xr_new


def get_sel_tps_lst(ds, tps_lst, remove_tp=False, drop_dim=True, verbose=False):
    ds_sel = ds
    for tps in tps_lst:
        ds_sel = get_sel_tps_ds(
            ds=ds_sel, tps=tps, remove_tp=remove_tp, drop_dim=drop_dim, verbose=verbose
        )

    return ds_sel


def get_sel_years_data(
    ds,
    years,
    start_day=None,
    start_month=None,
    end_day=None,
    end_month=None,
    include_last=True,
):
    ds_range = get_month_day_range(
        da=ds,
        start_day=start_day,
        end_day=end_day,
        start_month=start_month,
        end_month=end_month,
        include_last=include_last,
    )
    ds_range_years = get_values_by_year(dataarray=ds_range, years=years)
    return ds_range_years


def get_sel_years_dates(
    years,
    start_day=None,
    start_month=None,
    end_day=None,
    end_month=None,
    freq="D",
    include_last=True,
):
    if isinstance(years[0], str):
        years = str2datetime(years)
    if isinstance(years, list) and isinstance(years[0], xr.DataArray):
        years = merge_time_arrays(years)
    sd, ed = get_start_end_date(data=years)
    if include_last:
        ed = add_time_window(ed, time_step=1, freq="Y")
    all_dates = get_dates_in_range(start_date=sd, end_date=ed, freq=freq)
    sel_dates = get_sel_years_data(
        ds=all_dates,
        years=years,
        start_day=start_day,
        end_day=end_day,
        start_month=start_month,
        end_month=end_month,
    )

    return sel_dates


def remove_time_points(dataset, time_points, verbose=False):
    """
    Remove specified time points from an xarray Dataset.

    Parameters:
        dataset (xarray.Dataset): The input xarray Dataset from which time points should be removed.
        time_points (xarray.DataArray): An xarray DataArray containing time points to be removed.

    Returns:
        xarray.Dataset: The dataset with the specified time points removed.
    """
    if isinstance(time_points, list):
        time_points = merge_time_arrays(time_points)
    if isinstance(time_points, xr.DataArray):
        time_points = time_points.time

    # Convert time_points to a set for faster lookups
    rem_tps = get_intersect_tps(dataset.time, time_points)
    gut.myprint(f"Remove {len(rem_tps)} time points!", verbose=verbose)
    # Select time points that are not in the time_points_set
    filtered_dataset = dataset.drop_sel(time=rem_tps)

    return filtered_dataset


def remove_consecutive_tps(
    tps, steps=0, start=1, remove_last=True, nneighbours=False, verbose=False
):
    """Removes consecutive steps in a set of time points until steps after.

    Args:
        tps (xr.DataArray): time points steps (int): number of steps to have no
        consecutive time points.
        remove_last (bool, optional): remove last time point. Otherwise remove first time points. Defaults to True.
    """
    if start < 1 or steps < 1:
        gut.myprint(
            f"WARNING! start {start}/steps {steps} has to be > 1! Not removed!"
        )
        return tps
    if start > steps:
        raise ValueError(f"Start {start} has to be < steps {steps}!")
    num_init_tps = len(tps)
    rem_tps = copy.deepcopy(tps)
    common_tps = []
    for step in range(start, steps + 1):
        tps_step = add_time_step_tps(tps=rem_tps, time_step=step)
        this_common_tps = get_common_tps(tps1=rem_tps, tps2=tps_step)
        common_tps.append(this_common_tps)
    # common_tps are potential time points that are consecutive
    if len(common_tps) > 1:
        common_tps = merge_time_arrays(common_tps, verbose=verbose)
    else:
        common_tps = common_tps[0]

    # Consecutive to consecutive time points are not deleted!
    if nneighbours:
        for this_step in range(1, steps + 1):
            cons_common_tps = get_common_tps(
                tps1=common_tps,
                tps2=add_time_step_tps(tps=common_tps, time_step=this_step),
            )
            if len(cons_common_tps) > 0:
                common_tps = common_tps.drop_sel(time=cons_common_tps)

    if len(common_tps) > 0:
        if remove_last:
            rem_tps = rem_tps.drop_sel(time=common_tps)

        else:
            first_tps = add_time_step_tps(common_tps, time_step=-step)
            rem_tps = rem_tps.drop_sel(time=first_tps)
    gut.myprint(f"Removed {num_init_tps - len(rem_tps)} time points!",
                verbose=verbose)

    return rem_tps


def get_common_tps(tps1, tps2, offset=0, delay=0, step=1):
    """Get the common tps between two set of tps. Can also be used to get all tps that
    are a certain offset later until some max delay.

    Args:
        tps1 (xr.DataArray): dataarray containing time points. tps2 (xr.DataArray):
        dataarray containing time points. offset (int, optional): offset. Defaults to 0.
        delay (int, optional): maximum delay. Defaults to 0. step (int, optional):
        stepsize of time points later. Defaults to 1.

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
        raise ValueError("Only positive delay allowed!")
    else:
        common_tps = []
        for offset in np.arange(offset, delay + 1, step):
            # tps2 that have a delay are shifted by this delay backward
            tps2_tmp = add_time_step_tps(tps2.time, time_step=-1 * offset)
            tps = np.sort(np.intersect1d(tps1, tps2_tmp))
            common_tps.append(tps)

        tps = np.sort(np.unique(np.concatenate(common_tps, axis=0)))

    tps = gut.create_xr_ds(data=tps, dims=["time"], coords={"time": tps})

    return tps


def get_tps_month(
    ds,
    month,
):

    times = ds.time
    tps_month = get_month_range_data(times, start_month=month, end_month=month)

    if len(tps_month) == 0:
        raise ValueError("Tps not in dataset!")

    return tps_month


def get_np64(date):
    if not isinstance(date, np.datetime64):
        if isinstance(date, xr.DataArray):
            date = date.time.data
        date = np.datetime64(date)
    return date


def is_tp_smaller(date1, date2):
    date1 = get_np64(date1)
    date2 = get_np64(date2)

    bool_date = date1 < date2
    return bool_date


def find_common_time_range(time_series_array, round_hour=True):
    """
    Find the earliest and latest time points that are within all the time series in an
    array.

    Parameters:
        time_series_array (array-like): Array containing xarray DataArrays representing
        time series.

    Returns:
        tuple: A tuple of two datetime64 objects representing the earliest and latest
        common time points.
    """
    # Initialize the earliest and latest time points as None
    earliest_time = None
    latest_time = None

    # Iterate over the time series array
    for time_series in time_series_array:
        # Get the time range of the current time series
        time_range = time_series.time.values

        # Update the earliest and latest time points
        if earliest_time is None or time_range[0] > earliest_time:
            earliest_time = time_range[0]
        if latest_time is None or time_range[-1] < latest_time:
            latest_time = time_range[-1]

    earliest_time = create_xr_tp(earliest_time)
    latest_time = create_xr_tp(latest_time)

    # Round the earliest and latest time points to the nearest hour
    if round_hour:
        earliest_time = set_hours_to_zero(x=earliest_time)
        latest_time = set_hours_to_zero(x=latest_time)

    return earliest_time, latest_time


def is_larger_as(t1, t2):
    if isinstance(t1, xr.DataArray):
        t1 = t1.time.values
    if isinstance(t2, xr.DataArray):
        t2 = t2.time.values
    if isinstance(t1, str):
        t1 = str2datetime(t1)
    if isinstance(t2, str):
        t2 = str2datetime(t2)
    if gut.check_any_type([t1, t2], float):
        raise ValueError("Time points must not be floats!")

    return t1 > t2


def get_time_range_data(
    ds, time_range=None, start_month=None, end_month=None,
    freq="D", verbose=False, check=True,
):
    if time_range is None:
        return ds
    if check:
        sd, ed = get_time_range(ds)
        if isinstance(time_range[0], str):
            time_range_0 = str2datetime(time_range[0], verbose=False)
        else:
            time_range_0 = time_range[0]
        if isinstance(time_range[-1], str):
            time_range_1 = str2datetime(time_range[-1], verbose=False)
        else:
            time_range_1 = time_range[-1]
        if time_range_0 < sd:
            gut.myprint(
                f"WARNING: Selected {time_range_0} smaller dataset {sd}",
                verbose=verbose,
            )
            time_range_0 = sd
        if time_range_1 > ed:
            gut.myprint(
                f"WARNING: Selected {time_range_1} larger dataset {ed}", verbose=verbose
            )
            time_range_1 = ed

        if gut.is_datetime360(time_range_0):
            time_range_1 = add_time_window(time_range_1, freq=freq)

        tps = get_dates_of_time_range(
            time_range=[time_range_0, time_range_1], freq=freq
        )
        time_range = [tps[0], tps[-1]]
    if time_range is not None:
        ds_sel = ds.sel(time=slice(*time_range))
    else:
        ds_sel = ds
    if start_month is not None or end_month is not None:
        ds_sel = get_month_range_data(
            dataset=ds_sel, start_month=start_month, end_month=end_month
        )

    return ds_sel


def get_data_timerange(data, time_range=None, verbose=True):
    """Gets data in a certain time range.
    Checks as well if time range exists in file!

    Args:
        data (xr.Dataarray): xarray dataarray
        time_range (list, optional): List dim 2 that contains the time interval. Defaults to None.

    Raises:
        ValueError: If time range is not in time range of data

    Returns:
        xr.Dataarray: xr.Dataarray in seleced time range.
    """

    # if isinstance(time_range[0], np.datetime64):
    #     time_range =

    td = data.time.data
    if time_range is not None:
        if (is_larger_as(td[0], time_range[0])) or (
            is_larger_as(time_range[1], td[-1])
        ):
            raise ValueError(
                f"Chosen time {time_range} out of range {td[0]} - {td[-1]}!"
            )
        else:
            sd = tp2str(time_range[0])
            ed = tp2str(time_range[-1])
            gut.myprint(f"Time steps within {sd} to {ed} selected!")
        # da = data.interp(time=t, method='nearest')
        da = data.sel(time=slice(time_range[0], time_range[1]))
    else:
        da = data
    tr = get_time_range(ds=da)
    gut.myprint(f"Load data in time range {tr}!", verbose=verbose)
    return da


def split_by_year(ds, start_month="Jan", end_month="Dec"):
    """
    Splits an xarray object with a time dimension spanning multiple years into individual
    datasets for each year.

    Parameters:
        ds (xarray.DataArray or xarray.Dataset): The xarray object to split.

    Returns:
        list: A list of xarray datasets, with each dataset representing one year.
    """
    if start_month != "Jan" or end_month != "Dec":
        ds = get_month_range_data(
            dataset=ds, start_month=start_month, end_month=end_month
        )

    time_dim = ds.time.dims[0]
    grouped = ds.groupby(time_dim + ".year")
    return [group for _, group in grouped]


def is_in_month_range(month, start_month, end_month):
    start_month_idx = get_month_number(start_month)
    end_month_idx = get_month_number(end_month)

    if start_month_idx <= end_month_idx:
        mask = (month >= start_month_idx) & (month <= end_month_idx)
    else:
        mask = (month >= start_month_idx) | (month <= end_month_idx)
    return mask


def get_month_range_data(dataset,
                         start_month="Jan",
                         end_month="Dec",
                         set_zero=False,
                         verbose=False):
    """
    This function generates data within a given month range. It can be from smaller month
    to higher (eg. Jul-Sep) but as well from higher month to smaller month (eg. Dec-Feb)

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
    if start_month is None or end_month is None:
        raise ValueError("Please specify start and end month!")
    if verbose:
        gut.myprint(f"Select data from {start_month} - {end_month}!")

    if set_zero:
        # sets everything outside month range to 0
        seasonal_data = get_month_range_zero(dataarray=dataset,
                                             start_month=start_month,
                                             end_month=end_month)
    else:
        seasonal_data = dataset.sel(
            time=is_in_month_range(dataset["time.month"], start_month, end_month)
        )

    return seasonal_data


def number2str(day):
    day = str(int(day)) if day >= 10 else f"0{int(day)}"
    return day


def get_month_day_range(
    da,
    start_month=None,
    start_day=None,
    end_month=None,
    end_day=None,
    include_last=False,
):
    """
    Selects all values of an xarray DataArray that fall within a specified range of months
    and days, across all years.

    Parameters
    ----------
    da : xarray.DataArray
        The input dataarray.
    start_month : int
        The starting month of the range, as an integer from 1 to 12.
    start_day : int
        The starting day of the range, as an integer from 1 to 31.
    end_month : int
        The ending month of the range, as an integer from 1 to 12.
    end_day : int
        The ending day of the range, as an integer from 1 to 31.

    Returns
    -------
    xarray.DataArray
        A new dataarray that includes all values within the specified range of months and
        days.

    """

    if start_month is None and end_month is None:
        return da
    else:
        sy, ey, num_years = get_time_range_years(dataarray=da)
        if include_last:
            ey += 1
            num_years += 1
        start_month = start_month if start_month is not None else "Jan"
        start_month = (
            get_month_number(start_month)
            if isinstance(start_month, str)
            else start_month
        )
        end_month = end_month if end_month is not None else "Dec"

        end_month = (
            get_month_number(end_month) if isinstance(
                end_month, str) else end_month
        )
        start_day = number2str(
            start_day) if start_day is not None else number2str(1)
        end_day = (
            number2str(end_day)
            if end_day is not None
            else number2str(days_in_month(end_month))
        )
        smi = number2str(start_month)
        emi = number2str(end_month)
        ranges = []
        for year in range(sy, ey + 1, 1):
            sd = str2datetime(f"{year}-{smi}-{start_day}")
            ed = str2datetime(f"{year}-{emi}-{end_day}")
            ranges.append([sd, ed])
        dates = get_dates_of_time_ranges(ranges)
        selected_data = get_sel_tps_ds(ds=da, tps=dates)

    return selected_data


def get_idx_months(times, start_month, end_month):
    month_dates = get_month_range_data(
        dataset=times, start_month=start_month, end_month=end_month
    ).data
    fulltime = times.data
    idx_months = np.where(np.in1d(fulltime, month_dates))[0]
    idx_non_months = np.where(~np.in1d(fulltime, month_dates))[0]

    return idx_months, idx_non_months


def get_month_range_zero(dataarray, start_month, end_month):
    dataarray_copy = dataarray.copy()
    times = dataarray.time
    idx_in_months, idx_out_months = get_idx_months(
        times=times, start_month=start_month, end_month=end_month
    )
    non_dates = times.data[idx_out_months]

    dataarray_copy.loc[dict(time=non_dates)] = 0

    return dataarray_copy


def get_idx_tps_times(tps, times):
    """Get the indices of time points in another time series.

    Args:
        tps (list): list of time points times (xr.dataarray): dataarray of time series

    Returns:
        list: list of indices of tps in time series.
    """
    tps_idx = np.where(np.in1d(times, tps))[0]
    return tps_idx


def time_difference_in_hours(time1, time2, abs_diff=True):
    """
    Calculate the time difference in hours between two time points as xarray dataarrays.

    Parameters: time1 (xarray.DataArray): First time point. time2 (xarray.DataArray):
    Second time point.

    Returns: time_diff (float): Time difference in hours.
    """
    # Convert time1 and time2 to pandas Timestamp objects
    pd_time1 = pd.to_datetime(str(time1.values))
    pd_time2 = pd.to_datetime(str(time2.values))

    # Calculate the time difference in hours using the timedelta method
    tdelta = pd_time2 - pd_time1
    time_diff = tdelta.total_seconds() / 3600

    # Always return a positive time difference
    if abs_diff:
        time_diff = np.abs(time_diff)

    return time_diff


def set_hours_to_zero(x):
    """
    Sets all hours to 0 in an xarray object with a time dimension.

    Parameters:
        x (xarray.DataArray or xarray.Dataset): The input xarray object.

    Returns:
        xarray.DataArray or xarray.Dataset: The modified xarray object with hours set to
        0.
    """
    # Set hours to 0 using the dt accessor
    x["time"] = x["time"].dt.floor("D")

    return x


def get_month_range(da):
    """
    Get the month range of an xarray DataArray with a time dimension.

    Parameters
    ----------
    da : xarray.DataArray
        The input DataArray. It should have a time dimension.

    Returns
    -------
    tuple of two integers
        The first and last month of the year present in the DataArray's time dimension.
    """
    months = da.time.dt.month
    min_month, max_month = int(months.min()), int(months.max())
    start_month = get_month_name(month_number=min_month)
    end_month = get_month_name(month_number=max_month)
    return start_month, end_month


def get_time_range_years(dataarray: xr.DataArray) -> tuple:
    """
    Given an xarray DataArray, return the start and end year of the time dimension as well
    as the difference in years between them.

    Parameters
    ----------
    data : xr.DataArray
        The input xarray DataArray

    Returns
    -------
    tuple
        A tuple of three integers: the start year, end year, and difference in years

    Raises
    ------
    ValueError
        If the input data does not have a time dimension

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> data = xr.DataArray(np.random.rand(5, 10, 10), dims=('time', 'lon', 'lat'),
    ...                     coords={'time': xr.cftime_range(start='2000-01-01', periods=5, freq='YS'),
    ...                             'lon': np.linspace(-180, 180, 10),
    ...                             'lat': np.linspace(-90, 90, 10)})
    >>> start_year, end_year, num_years = get_time_range(data)
    >>> print(f"Start year: {start_year}, End year: {end_year}, Number of years: {num_years}")
    Start year: 2000, End year: 2004, Number of years: 5
    """
    assert_has_time_dimension(da=dataarray)

    start_year = int(dataarray.time.dt.year.min())
    end_year = int(dataarray.time.dt.year.max())
    num_years = end_year - start_year + 1

    return start_year, end_year, num_years


def get_sy_ey_time(times, sy=None, ey=None, sm=None, em=None):
    """Returns the start and end year of a xr Dataarray
    datetime object

    Args:
        times (xr.Datetime): xr.Datetime object that contains time sy (int, optional):
        other startyear if specified. Defaults to None. ey (int, optional): end endyear.
        Defaults to None.

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
    base_period = np.array([data.time.data.min(), data.time.data.max()])

    sd = xr.DataArray(base_period[0], coords={"time": base_period[0]})
    ed = xr.DataArray(base_period[1], coords={"time": base_period[1]})
    return sd, ed


def get_start_end_date_shift(time, sm, em, shift=0):
    """Same as normal get_start_date_year but the start and end shifted by shift days

    Args:
        sm (str): Start Month em (str): End Month sy (int): start Year ey (int): end Year
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


def str2datetime(string, numpy=True, verbose=False):
    if type(string) is str:
        date = np.datetime64(string)
        if not numpy:
            y, m, d, h = get_date2ymdh(date=date)
            date = datetime.datetime(year=y, month=m, day=d, hour=h)
        date = xr.DataArray(date, coords={"time": date})
    else:
        date = string
        gut.myprint(f"WARNING {string} is not string!", verbose=verbose)
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


def get_frequency(x):
    """
    Determines the frequency of an xarray object with a time dimension.

    Parameters:
        x (xarray.DataArray or xarray.Dataset): The input xarray object.

    Returns:
        str: The frequency of the time dimension (daily, monthly, hourly, none).
    """
    # Convert the time dimension to a pandas Series
    time_series = pd.Series(x.time.values)

    # Compute the time differences between consecutive timestamps
    time_diff = time_series.diff()

    # Check the most common time difference
    most_common_diff = time_diff.value_counts().idxmax()

    # Determine the frequency based on the most common time difference
    if pd.Timedelta(days=1) == most_common_diff:
        return "day"
    elif (
        pd.Timedelta(days=30) == most_common_diff
        or pd.Timedelta(days=31) == most_common_diff
    ):
        return "month"
    elif pd.Timedelta(hours=1) >= most_common_diff and pd.Timedelta(hours=24) < most_common_diff:
        return "hour"
    elif pd.Timedelta(hours=6) == most_common_diff:
        return "hour"
    else:
        return "none"


def convert_time_resolution(dataarray, keep_time_points=6, average=False):
    """
    Convert an xarray DataArray with hourly time points to 6-hourly resolution.

    Parameters:
    ----------
    dataarray : xarray.DataArray
        Input DataArray with hourly time points.

    Returns:
    -------
    xarray.DataArray
        DataArray with only every 6th time point, representing 6-hourly resolution.
    """
    # Ensure the time dimension exists
    if "time" not in dataarray.dims:
        raise ValueError("The DataArray must have a 'time' dimension.")

    if average:
        return dataarray.rolling(time=6, center=False).mean()
    else:
        # Subset the DataArray to keep every 6th time point
        return dataarray.isel(time=slice(None, None, keep_time_points))


def check_hour_equality(da1, da2):
    """
    Check whether the hour of the time dimension is equal in two xarray DataArrays.

    Parameters:
    - da1 (xarray.DataArray): First data array.
    - da2 (xarray.DataArray): Second data array.

    Returns:
    - bool: True if the hour is equal, False otherwise.
    """

    # Ensure both DataArrays have a time dimension
    if "time" not in da1.dims or "time" not in da2.dims:
        raise ValueError("Both DataArrays must have a 'time' dimension.")

    # Extract time coordinates
    time1 = pd.Series(da1.time.values)
    time2 = pd.Series(da2.time.values)

    if len(time1) != len(time2):
        if len(time1) < len(time2):
            time2 = time2[0: len(time1)]
        elif len(time2) < len(time1):
            time1 = time1[0: len(time2)]
    # Check if the length of time dimensions is the same
    if len(time1) != len(time2):
        raise ValueError(
            f"Length of time dimensions {len(time1)} != {len(time2)}"
        )

    # Check hour equality for each timestamp
    for t1, t2 in zip(time1, time2):
        if t1.hour != t2.hour:
            gut.myprint(f"Hour mismatch: {t1} != {t2}")
            return False

    # If all hours are equal, return True
    return True


def check_hour_occurrence(da):

    # Ensure both DataArrays have a time dimension
    if "time" not in da.dims:
        raise ValueError("DataArrays must have a 'time' dimension.")

    # Extract time coordinates
    time1 = pd.Series(da.time.values)

    for t1 in time1:
        t_hour = t1.hour
        if t_hour != 0:
            gut.myprint(f"Hour mismatch: {t_hour}!")
            return False

    # If all hours are 0, return True
    return True


def get_tm_name(timemean):
    if timemean == "day":
        tm = "1D"
    elif timemean == "week":
        tm = "1W"
    elif timemean == "month":
        tm = "1MS"
    elif (
        timemean == "season"
    ):  # This is not just 3MS!  found here: https://github.com/pydata/xarray/issues/810
        tm = "Q-FEB"
    elif timemean == "year":
        tm = "1Y"
    elif timemean == "pentad" or timemean == "5D" or timemean == "5d":
        tm = "5D"
    elif timemean == "3D" or timemean == "3d":
        tm = "3D"
    elif timemean == "2D" or timemean == "2d":
        tm = "2D"
    elif timemean in ["1D", "1W", "1MS", "Q-FEB", "1Y", "5D", "3D", "2D"]:
        tm = timemean
    elif timemean == "all":
        tm = None
    else:
        raise ValueError(
            f"{timemean} does not exist! Choose week, month, season, year!"
        )

    return tm


def get_mean_time_series(da, lon_range, lat_range, time_roll=0, q=None):
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
    da_area = sput.cut_map(da, lon_range=lon_range, lat_range=lat_range)
    if q is None:
        ts_mean = da_area.mean(dim=("lon", "lat"), skipna=True)
    else:
        ts_mean = da_area.quantile(q=q, dim=("lon", "lat"), skipna=True)
    ts_std = da_area.std(dim=("lon", "lat"), skipna=True)
    if time_roll > 0:
        ts_mean = ts_mean.rolling(time=time_roll, center=True).mean()
        ts_std = ts_std.rolling(time=time_roll, center=True).mean()

    return ts_mean, ts_std


def compute_timemean(
    ds, timemean, dropna=True, groupby=False, verbose=True, reset_time=False
):
    """Computes the monmean average on a given xr.dataset

    Args:
        ds (xr.dataset): xr dataset for the dataset

    Returns:
        xr.dataset: monthly average dataset
    """

    if groupby:
        ds = ds.groupby(f"time.{timemean}").mean(dim="time")
        if reset_time:
            ds = gut.rename_dim(ds, timemean, name="time")
    else:
        if timemean is None:
            return ds
        tm = get_tm_name(timemean)

        if tm is None:
            return ds.mean(dim="time")

        gut.myprint(
            f"Compute {timemean}ly means of all variables!", verbose=verbose)
        if dropna:
            ds = ds.resample(time=tm).mean(
                dim="time", skipna=True).dropna(dim="time")
        else:
            ds = ds.resample(time=tm).mean(dim="time", skipna=True)

    return ds


def compute_mean(ds, dropna=True, verbose=False):
    """Computes the mean of a given xr.dataset

    Args:
        ds (xr.dataset): xr dataset for the dataset

    Returns:
        xr.dataset: monthly average dataset
    """
    tps = ds.time
    if gut.is_single_tp(tps=tps):
        gut.myprint("Single time point selected!", verbose=verbose)
        return ds
    else:
        gut.myprint(f"Compute mean of all variables!", verbose=verbose)
        if dropna:
            ds = ds.dropna(dim="time").mean("time")
        else:
            ds = ds.mean(dim="time")
    return ds


def compute_quantile(ds, q, dropna=True, verbose=False):
    """Computes the quantile of a given xr.dataset

    Args:
        ds (xr.dataset): xr dataset for the dataset

    Returns:
        xr.dataset: monthly average dataset
    """
    tps = ds.time
    if gut.is_single_tp(tps=tps):
        gut.myprint("Single time point selected!", verbose=verbose)
        return ds
    else:
        gut.myprint(f"Compute quantile of all variables!", verbose=verbose)
        if dropna:
            ds = ds.dropna(dim="time").quantile(q, dim="time")
        else:
            ds = ds.quantile(q, dim="time")
    return ds


def get_extremes(ds, q=0.9, dropna=True, verbose=False):
    quantile_val = compute_quantile(ds=ds, q=q, dropna=dropna, verbose=verbose)
    quantile_val_below = compute_quantile(
        ds=ds, q=1 - q, dropna=dropna, verbose=verbose
    )
    above_q = ds.where(ds > quantile_val).dropna(dim="time")
    below_q = ds.where(ds < quantile_val_below).dropna(dim="time")

    return above_q, below_q


def compute_sum(ds, dropna=True, verbose=False):
    """Computes the sum of a given xr.dataset

    Args:
        ds (xr.dataset): xr dataset for the dataset

    Returns:
        xr.dataset: monthly average dataset
    """
    tps = ds.time
    if gut.is_single_tp(tps=tps):
        gut.myprint("Single time point selected!", verbose=verbose)
        return ds
    else:
        gut.myprint(f"Compute sum of all variables!", verbose=verbose)
        if dropna:
            ds = ds.dropna(dim="time").sum("time")
        else:
            ds = ds.sum(dim="time")
    return ds


def compute_timemax(ds, timemean, dropna=True, verbose=True):
    """Computes the monmax on a given xr.dataset

    Args:
        ds (xr.dataset): xr dataset for the dataset

    Returns:
        xr.dataset: monthly average dataset
    """
    tm = get_tm_name(timemean)

    gut.myprint(
        f"Compute {timemean}ly maximum of all variables!", verbose=verbose)
    if dropna:
        ds = ds.resample(time=tm).max(
            dim="time", skipna=True).dropna(dim="time")
    else:
        ds = ds.resample(time=tm).max(dim="time", skipna=True)

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


def remove_duplicate_times(da):
    """
    Remove duplicate time points from an xarray DataArray.

    Parameters
    ----------
    da : xarray.DataArray
        The input DataArray. It should have a time dimension.

    Returns
    -------
    xarray.DataArray
        The DataArray with duplicate time points removed.
    """
    _, index = np.unique(da.time, return_index=True)
    return da.isel(time=index)


def compute_anomalies(
    dataarray,
    climatology_array=None,
    group=None,
    base_period=None,
    chunk_data=False,
    normalize=False,
    verbose=True,
):
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
    if group is None and climatology_array is None:
        raise ValueError(
            "ERROR! If climatology_array is None, group has to be specified!"
        )
    if base_period is None:
        base_period = np.array(
            [dataarray.time.data.min(), dataarray.time.data.max()])

    if group in ["dayofyear", "month", "season"]:
        if climatology_array is None:
            climatology_array = dataarray
        climatology = (
            climatology_array.sel(time=slice(base_period[0], base_period[1]))
            .groupby(f"time.{group}")
            .mean(dim="time")
        )
        std_climatology = (
            climatology_array.sel(time=slice(base_period[0], base_period[1]))
            .groupby(f"time.{group}")
            .std(dim="time")
        )
        if normalize:
            gut.myprint(f"Normalize {group}ly anomalies!")
            anomalies = dataarray.groupby(f"time.{group}") / std_climatology
            # Needs to be done in a two-step process, because the mean and std are calculated
            anomalies = (
                anomalies.groupby(f"time.{group}") -
                climatology / std_climatology
            )
        else:
            anomalies = dataarray.groupby(f"time.{group}") - climatology

        if chunk_data:
            anomalies = anomalies.chunk(dict(time=-1))
    else:
        month_ids = []
        if climatology_array is None:
            monthly_groups = dataarray.groupby("time.month")
            if group == "JJAS":
                for month in ["Jun", "Jul", "Aug", "Sep"]:
                    month_ids.append(get_month_number(month))
            elif (
                group == "DJFM"
            ):  # here it is important to include the December of the previous year
                for month in ["Dec", "Jan", "Feb", "Mar"]:
                    month_ids.append(get_month_number(month))
            else:
                raise ValueError(
                    f"ERROR! {group} is not a valid group for anomalies!")
            climatology = (
                monthly_groups.mean(dim="time").sel(
                    month=month_ids).mean(dim="month")
            )
        else:
            climatology = climatology_array
        anomalies = dataarray - climatology
    if verbose:
        gut.myprint(f"Created {group}ly anomalies!")

    var_name = anomalies.name
    anomalies = gut.rename_da(da=anomalies, name=f"{var_name}_an_{group}")

    return anomalies


def compute_anomalies_ds(
    ds,
    var_name=None,
    an_types=["month", "JJAS"],
    #  an_types=['dayofyear', 'month', 'JJAS'],
    verbose=True,
):
    vars = gut.get_vars(ds=ds)
    an_vars = [var_name] if var_name in vars else vars
    for var_name in an_vars:
        for an_type in an_types:
            if not gut.check_contains_substring(var_name, an_type):
                da_an = compute_anomalies(
                    ds[var_name], group=an_type, verbose=verbose)
                ds[da_an.name] = da_an
    return ds


def get_ee_ds(
    dataarray,
    q=0.95,
    threshold=None,
    min_threshold=None,
    reverse_threshold=False,
    th_eev=None,
    verbose=True,
):
    if threshold is None and q is None:
        raise ValueError("ERROR! Either q or threshold has to be    provided!")

    if threshold is not None:
        q_val_map = xr.where(~np.isnan(dataarray), threshold, np.nan)

    if min_threshold is not None:
        # Remove days without rain
        dataarray = dataarray.where(dataarray > min_threshold)
    if threshold is not None:
        gut.myprint(f'Compute extreme events with threshold {threshold}!')
        if q is not None:
            gut.myprint(f'q is given, but will be ignored!')
        if reverse_threshold:
            data_quantile = xr.where(dataarray < threshold, dataarray, np.nan)
        else:
            data_quantile = xr.where(dataarray > threshold, dataarray, np.nan)
    else:
        gut.myprint(f"Compute extreme events with quantile {q}!")
        # Gives the quanile value for each cell
        q_val_map = get_q_val_map(dataarray=dataarray, q=q)
        # Set values below quantile to 0
        if q > 0.5:
            data_quantile = xr.where(dataarray > q_val_map, dataarray, np.nan)
        elif q <= 0.5:
            data_quantile = xr.where(dataarray < q_val_map, dataarray, np.nan)
        else:
            raise ValueError(f"ERROR! q = {q} has to be in range [0, 1]!")
    if th_eev is not None:
        # Set values to 0 that have not at least the value th_eev
        data_quantile = xr.where(data_quantile > th_eev, data_quantile, np.nan)
    ee_map = data_quantile.count(dim="time")

    if verbose:
        tot_frac_events = float(ee_map.sum()) / dataarray.size
        gut.myprint(f"Fraction of events: {tot_frac_events}!")

    rel_frac_q_map = data_quantile.sum(dim="time") / dataarray.sum(dim="time")

    return q_val_map, ee_map, data_quantile, rel_frac_q_map


def get_q_val_map(dataarray, q=0.95):
    if q > 1 or q < 0:
        raise ValueError(f"ERROR! q = {q} has to be in range [0, 1]!")

    if 'time' not in dataarray.dims:
        raise ValueError("ERROR! No time dimension found!")
    q_val_map = dataarray.quantile(q, dim="time")
    return q_val_map


def get_ee_count_ds(ds, q=0.95):
    varnames = gut.get_varnames_ds(ds)
    if "evs" in varnames:
        evs = ds["evs"]
        evs_cnt = evs.sum(dim="time")
    else:
        raise ValueError("No Evs in dataset found!")
    return evs_cnt


def compute_evs(
    dataarray,
    q=0.9,
    threshold=None,
    reverse_treshold=False,
    min_threshold=None,
    th_eev=None,
    min_num_events=1,
    verbose=True,
):
    """Creates an event series from an input time series.

    Args:
        dataarray (xr.dataarray): The time series of a variable of q (float, optional):
        Quantile for defining an extreme event. Defaults to 0.95. min_threshold (int,
        optional): Threshold. Removes all values in time series.
            Eg. important to get wet days. Defaults to 1.
        th_eev (int, optional): Minimum value of an extreme event. Defaults to 15. min_num_events
        (int, optional): Minimum number of extreme event within 1 time series. Defaults to
        20.

    Raises:
        ValueError: [description] ValueError: [description]

    Returns:
        event_series (xr.Dataarray): Event time series of 0 and 1 (1=event). mask
        (xr.dataarray): Gives out which values are masked out.
    """
    # Compute percentile data, remove all values below percentile, but with a minimum of
    # threshold q
    _, ee_map, data_quantile, _ = get_ee_ds(
        dataarray=dataarray,
        q=q,
        threshold=threshold,
        reverse_threshold=reverse_treshold,
        min_threshold=min_threshold,
        th_eev=th_eev,
        verbose=verbose,
    )
    # Create mask for which cells are left out
    mask = ee_map > min_num_events
    final_data = data_quantile.where(mask, np.nan)

    event_series = xr.where(~np.isnan(final_data[:]), 1, 0)
    event_series = event_series.rename("evs")

    # Create new mask for dataset: Masked values are areas with no events!
    mask = xr.where(ee_map > min_num_events, 1, 0)
    fraction_masked = 1 - float(mask.sum()) / mask.size
    gut.myprint(f"Fraction of masked values: {fraction_masked:.2f}!")

    return event_series, mask


def detrend_dim(da, dim="time", deg=1, startyear=None, freq="D"):
    import cftime

    if startyear is None:
        p = da.polyfit(dim=dim, deg=deg)
        fit = xr.polyval(da[dim], p.polyfit_coefficients)
        start_val = fit[0]
        detrended_da = da - fit + start_val
    else:
        start_date, end_date = get_start_end_date(data=da)
        if gut.is_datetime360(time=da.time.data[0]):
            date_before_detrend = cftime.Datetime360Day(startyear - 1, 12, 30)
            date_start_detrend = cftime.Datetime360Day(startyear, 1, 1)
        else:
            date_before_detrend = np.datetime64(f"{startyear-1}-12-31")
            date_start_detrend = np.datetime64(f"{startyear}-01-01")
        gut.myprint(f"Start detrending from {date_start_detrend}...")
        da_no_detrend = get_time_range_data(
            ds=da,
            time_range=[start_date, date_before_detrend],
            freq=freq,
            verbose=False,
        )
        da_detrend = get_time_range_data(
            ds=da, time_range=[date_start_detrend,
                               end_date], freq=freq, verbose=False
        )
        p = da_detrend.polyfit(dim=dim, deg=deg)
        fit = xr.polyval(da_detrend[dim], p.polyfit_coefficients)
        start_val = fit[0]
        detrended_da = da_detrend - fit + start_val
        detrended_da = xr.concat([da_no_detrend, detrended_da], dim="time")

    return detrended_da


def correlation_per_timeperiod(x, y, time_period):
    """Correlation per time period.

    Args:
        x ([type]): [description] y ([type]): [description] time_period ([type]):
        [description]

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


def tp2str(tp, m=True, d=True, h=False):
    """Returns the string for np.datetime(64) object.

    Args:
        tp (np.datetime): time point m (bool, optional): Return month as well. Defaults to
        False. d (bool, optional): Return day as well. Defaults to False.

    Returns:
        str: string of the date
    """

    if isinstance(tp, xr.DataArray):
        tp = tp.time.data
    if isinstance(tp, str):
        return tp
    if gut.is_datetime360(tp):
        date = f"{tp.year}"
    else:
        ts = pd.to_datetime(str(tp))
        date = ts.strftime("%Y")
        if m:
            date = ts.strftime("%Y-%m")
        if d:
            date = ts.strftime("%Y-%m-%d")
        if h:
            date = ts.strftime("%Y-%m-%d-%H:00")
    return date


def tps2str(tps, m=True, d=True, h=False):
    if isinstance(tps, (xr.DataArray, xr.Dataset)):
        if gut.is_single_tp(tps=tps):
            return tp2str(tp=tps, m=m, d=d, h=h)
    else:
        tps_str = []
        for tp in tps:
            tps_str.append(tp2str(tp=tp, m=m, d=d, h=h))
        return tps_str


def get_ymdh_date(date):
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
        h = date.astype("M8[h]")

    return y, m, d, h


def get_date2ymdh(date):
    if isinstance(date, xr.DataArray):
        date = date.data
    else:
        date = np.datetime64(date)
    # this object knows what y, m, d, and hours are
    date = pd.to_datetime(date)

    yi = int(date.year)
    mi = int(date.month)
    di = int(date.day)
    hi = int(date.hour)

    return yi, mi, di, hi


def date2ymdhstr(date, seperate_hour=True):
    if isinstance(date, xr.DataArray):
        date = date.data
    else:
        date = np.datetime64(date)
    # this object knows what y, m, d, and hours are
    yi, mi, di, hi = get_date2ymdh(date)

    ystr = f"{yi}"
    mstr = f"{mi}" if mi > 9 else f"0{mi}"
    dstr = f"{di}" if di > 9 else f"0{di}"
    hstr = f"{hi}" if hi > 9 else f"0{hi}"

    str_hr = f"{ystr}{mstr}{dstr}{hstr}"
    str_day = f"{ystr}{mstr}{dstr}_{hstr}"
    strdate = (
        str_day if seperate_hour else str_hr
    )

    return strdate


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


def get_ym_date(date):
    if isinstance(date, xr.DataArray):
        date = date.data
    else:
        date = np.datetime64(date)
    y, m, d = get_ymd_date(date)
    mi = m.astype(int) % 12
    mname = months[mi]
    return f"{mname} {y}"


def add_time_window(date, time_step=1, freq="D"):
    """
    Add a specified number of days, months, or years to the time dimension of an xarray
    DataArray.

    Parameters
    ----------
    dataarray : xr.DataArray
        The input dataarray containing the time dimension to be modified.
    num : int
        The number of units (days, months or years) to add to the time dimension.
    freq : str, optional
        The frequency of the units to add. Valid values are 'days', 'months' or 'years'.
        Defaults to 'days'.

    Returns
    -------
    xr.DataArray
        The modified dataarray with the time dimension updated.
    """
    # Define the time delta to add based on the frequency parameter
    if freq == "D":
        tdelta = pd.DateOffset(days=time_step)
    elif freq == "M":
        tdelta = pd.DateOffset(months=time_step)
    elif freq == "Y":
        tdelta = pd.DateOffset(years=time_step)
    else:
        raise ValueError(f"Invalid '{freq}', must be 'D', 'M', or 'Y'")
    if isinstance(date, xr.DataArray):
        date = date.time

    if gut.is_single_tp(tps=date):
        shifted_time = pd.to_datetime([np.datetime64(date.time.data)]) + tdelta
        time_xr = xr.DataArray(shifted_time.values[0], coords={
                               "time": shifted_time[0]})
    else:
        shifted_time = pd.to_datetime(date.time.data) + tdelta
        # Convert the modified time dimension back to xarray.DataArray format
        time_xr = xr.DataArray(
            shifted_time.values, dims="time", coords={"time": shifted_time}
        )

    return time_xr


def add_time_step_tps(
    tps,
    time_step=1,
    freq="D",
):
    return add_time_window(date=tps, time_step=time_step, freq=freq)


def get_tps_range(
    tps,
    start=0,
    time_step=0,
    freq="D",
):
    if isinstance(tps, xr.DataArray):
        tps = tps.time
    if len(np.array([tps.time.data]).shape) == 1:
        tps = [tps]
    if start == 0 and time_step == 0:
        return tps
    else:
        ntps = []
        tps = add_time_step_tps(tps=tps, time_step=start, freq=freq)
        if time_step > 0:
            for tp in tps:
                ntp = add_time_step_tps(tps=tp, time_step=time_step, freq=freq)
                if time_step > 0:
                    new_tps = get_dates_in_range(
                        start_date=tp, end_date=ntp, freq=freq)
                else:
                    new_tps = get_dates_in_range(
                        start_date=ntp, end_date=tp, freq=freq)
                ntps.append(new_tps)
        else:
            ntps = tps
        ntps = merge_time_arrays(ntps, multiple=None, new_dim=False)
        ntps = remove_duplicate_times(ntps)

        return ntps


def add_time_step_tps_old(
    tps,
    time_step=1,
    freq="D",
):
    ntps = []
    if isinstance(tps, xr.DataArray):
        tps = tps.time
    if len(np.array([tps.time.data]).shape) == 1:
        tps = [tps]
    for tp in tps:
        ntp = add_time_window(date=tp, time_step=time_step, freq=freq)
        ntps.append(ntp)
    ntps = merge_time_arrays(ntps, multiple=None)

    return xr.DataArray(ntps, dims=["time"], coords={"time": ntps})


def merge_time_arrays(time_arrays, multiple="duplicate", new_dim=False, verbose=False):
    # Combine the time arrays into a single DataArray with a new "time" dimension
    if new_dim:
        combined_data = xr.concat(time_arrays, dim="t")
    else:
        combined_data = xr.concat(time_arrays, dim="time")
    if multiple is not None:
        gut.myprint(
            f"Group multiple files by {multiple} if time points occur twice!",
            verbose=verbose,
        )
    if multiple == "duplicate":
        # Remove duplicate time points
        combined_data = remove_duplicate_times(combined_data)
    elif multiple == "max":
        # Group the data by time and take the maximum value for each group
        combined_data = combined_data.groupby("time").max()
    elif multiple == "mean":
        combined_data = combined_data.groupby("time").mean()
    elif multiple == "min":
        combined_data = combined_data.groupby("time").min()

    # Sort the new "time" dimension
    combined_data = combined_data.sortby("time")

    return combined_data


def get_tw_periods(
    sd, ed, tw_length=1, tw_unit="Y", sliding_length=1, sliding_unit="M"
):

    ep = sd
    all_time_periods = []
    all_tps = []
    while ep < ed:
        ep = add_time_window(sd, time_step=tw_length, freq=tw_unit)
        if ep < ed:
            tw_range = get_dates_of_time_range([sd, ep])
            all_time_periods.append(tw_range)
            all_tps.append(ep)
            sd = add_time_window(
                sd, time_step=sliding_length, freq=sliding_unit)

    return {"range": all_time_periods, "tps": np.array(all_tps)}


def get_periods_tps(tps, start=0, end=1, freq="D", include_start=True):
    """Gives the all time points from tps to end."""
    if end == 0:
        return tps
    else:
        if not include_start and start == 0:
            sign = math.copysign(end)
            tps = add_time_step_tps(tps=tps, time_step=sign * 1)
            end += sign  # because we have shifted the end
        if start != 0:
            if start < 0:
                if start > end:
                    start, end = end, start  # always start from smaller number
            elif np.abs(start) == np.abs(end):
                return add_time_step_tps(tps=tps, time_step=start, freq=freq)
            else:
                gut.myprint(f"Warning! Start {start} is > end {end}!")
            stps = add_time_step_tps(tps=tps, time_step=start, freq=freq)

        else:
            stps = tps
        etps = add_time_step_tps(tps=tps, time_step=end, freq=freq)
        all_time_periods = []

        if gut.is_single_tp(stps):
            stps = [stps]
            etps = [etps]

        for idx, stp in enumerate(stps):
            etp = etps[idx]
            tw_range = get_dates_of_time_range([stp, etp], freq=freq)
            all_time_periods.append(tw_range)

        all_time_periods = np.concatenate(all_time_periods, axis=0)
        # Removes duplicates of time points
        all_time_periods = np.unique(all_time_periods)
        all_time_periods = create_xr_ts(
            data=all_time_periods, times=all_time_periods)
        all_time_periods = remove_duplicate_times(all_time_periods)

        return all_time_periods


def get_dates_of_time_range(time_range, freq="D", start_month="Jan", end_month="Dec"):
    dtype = f"datetime64[{freq}]"
    if type(time_range[0]) is str:
        gut.myprint("Convert String", verbose=False)
        time_range = np.array(time_range, dtype=dtype)
    if gut.is_datetime360(time=time_range[0]):
        date_arr = xr.cftime_range(
            start=time_range[0],
            end=time_range[-1],
            normalize=True,  # Normalize to Midnight
            freq=freq,
        )
    else:
        sp, ep = time_range[0], time_range[-1]
        if isinstance(sp, xr.DataArray):
            sp = sp.time.data
        if isinstance(ep, xr.DataArray):
            ep = ep.time.data

        sp, ep = np.sort([sp, ep])  # Order in time
        date_arr = get_dates_in_range(
            start_date=sp, end_date=ep, freq=freq, make_xr=False
        )
        # Include as well last time point
        date_arr = np.concatenate(
            [date_arr, [date_arr[-1] + np.timedelta64(1, freq)]], axis=0
        )

    date_arr = gut.create_xr_ds(
        data=date_arr, dims=["time"], coords={"time": date_arr})

    if start_month != "Jan" or end_month != "Dec":
        date_arr = get_month_range_data(
            dataset=date_arr, start_month=start_month, end_month=end_month
        )

    return date_arr


def get_dates_of_time_ranges(time_ranges, freq="D"):
    dtype = f"datetime64[{freq}]"
    arr = np.array([], dtype=dtype)
    for time_range in time_ranges:
        arr = np.concatenate(
            [arr, get_dates_of_time_range(
                time_range, freq=freq).time.data], axis=0
        )
    # Also sort the array
    arr = np.sort(arr)
    arr = gut.create_xr_ds(data=arr, dims=["time"], coords={"time": arr})
    return arr


def get_dates_in_range(start_date, end_date, freq="D", make_xr=True):

    if isinstance(start_date, xr.DataArray):
        start_date = np.datetime64(start_date.time.data)
    if isinstance(end_date, xr.DataArray):
        end_date = np.datetime64(end_date.time.data)
    tps = np.arange(start_date, end_date, dtype=f"datetime64[{freq}]")

    if make_xr:
        tps = create_xr_ts(data=tps, times=tps)

    return tps


def get_dates_for_time_steps(start="0-01-01", num_steps=1, freq="D"):
    """Computes an array of time steps for a number of steps starting from a
    specified date.

    Args:
        start (str, optional): startpoint. Defaults to '0-01-01'. num_steps (int,
        optional): number of steps. Defaults to 1. freq (str, optional): frequency of
        timesteps (day, month, year...). Defaults to 'D'.
    """
    dates = []
    for step in np.arange(num_steps):
        dates.append(add_time_window(
            start, time_step=step, freq=freq).time.data)
    return np.array(dates)


def get_dates_arr_for_time_steps(tps, num_steps=1, freq="D"):
    dates = np.array([], dtype="datetime64")
    for tp in tps:
        this_dates = get_dates_for_time_steps(
            start=tp, num_steps=num_steps, freq=freq)
        dates = np.concatenate([dates, this_dates], axis=0)

    dates = xr.DataArray(data=dates, dims=["time"], coords={"time": dates})

    return dates


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


def arr_lagged_ts(ts_arr, lag):
    times = ts_arr[0].time
    ntimes = len(times)
    num_tps = ntimes - lag
    df = pd.DataFrame(index=times[0:num_tps])
    # df = pd.DataFrame(index=times)
    for idx, ts in enumerate(ts_arr):
        data = ts.data
        t_data = np.vstack(data)
        for tlag in range(0, lag + 1, 1):
            # t_data = np.roll(t_data, tlag)  # wrong but somehow predictive...
            # df[f'{idx}_{tlag}'] = t_data
            if lag == tlag:
                df[f"{idx}_{tlag}"] = t_data[tlag:]
            else:
                df[f"{idx}_{tlag}"] = t_data[tlag: -(lag - tlag)]

    return df


def get_lagged_ts(ts1, ts2=None, lag=0):
    if ts2 is None:
        ts2 = ts1
    if lag > 0:
        ts1 = ts1[: -np.abs(lag)]
        ts2 = ts2[np.abs(lag):]
    elif lag < 0:
        ts1 = ts1[np.abs(lag):]
        ts2 = ts2[: -np.abs(lag)]

    return ts1, ts2


def get_lagged_ts_arr(ts1_arr, ts2_arr=None, lag=0):
    if ts2_arr is None:
        ts2_arr = ts1_arr
    if lag > 0:
        ts1_arr = ts1_arr[:, : -np.abs(lag)]
        ts2_arr = ts2_arr[:, np.abs(lag):]
    elif lag < 0:
        ts1_arr = ts1_arr[:, np.abs(lag):]
        ts2_arr = ts2_arr[:, : -np.abs(lag)]

    return ts1_arr, ts2_arr


def lead_lag_corr(ts1, ts2, maxlags=20, corr_method="spearman", cutoff=1, cutoff_ts=1):
    reload(gut)
    reload(flt)
    ts1, ts2 = equalize_time_points(ts1, ts2)

    Nx = len(ts1)
    if Nx != len(ts2):
        raise ValueError("ts1 and ts2 must be equal length")
    nts1 = sut.standardize(ts1)
    nts2 = sut.standardize(ts2)

    if cutoff != 1:
        nts1 = flt.apply_butter_filter(ts=nts1, cutoff=cutoff)
        nts2 = flt.apply_butter_filter(ts=nts2, cutoff=cutoff)

    corr_range = []
    p_val_arr = []

    if corr_method == "spearman":
        corr_func = st.stats.spearmanr
    elif corr_method == "pearson":
        corr_func = st.stats.pearsonr
    tau_arr = np.arange(-maxlags, maxlags + 1, 1)
    for lag in tau_arr:
        ts1_lag, ts2_lag = get_lagged_ts(ts1=nts1, ts2=nts2, lag=lag)
        corr, p_val = corr_func(ts1_lag, ts2_lag)
        corr_range.append(corr)
        p_val_arr.append(p_val)
    corr_range = np.array(corr_range)
    # if cutoff > 1: corr_range = flt.apply_butter_filter(corr_range, cutoff=cutoff)

    min_max_dict = gut.find_local_min_max_xy(data=corr_range, x=tau_arr)

    # abs_max_dict = gut.find_local_max_xy(data=np.abs(corr_range), x=tau_arr)

    ll_dict = {
        "corr": corr_range,
        "p_val": np.array(p_val_arr),
        "tau": tau_arr,
        "tau_max": min_max_dict["x_max"],
        "max": min_max_dict["max"],
        "all_max": min_max_dict["local_maxima"],
        "all_tau_max": min_max_dict["x_local_max"],
        "tau_min": min_max_dict["x_min"],
        "min": min_max_dict["min"],
        "all_min": min_max_dict["local_minima"],
        "all_tau_min": min_max_dict["x_local_min"],
    }

    return ll_dict


def create_xr_tp(tp, name="time"):
    if isinstance(tp, (list, np.ndarray)):
        raise ValueError("ERROR! tp has to be a single time point!")
    if isinstance(tp, str):
        tp = str2datetime(tp)
        return tp
    else:
        return xr.DataArray(data=tp, coords={f"{name}": tp})  # [0]


def create_xr_tps(times, name="time"):
    if isinstance(times, (list, np.ndarray)):
        if isinstance(times[0], str):
            tps = []
            for tp in times:
                tps.append(create_xr_tp(tp))
            times = xr.concat(tps, dim=name)
    else:
        # single time point
        times = create_xr_tp(times)

    return times


def create_xr_ts(times, data=None, name="time"):

    times = create_xr_tps(times, name=name)
    if gut.is_single_tp(tps=times):
        times = [times.data]
    if data is None:
        data = times
    if len(times) != len(data):
        raise ValueError("ERROR! Times and data must have same length!")

    return xr.DataArray(data=data, dims=[name], coords={f"{name}": times})


def create_zero_ts(times):
    return create_xr_ts(data=np.zeros(len(times)), times=times)


def set_tps_val(ts, tps, val, replace=True):
    times_ts = ts.time

    for x in tps.data:
        if x not in times_ts:
            gut.myprint(f"WARNING! tp not in ds: {x}")
    if replace:
        ts.loc[dict(time=tps)] = val
    else:
        ts.loc[dict(time=tps)] += val
    return ts


def get_time_derivative(ts, dx=1):
    dtdx = np.gradient(ts.data, dx)

    return gut.create_xr_ds(
        data=dtdx, dims=ts.dims, coords=ts.coords, name=f"{ts.name}/dt"
    )


def select_time_snippets(ds, time_snippets):
    """Cut time snippets from dataset and concatenate them.

    Args:
        ds (xr.Dataset): Dataset to snip. time_snippets (np.array): Array of n time
        snippets
            with dimension (n,2).

    Returns:
        (xr.Dataset): Dataset with concatenate times
    """
    ds_lst = []
    for time_range in time_snippets:
        # ds_lst.append(ds.sel(time=slice(time_range[0], time_range[1])))
        ds_lst.append(get_time_range_data(
            ds=ds, time_range=time_range, verbose=False))

    ds_snip = xr.concat(ds_lst, dim="time")

    return ds_snip


def convert_datetime64_to_datetime(usert: np.datetime64) -> datetime.datetime:
    t = np.datetime64(usert, "us").astype(datetime.datetime)
    return t


def fill_time_series_val(
    ts: xr.DataArray,
    start_month: str = "Jan",
    end_month: str = "Dec",
    freq: str = "D",
    fill_value: float = 0,
) -> xr.DataArray:
    """
    Adds zeros to a time series in an xarray dataarray object.

    Args:
        ts (xarray.DataArray): The input time series as an xarray DataArray object with
        dimensions (time, lon, lat). start_month (str): The starting month for the time
        series. Defaults to 'Jan'. end_month (str): The ending month for the time series.
        Defaults to 'Dec'. freq (str): The frequency of the time series. Defaults to 'D'
        for daily. fill_value (float): The value to use for filling any missing data in
        the time series. Defaults to 0.

    Returns:
        xarray.DataArray: The padded time series as an xarray DataArray object with
        dimensions (time, lon, lat).

    Raises:
        ValueError: If the input time series does not have a time dimension.

    Examples:
        import xarray as xr

        # create example dataarray with monthly data time =
        pd.date_range(start='2000-01-01', end='2001-12-31', freq='M') lon = [-100, -90,
        -80, -70] lat = [30, 40, 50] data = np.random.rand(len(time), len(lon), len(lat))
        da = xr.DataArray(data, coords={'time': time, 'lon': lon, 'lat': lat},
        dims=('time', 'lon', 'lat'))

        # add padding to time series padded_da = fill_time_series_val(da,
        start_month='Apr', end_month='Nov', freq='M', fill_value=0)
    """
    assert_has_time_dimension(da=ts)
    sy, ey, _ = get_time_range_years(dataarray=ts)
    sm = month2str(start_month)
    em = month2str(end_month)

    sdate = str2datetime(f"{sy}-{sm}-01")
    edate = str2datetime(f"{ey}-{em}-31")
    all_dates = get_dates_of_time_range([sdate, edate], freq=freq)
    padded_ts = ts.reindex(time=all_dates, fill_value=fill_value)

    padded_ts = get_month_range_data(
        dataset=padded_ts, start_month=start_month, end_month=end_month
    )

    return padded_ts


def get_values_by_year(dataarray, years):
    """Return all values of the input dataarray that belong to one of the specified years.

    Args:
        dataarray (xarray.DataArray): Input dataarray with a time dimension. years (list
        or array-like): List or array of years to select.

    Returns:
        xarray.DataArray: New dataarray with only the values belonging to the specified
        years.
    """
    assert_has_time_dimension(dataarray)
    if isinstance(years, xr.DataArray):
        years = years.time.dt.year
    # Extract years from dataarray time dimension
    years_in_data = xr.DataArray(dataarray.time.dt.year, dims=["time"])

    # Select values that are in the specified years
    mask = years_in_data.isin(years)
    data_selected = dataarray.where(mask, drop=True)

    return data_selected


def count_time_points(
    time_points,
    freq="Y",
    start_year=None,
    end_year=None,
):
    """
    Returns a time series that gives per year (or per month in the year) the number of
    time points.

    Parameters
    ----------
    time_points : xarray.DataArray
        An xarray DataArray containing a time dimension of multiple time points.
    freq : str, optional
        The frequency of the output counts. Valid options are 'Y' (per year) and 'M' (per
        month in the year). Default is 'Y'.

    Returns
    -------
    xarray.DataArray
        An xarray DataArray with a time dimension of type np.datetime64 and the number of
        time points as data. The time dimension is continuous from the earliest time point
        in the input array to the last time point of the input array.
    """
    assert_has_time_dimension(time_points)

    # Get start and end time points
    if start_year is not None and end_year is not None:
        start_time = str2datetime(f"{start_year}-01-01")
        end_time = str2datetime(f"{end_year}-12-31")
    else:
        start_time, end_time = get_start_end_date(data=time_points)

    # Compute the number of years or months between start and end time points
    if freq == "Y":
        time_range = pd.date_range(
            start=start_time.values, end=end_time.values, freq="YS"
        )
    elif freq == "M":
        time_range = pd.date_range(
            start=start_time.values, end=end_time.values, freq="MS"
        )
    else:
        raise ValueError(
            f"Invalid {freq}. Valid options are 'Y' and 'M'."
        )

    # Count the number of time points per year or month
    counts = []
    for t in time_range:
        if freq == "Y":
            count = np.count_nonzero(
                (time_points.time.dt.year == t.year).values)
        else:
            count = np.count_nonzero(
                (time_points.time.dt.year == t.year)
                & (time_points.time.dt.month == t.month).values
            )
        counts.append(count)

    # Create output xarray DataArray
    output = xr.DataArray(counts, dims=("time",), coords={"time": time_range})
    return output


def sort_time_points_by_year(tps, val=None, q=None):

    yearly_tps = count_time_points(time_points=tps, freq="Y")

    separate_year_arr = sut.get_values_above_val(
        dataarray=yearly_tps, val=val, q=q)
    a_ys = separate_year_arr["above"].time.dt.year
    b_ys = separate_year_arr["below"].time.dt.year

    a_tps = get_values_by_year(tps, a_ys)
    b_tps = get_values_by_year(tps, b_ys)

    separate_arr = dict(above=a_tps, below=b_tps)
    return separate_arr


def get_time_count_number(tps, counter="week"):
    assert_has_time_dimension(tps)
    times = tps.time
    if counter == "year":
        counts = times.dt.isocalendar().year
    elif counter == "month":
        counts = times.dt.month
    elif counter == "week":
        counts = times.dt.isocalendar().week
    elif counter == "day":
        counts = times.dt.day
    elif counter == "hour":
        counts = times.dt.hour
    else:
        raise ValueError(f"The counter {counter} does not exist!")
    return counts


def get_week_dates(weeks):
    """
    Given an array of week numbers, return a list of tuples, where each tuple contains the
    first calendar day of the corresponding week and the month as a string.

    Parameters
    ----------
    weeks : list of int
        An array of week numbers (1-52)

    Returns
    -------
    list of tuple
        A list of tuples, where each tuple contains a datetime object representing the
        first calendar day of the corresponding week, and a string representing the month
        (e.g., "January", "February", etc.)

    """
    year = 1000
    start_date = datetime.datetime(year, 1, 1)
    week_dates = []
    for week in weeks:
        week_start = start_date + datetime.timedelta(days=(week - 1) * 7)
        day_of_month = week_start.day
        month_num = week_start.month
        month_name = get_month_name(month_number=month_num)
        week_dates.append(f"{day_of_month}\n{month_name}")
    return week_dates


def daily_max(da):
    """
    Compute the daily maximum for each location in an xarray DataArray of hourly data.

    Parameters
    ----------
    da : xarray.DataArray
        An xarray DataArray containing hourly data.

    Returns
    -------
    daily_max : xarray.DataArray
        An xarray DataArray containing the daily maximum for each location. Only days with
        values are included and NaNs at the end are excluded.
    """
    # Resample the data to daily frequency, taking the maximum value for each day
    da_daily = da.resample(time="1D").max(keep_attrs=True)

    # Exclude NaNs at the end
    da_daily = da_daily.dropna(dim="time", how="all")

    return da_daily


def parse_AR_date_string(date_string):
    """
    Parses a date string in the format 'yyyy-mm-dd hh:mm:ss_x' and returns a
    numpy.datetime64 object representing the date and time.

    Parameters
    ----------
    date_string : str
        A date string in the format 'yyyy-mm-dd hh:mm:ss_x'.

    Returns
    -------
    numpy.datetime64
        A numpy.datetime64 object representing the date and time in the input string.

    """
    date, _ = date_string.split("_")  # discard the '_x' part
    return np.datetime64(date)


def convert_AR_xarray_to_timepoints(da):
    """
    Converts an xarray dataarray containing dates in the format 'yyyy-mm-dd hh:mm:ss_x' to
    an xarray dataarray of time points.

    Parameters
    ----------
    da : xarray.core.dataarray.DataArray
        An xarray dataarray containing dates in the format 'yyyy-mm-dd hh:mm:ss_x'.

    Returns
    -------
    xarray.core.dataarray.DataArray
        An xarray dataarray of time points.

    """
    # Parse date strings using parse_date_string function
    dates = np.array([parse_AR_date_string(ds) for ds in da.values])

    # Create an xarray dataarray of time points
    time_coords = xr.DataArray(dates, dims="time", coords={"time": dates})
    return time_coords


def check_AR_strings(da):
    """
    Checks if all strings in a given xarray object match the format 'X-Y-Z A:B:C_D' where
    X, Y, Z, A, B, C and D can be any integer.

    Args:
        da (xarray.DataArray): The input xarray object with the data array to check.

    Returns:
        bool: True if all strings match the format, False otherwise.
    """
    arr = da.values.flatten().astype(str)
    for s in arr:
        if not check_AR_string_format(s):
            return False
    return True


def check_AR_string_format(s):
    """
    Checks if a given string matches the format 'X-Y-Z A:B:C_D' where X, Y, Z, A, B, C and
    D can be any integer.

    Args:
        s (str): The input string to check.

    Returns:
        bool: True if the string matches the format, False otherwise.
    """
    pattern = r"^-?\d+-\d+-?\d* \d+:\d+:\d+_\d+$"
    return bool(re.match(pattern, s))


def filter_nan_values(dataarray, dims=["lon", "lat"], th=1.0):
    """
    Takes an xarray DataArray with dimensions (time, lon, lat) and removes time points
    where the number of NaN values exceeds th.

    Parameters: da (xarray.DataArray): The input DataArray with dimensions (time, lon,
    lat).

    Returns: xarray.DataArray: A new DataArray without time points where the number of NaN
    values exceeds 100.
    """
    # Count number of non-NaN values along the time dimension
    non_nan_counts = dataarray.count(dim=dims)

    # Boolean index to select time points with at least th non-NaN values
    time_filter = non_nan_counts >= th

    rem_frac = 100 - np.count_nonzero(time_filter) / len(time_filter) * 100
    gut.myprint(
        f"Remove {rem_frac:.1f}% of points with < {th} non-nan values!"
    )
    # Filter the time dimension using boolean indexing
    filtered_dataarray = dataarray.isel(time=time_filter)

    return filtered_dataarray


def get_dates_later_as(times, date):
    if isinstance(date, str):
        date = str2datetime(date)
    elif isinstance(date, xr.DataArray):
        date = date.time

    sd, ed = get_start_end_date(data=times)

    new_times = get_time_range_data(ds=times, time_range=[date, ed])

    return new_times


def are_same_time_points(dataarray1, dataarray2):
    """
    Check if two xarray DataArrays have exactly the same time points.

    Parameters:
        dataarray1 (xarray.DataArray): The first xarray DataArray. dataarray2
        (xarray.DataArray): The second xarray DataArray.

    Returns:
        bool: True if both DataArrays have the same time points, False otherwise.
    """
    assert_has_time_dimension(dataarray1)
    assert_has_time_dimension(dataarray2)
    return set(dataarray1.time.values) == set(dataarray2.time.values)


def equalize_time_points(ts1, ts2, verbose=True):
    """Equalize the time points of two xarray dataarrays.

    Args:
        ts1 (xr.Dataarray): dataarray or dataset ts2 (xr.Dataarray): dataarray or dataset

    Returns:
        tuple: tuple of two datasets with equal time points
    """
    if are_same_time_points(ts1, ts2):
        return ts1, ts2
    else:
        assert_has_time_dimension(ts1)
        assert_has_time_dimension(ts2)
        time1 = ts1.time.values
        time2 = ts2.time.values
        common_times = np.intersect1d(time1, time2)
        if len(common_times) == 0:
            gut.myprint(f"No common time points between both datasets!")
            return [], []
        else:
            gut.myprint("Equalize time points of both datasets!",
                        verbose=verbose)
            ts1 = get_sel_tps_ds(ts1, tps=ts2.time)
            ts2 = get_sel_tps_ds(ts2, tps=ts1.time)

    return ts1, ts2


def get_intersect_tps(tps1, tps2):
    tps, _ = equalize_time_points(ts1=tps1, ts2=tps2, verbose=False)
    if len(tps) == 0:
        return []
    return tps.time


def check_time_overlap(da1, da2, return_overlap=False):
    """
    Check if there is any overlap in the time dimension between two xarray dataarrays

    Parameters:
    -----------
    da1 : xarray DataArray
        First input dataarray
    da2 : xarray DataArray
        Second input dataarray

    Returns:
    --------
    bool
        True if there is any overlap in the time dimension between both dataarrays, False
        if not
    """
    assert_has_time_dimension(da1)
    assert_has_time_dimension(da2)
    time1 = set(da1.time.values)
    time2 = set(da2.time.values)
    if return_overlap:
        time1 = da1.time.values
        time2 = da2.time.values
        common_times = np.intersect1d(time1, time2)
        if common_times.size > 0:
            return True, common_times
        else:
            return False, np.array([])
    else:
        return bool(time1.intersection(time2))


def select_random_timepoints(dataarray, sample_size=1, seed=None):
    """
    Returns a random sample of time points from an xarray dataarray along the time
    dimension. The time points are sorted in time.

    Parameters:
        dataarray (xarray.DataArray): The input dataarray. sample_size (int): The number
        of time points to sample.

    Returns:
        xarray.DataArray: A DataArray containing a random sample of time points, sorted in
        time.
    """
    time_vals = dataarray.time.values
    if seed is not None:
        np.random.seed(seed)
    random_sample = np.random.choice(
        time_vals, size=sample_size, replace=False)
    random_sample.sort()
    return dataarray.sel(time=random_sample)


def sliding_window_mean(da, length):
    """
    Compute the sliding window mean for an xarray DataArray where each value represents
    the average of the previous l time steps.

    Parameters: da (xarray.DataArray): The input DataArray. length (int): The length of
    the sliding window.

    Returns: xarray.DataArray: The DataArray with sliding window means.
    """
    # Initialize an empty array to store the results
    result = np.zeros_like(da)

    # Iterate through each time step
    for t in range(len(da)):
        # Calculate the start and end indices for the window
        start = max(0, t - length + 1)
        end = t + 1

        # Slice the data for the current window
        window_data = da[start:end]

        # Compute the mean for the window and assign it to the result array
        result[t] = window_data.mean()

    # Create a new DataArray with the computed means
    sliding_mean = xr.DataArray(
        result, coords=da.coords, dims=da.dims, attrs=da.attrs)

    return sliding_mean
