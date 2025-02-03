import numpy as np
import time
import geoutils.utils.general_utils as gut
import geoutils.utils.file_utils as fut
import geoutils.utils.time_utils as tu
import geoutils.utils.spatial_utils as sput
from datetime import datetime
from importlib import reload
import xarray as xr
reload(sput)
reload(gut)
reload(fut)
reload(tu)


def open_nc_file(
        nc_files,
        plevels=None,
        decode_times=True,
        verbose=True,
        var_name=None,
        lat_range=None,
        lon_range=None,
        time_range=None,
        month_range=None,
        hours_to_zero=False,
        **kwargs,):
    reload(gut)
    fut.print_file_location_and_size(nc_files, verbose=verbose)

    ds = open_ds(
        nc_files=nc_files,
        plevels=plevels,
        decode_times=decode_times,
        **kwargs)

    ds, dims = check_dimensions(
        ds, datetime_ts=decode_times, verbose=verbose,
        hours_to_zero=hours_to_zero, **kwargs)
    dims = gut.get_dims(ds=ds)
    ds = gut.rename_var_era5(ds=ds, verbose=verbose, **kwargs)
    gut.myprint(f"End processing data! Dimensions: {dims}", verbose=verbose)

    if lat_range is not None or lon_range is not None:
        ds = sput.cut_map(ds=ds,
                          lat_range=lat_range,
                          lon_range=lon_range,
                          verbose=verbose)
    if time_range is not None:
        start_month = month_range[0] if month_range is not None else None
        end_month = month_range[-1] if month_range is not None else None
        ds = tu.get_time_range_data(
            time_range=time_range, ds=ds,
            start_month=start_month, end_month=end_month,
            verbose=verbose)

    if var_name is not None:
        ds = ds[var_name]

    return ds


def open_ds(nc_files, plevels=None,
            decode_times=True, **kwargs):
    plevel_name = kwargs.pop('plevel_name', 'lev')

    if plevels is None:
        ds = my_open_mfdataset(nc_files, decode_times)
    else:
        if not check_mva(files=nc_files):
            ds = open_plevels(nc_files,
                              decode_times=decode_times,
                              plevels=plevels,
                              plevel_name=plevel_name)
        else:
            var_file_dict = get_files_same_vars(files=nc_files)
            ds_array = []
            for var, files in var_file_dict.items():
                if len(files) != len(plevels):
                    raise ValueError(
                        f'Number of files {files} and pressure levels {plevels} do not match!')
                this_ds = open_plevels(files,
                                       decode_times=decode_times,
                                       plevels=plevels,
                                       plevel_name=plevel_name)
                ds_array.append(this_ds)
            ds = xr.merge(ds_array)

    return ds


def open_plevels(nc_files, decode_times, plevels, plevel_name):
    da_arrays = []
    for file in nc_files:
        gut.myprint(f'Open file: {file}')
        da_arrays.append(my_open_mfdataset(nc_files=file,
                                           decode_times=decode_times))
    ds = xr.concat(da_arrays, dim=plevel_name)
    ds.coords[plevel_name] = plevels
    return ds


def my_open_mfdataset(nc_files, decode_times=True, mfdataset=True):
    if isinstance(nc_files, str):
        nc_files = [nc_files]
    if len(nc_files) == 1:
        print(nc_files)
        ds = xr.open_dataset(nc_files[0],
                             decode_times=decode_times,
                             )
    else:
        if mfdataset:
            ds = xr.open_mfdataset(nc_files,
                                   decode_times=decode_times,
                                   )
        else:
            data_array = []
            for file in nc_files:
                print(f'Open file: {file}')
                data_array.append(xr.open_dataarray(file,
                                                    decode_times=decode_times,
                                                    )
                                  )
            ds = xr.merge(data_array)
    return ds


def check_mva(files):
    """Checks whether the variables are the same in the files!

    Args:
        files (list): list of file names
    """
    if isinstance(files, str):
        files = [files]
    var_list = []
    for file in files:
        da = xr.open_dataarray(file)
        var_list.append(da.name)
    vars = np.unique(var_list)
    if len(vars) > 1:
        return True
    elif len(vars) == 1:
        return False
    else:
        raise ValueError('No variable found!')


def get_files_same_vars(files):
    """Returns the files with the same variables!

    Args:
        files (list): list of file names

    Returns:
        list: list of file names
    """
    if isinstance(files, str):
        files = [files]
    files = np.array(files)
    var_list = []
    for file in files:
        da = xr.open_dataarray(file)
        var_list.append(da.name)
    vars, indices = np.unique(var_list, return_inverse=True)
    # Create a dictionary to hold the indices for each unique value
    indices_dict = {val: np.where(indices == i)[
        0] for i, val in enumerate(vars)}
    return {var: files[i] for var, i in indices_dict.items()}


def add_dummy_dim(xda, set_hours_zero=True):
    if set_hours_zero:
        xda = tu.set_hours_to_zero(x=xda)

    time.sleep(0.1)  # To ensure that data is read in correct order!
    xda = xda.expand_dims(dummy=[datetime.now()])
    time.sleep(0.1)
    return xda


def check_dimensions(ds, verbose=True, **kwargs):
    """
    Checks whether the dimensions are the correct ones for xarray!
    """
    reload(sput)
    sort = kwargs.pop('sort', True)
    lon360 = kwargs.pop('lon360', False)
    datetime_ts = kwargs.pop('datetime_ts', True)
    keep_time = kwargs.pop('keep_time', False)
    freq = kwargs.pop('freq', 'D')
    transpose = kwargs.pop('transpose_dims', False)
    ds = sput.check_dimensions(ds=ds,
                               datetime_ts=datetime_ts,
                               lon360=lon360,
                               sort=sort,
                               keep_time=keep_time,
                               freq=freq,
                               transpose_dims=transpose,
                               verbose=verbose)
    # Set time series to days
    dims = gut.get_dims(ds=ds)

    if len(dims) > 2:
        if 'time' in dims:
            t360 = check_time(ds, **kwargs)

    return ds, dims


def check_time(ds, **kwargs):
    """Sets the respective calender!

    Args:
        ds (xr.dataset): dataset

    Returns:
        xr.dataset: dataset
    """
    datetime_ts = kwargs.pop("datetime_ts", True)
    if datetime_ts:
        if not gut.is_datetime360(time=ds.time.data[0]):
            calender360 = False
        else:
            gut.myprint('WARNING: 360 day calender is used!')
            calender360 = True
    else:
        gut.myprint('WARNING! No standard calender is used!')
        calender360 = False
    return calender360
