import time
import numpy as np
import copy
from tqdm import tqdm
import geoutils.utils.general_utils as gut
import geoutils.utils.file_utils as fut
import geoutils.utils.time_utils as tu
import geoutils.utils.spatial_utils as sput
import geoutils.utils.met_utils as mut
from datetime import datetime
from importlib import reload
import xarray as xr


def open_nc_file(
        nc_files,
        plevels=None,
        decode_times=True,
        verbose=True,
        var_name=None,
        **kwargs,):
    reload(gut)
    gut.myprint("Start processing data!", verbose=verbose)

    ds = open_ds(
        nc_files=nc_files,
        plevels=plevels,
        decode_times=decode_times,
        **kwargs)

    if var_name is not None:
        ds = ds[var_name]
    ds, dims = check_dimensions(
        ds, ts_days=decode_times, verbose=verbose,
        **kwargs)
    ds = gut.rename_var_era5(ds=ds, verbose=verbose, **kwargs)

    gut.myprint(f"End processing data! Dimensions: {dims}", verbose=verbose)

    return ds


def open_ds(nc_files, plevels=None,
            decode_times=True, **kwargs):
    plevel_name = kwargs.pop('plevel_name', 'lev')

    if plevels is None:
        ds = xr.open_mfdataset(nc_files,
                               decode_times=decode_times,
                               parallel=True,
                               chunks={'time': -1}
                               )
    else:
        ds = xr.open_mfdataset(nc_files, decode_times=decode_times,
                               preprocess=add_dummy_dim,
                               chunks={"time": 1000}
                               )

        ds = ds.rename({'dummy': plevel_name})
        ds[plevel_name] = plevels

    return ds


def add_dummy_dim(xda):
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
    ts_days = kwargs.pop('ts_days', True)
    keep_time = kwargs.pop('keep_time', False)
    freq = kwargs.pop('freq', 'D')
    ds = sput.check_dimensions(ds=ds,
                               ts_days=ts_days,
                               lon360=lon360,
                               sort=sort,
                               keep_time=keep_time,
                               freq=freq,
                               verbose=verbose)
    # Set time series to days
    dims = get_dims(ds=ds)

    if len(dims) > 2:
        if 'time' in dims:
            t360 = check_time(ds, **kwargs)

    return ds, dims


def get_dims(ds=None):

    if isinstance(ds, xr.Dataset):
        # check if xarray version is new
        if xr.__version__ != '2024.6.0':
            dims = list(ds.dims.keys())
        else:
            dims = list(ds.dims)  # new in xarray 2023.06.

    elif isinstance(ds, xr.DataArray):
        dims = ds.dims
    else:
        dtype = type(ds)
        raise ValueError(
            f'ds needs to be of type xr.DataArray but is of type {dtype}!')

    return dims


def check_time(ds, **kwargs):
    """Sets the respective calender!

    Args:
        ds (xr.dataset): dataset

    Returns:
        xr.dataset: dataset
    """
    ts_days = kwargs.pop("ts_days", True)
    if ts_days:
        if not gut.is_datetime360(time=ds.time.data[0]):
            calender360 = False
        else:
            gut.myprint('WARNING: 360 day calender is used!')
            calender360 = True
    else:
        gut.myprint('WARNING! No standard calender is used!')
        calender360 = False
    return calender360
