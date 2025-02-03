import re
import cdsapi
import sys
import geoutils.utils.general_utils as gut
import geoutils.utils.file_utils as fut
import geoutils.utils.time_utils as tu
import geoutils.preprocessing.open_nc_file as onf
import os
import numpy as np
from cdo import Cdo
import argparse
from importlib import reload
reload(tu)
reload(fut)
reload(onf)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


dict_era5 = {'t2m': '2m_temperature',
             'sp': 'surface_pressure',
             'pr': 'total_precipitation',
             'msl': 'mean_sea_level_pressure',
             'u10': '10m_u_component_of_wind',
             'v10': '10m_v_component_of_wind',
             'sst': 'sea_surface_temperature',
             'ttr': 'top_net_thermal_radiation',
             'olr': 'top_net_thermal_radiation',
             'z': 'geopotential',
             }
times_all = [
    '00:00', '01:00', '02:00',
    '03:00', '04:00', '05:00',
    '06:00', '07:00', '08:00',
    '09:00', '10:00', '11:00',
    '12:00', '13:00', '14:00',
    '15:00', '16:00', '17:00',
    '18:00', '19:00', '20:00',
    '21:00', '22:00', '23:00',
]
times3h = [
    '00:00',
    '03:00',
    '06:00',
    '09:00',
    '12:00',
    '15:00',
    '18:00',
    '21:00',
]
times6h = [
    '00:00',
    '06:00',
    '12:00',
    '18:00',
]
times12h = [
    '00:00',
    '12:00',
]
times24h = [
    '12:00',
]


def rename_era5(variable):
    if variable in dict_era5:
        return dict_era5[variable]
    else:
        return variable


def download_era5(variable, plevel=None,
                  starty=1959, endy=2022,
                  start_month='Jan', end_month='Dec',
                  start_day=1, end_day=31,
                  times='1h',
                  filename=None,
                  folder='./',
                  daymean=False,
                  run=True,
                  **kwargs):
    """Download ERA5 data for a given variable.

    Args:
        variable (str): Variable to be downloaded.
    """
    tstr = times
    if times == '1h':
        times = times_all
    elif times == '3h':
        times = times3h
    elif times == '6h':
        times = times6h
    elif times == '12h':
        times = times12h
    elif times == '24h':
        times = times24h
    else:
        gut.myprint(f"User-defined time resolution {times}")
        tstr = ''

    variable = rename_era5(variable)
    if plevel == 'all':
        # Default are all pressure levels from 100 -- 1000
        plevel_step = kwargs.pop('plevel_step', 100)
        plevel_start = kwargs.pop('plevel_start', 100)
        plevel_end = kwargs.pop('plevel_end', 1000)
        plevel = np.arange(plevel_start, plevel_end+1, plevel_step)
        spl = False
    else:
        spl = True

    if not run:
        gut.myprint("WARNING! Dry run!", color='red')
        run = False
    else:
        run = True

    years = np.arange(starty, endy+1, 1) # include last year as well
    smonths = tu.get_month_number(start_month)
    emonths = tu.get_month_number(end_month)
    marray = np.arange(smonths, emonths+1, 1)
    months = tu.num2str_list(marray)
    days = np.arange(start_day, end_day+1, 1)
    sdays = tu.num2str_list(days)

    gut.myprint(f"We use the following variables: {variable}")
    gut.myprint(f'Download pressure levels {plevel}...')
    gut.myprint(f"Download years {years}...")
    gut.myprint(f"Download months {months}...")
    gut.myprint(f"Download days {days}...")
    gut.myprint(f"Download times {times}...")

    inFiles = []
    if not spl:
        dir = f'{folder}/multi_pressure_level/{variable}/{plevel}/'
        prefix = f'{variable}_{plevel}_{start_month}_{end_month}'
    else:
        dir = f'{folder}/single_pressure_level/{variable}/'
        if start_month != 'Jan' or end_month != 'Dec':
            prefix = f'{variable}_{start_month}_{end_month}'
        else:
            prefix = f'{variable}'

    fut.create_folder(dir)
    for year in years:
        gut.myprint(
            f"Year {year}, Pressure Level {plevel}, Variable {variable}")
        if tstr != '1h':
            filename = f'{dir}/{prefix}_{tstr}_{year}.nc'
        else:
            filename = f'{dir}/{prefix}_{year}.nc'
        fname_daymean = f'{dir}/{prefix}_{year}_daymean.nc'
        if not fut.exist_file(filename, verbose=True):
            # This runs the ERA5 downloader
            gut.myprint(f"Download file {filename}...")
            if run is True:
                request = {
                    'product_type': ['reanalysis'],
                    'data_format': 'netcdf',
                    'download_format': 'unarchived',
                    'variable': [variable],
                    'year': [str(year)],
                    'month': months,
                    'day': sdays,
                    'time': times,
                }

                if not spl:
                    dataset = 'reanalysis-era5-pressure-levels'
                    request['pressure_level'] = plevel,
                else:
                    dataset = 'reanalysis-era5-single-levels'

                client = cdsapi.Client()
                client.retrieve(dataset, request, filename)

        if daymean:
            cdo = Cdo()    # Parameters
            if not fut.exist_file(fname_daymean, verbose=True):
                if run:
                    gut.myprint(f'Create daymean file {fname_daymean}...')
                    ds = onf.open_nc_file(filename)
                    ds_daymean = tu.compute_timemean(ds=ds, timemean='day')
                    fut.save_ds(ds_daymean,
                                filepath=fname_daymean)
            inFiles.append(fname_daymean)
    if daymean:
        if run:
            fname_daymean_yrange = f'{dir}/{prefix}_{starty}_{endy}_daymean.nc'
            if not fut.exist_file(fname_daymean_yrange, verbose=True):
                gut.myprint(
                    f'Create yearly daymean file {fname_daymean_yrange}...')
                ds = onf.open_nc_file(inFiles)
                fut.save_ds(ds=ds,
                            filepath=fname_daymean_yrange)
