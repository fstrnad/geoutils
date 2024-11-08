import re
import cdsapi
import sys
import geoutils.utils.general_utils as gut
import geoutils.utils.time_utils as tu
import os
import numpy as np
from cdo import Cdo
import argparse
from importlib import reload
reload(tu)


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


def rename_era5(variable):
    if variable in dict_era5:
        return dict_era5[variable]
    else:
        return variable


def download_era5(variable, plevels=None,
                  starty=1959, endy=2022,
                  start_day=1, end_day=31,
                  start_month='Jan', end_month='Dec',
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
    if times == '1h':
        times = times_all
    elif times == '3h':
        times = times3h
    elif times == '6h':
        times = times6h
    elif times == '12h':
        times = times12h
    else:
        gut.myprint(f"User-defined time resolution {times}")

    variable = rename_era5(variable)
    if plevels is not None:
        if plevels == 'all':
            # Default are all pressure levels from 100 -- 1000
            plevel_step = kwargs.pop('plevel_step', 100)
            plevel_start = kwargs.pop('plevel_start', 100)
            plevel_end = kwargs.pop('plevel_end', 1000)
            plevels = np.arange(plevel_start, plevel_end+1, plevel_step)
        else:
            plevels = np.array(plevels)
        spl = False
    else:
        plevels = ['sfc']
        spl = True

    if not run:
        print("WARNING! Dry run!")
        run = False
    else:
        run = True

    years = np.arange(starty, endy+1, 1)
    smonths = tu.get_month_number(start_month)
    emonths = tu.get_month_number(end_month)
    marray = np.arange(smonths, emonths+1, 1)
    months = tu.num2str_list(marray)
    days = np.arange(start_day, end_day+1, 1)
    sdays = tu.num2str_list(days)

    gut.myprint(f"We use the following variables: {variable}")
    gut.myprint(f'Download pressure levels {plevels}...')
    gut.myprint(f"Download years {years}...")
    gut.myprint(f"Download months {months}...")
    gut.myprint(f"Download days {days}...")

    cdo = Cdo()    # Parameters

    for plevel in plevels:
        inFiles = []
        for year in years:
            gut.myprint(
                f"Year {year}, Pressure Level {plevel}, Variable {variable}")
            if filename is None:
                if not spl:
                    folder += f'/multi_pressure_level/{variable}/{plevel}/'
                    filename = f'{variable}_{year}_{
                        plevel}_{months[0]}_{months[-1]}.nc'
                    fname_daymean = f'{variable}_{year}_{plevel}_{
                        months[0]}_{months[-1]}_daymean.nc'
                else:
                    folder += f'/single_pressure_level/{variable}/'
                    filename = f'{variable}_{year}_{months[0]}_{months[-1]}.nc'
                    fname_daymean = f'{variable}_{year}_{
                        months[0]}_{months[-1]}_daymean.nc'

            if not os.path.exists(folder):
                gut.myprint(f"Make Dir: {folder}")
                os.makedirs(folder)

            filename = folder + filename
            if os.path.exists(filename):
                gut.myprint(f"File {filename} already exists!")
            else:
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
                    client.retrieve(dataset, request, filename).download()

            fname_daymean = folder + fname_daymean
            if daymean:
                if os.path.exists(fname_daymean):
                    print(
                        f"File {fname_daymean} already exists!", flush=True)
                else:
                    if run:
                        cdo.daymean(input=filename, output=fname_daymean)
                        inFiles.append(fname_daymean)
        if daymean:
            if run:
                fname_daymean_yrange = f'{
                    folder}/{variable}_{plevel}_{starty}_{endy}.nc'
                cdo.mergetime(options='-b F32 -f nc',
                              input=inFiles,
                              output=fname_daymean_yrange)
