import cdsapi
import sys
import geoutils.utils.general_utils as gut
import geoutils.utils.time_utils as tu
import os
import numpy as np
from cdo import Cdo
import argparse


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
             }


def rename_era5(variable):
    if variable in dict_era5:
        return dict_era5[variable]
    else:
        return variable


def download_era5(variable, plevels=None,
                  starty=1959, endy=2022,
                  start_day=1, end_day=31,
                  start_month='Jan', end_month='Dec',
                  start_hour=0, end_hour=23,
                  run=True, **kwargs):
    """Download ERA5 data for a given variable.

    Args:
        variable (str): Variable to be downloaded.
    """
    variable = rename_era5(variable)
    gut.myprint(f"We use the following variables: {variable}")
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

    gut.myprint(f'Download pressure levels {plevels}...')

    if not run:
        print("WARNING! Dry run!")
        run = False
    else:
        run = True

    years = np.arange(starty, endy+1, 1)
    smonths = tu.get_index_of_month(month=start_month)
    emonths = tu.get_index_of_month(month=end_month)
    months = np.arange(smonths, emonths+1, 1)
    sdays = np.arange(start_day, end_day+1, 1)

    cdo = Cdo()    # Parameters

    for plevel in plevels:
        inFiles = []
        for year in years:
            gut.myprint(
                f"Year {year}, Pressure Level {plevel}, Variable {variable}")

            if spl is None:
                folder = f'./multi_pressure_level/{variable}/{plevel}/'
                fname_daymean = folder + \
                    f'{variable}_{year}_{plevel}_daymean.nc'
                filename = folder + f'{variable}_{year}_{plevel}.nc'
            else:
                folder = f'./single_pressure_level/{variable}/'
                fname_daymean = folder + f'{variable}_{year}_daymean.nc'
                filename = folder + f'{variable}_{year}.nc'

            if not os.path.exists(folder):
                gut.myprint(f"Make Dir: {folder}")
                os.makedirs(folder)

            if os.path.exists(filename):
                gut.myprint(f"File {filename} already exists!")
            else:
                gut.myprint(f"Download file {filename}...")
                if run is True:
                    c = cdsapi.Client()

                    if not spl:
                        c.retrieve(
                            'reanalysis-era5-pressure-levels',
                            {
                                'product_type': 'reanalysis',
                                'format': 'netcdf',
                                'variable': [variable],
                                'year': [str(year)],
                                'pressure_level': plevel,
                                'month': months,
                                'day': sdays,
                                'time': [
                                    '00:00', '01:00', '02:00',
                                    '03:00', '04:00', '05:00',
                                    '06:00', '07:00', '08:00',
                                    '09:00', '10:00', '11:00',
                                    '12:00', '13:00', '14:00',
                                    '15:00', '16:00', '17:00',
                                    '18:00', '19:00', '20:00',
                                    '21:00', '22:00', '23:00',
                                ],
                            },
                            filename)
                    else:
                        c.retrieve(
                            'reanalysis-era5-single-levels',
                            {
                                'product_type': 'reanalysis',
                                'format': 'netcdf',
                                'variable': [variable],
                                'year': [str(year)],
                                'month': [
                                    '01', '02', '03',
                                    '04', '05', '06',
                                    '07', '08', '09',
                                    '10', '11', '12',
                                ],
                                'day': [
                                    '01', '02', '03',
                                    '04', '05', '06',
                                    '07', '08', '09',
                                    '10', '11', '12',
                                    '13', '14', '15',
                                    '16', '17', '18',
                                    '19', '20', '21',
                                    '22', '23', '24',
                                    '25', '26', '27',
                                    '28', '29', '30',
                                    '31',
                                ],
                                'time': [
                                    '00:00', '01:00', '02:00',
                                    '03:00', '04:00', '05:00',
                                    '06:00', '07:00', '08:00',
                                    '09:00', '10:00', '11:00',
                                    '12:00', '13:00', '14:00',
                                    '15:00', '16:00', '17:00',
                                    '18:00', '19:00', '20:00',
                                    '21:00', '22:00', '23:00',
                                ],
                            },
                            filename)
                    del c
            if run:
                if os.path.exists(fname_daymean):
                    print(f"File {fname_daymean} already exists!", flush=True)
                else:
                    cdo.daymean(input=filename, output=fname_daymean)

            inFiles.append(fname_daymean)
        if run:
            fname_daymean_yrange = folder + \
                f'{variable}_{plevel}_{starty}_{endy}.nc'
            cdo.mergetime(options='-b F32 -f nc',
                          input=inFiles,
                          output=fname_daymean_yrange)
