import re
import time
import cdsapi
import sys
import geoutils.geodata.downloader.download_era5 as d5
import geoutils.geodata.downloader.download_pan_europe as dpe
import geoutils.utils.general_utils as gut
import geoutils.utils.file_utils as fut
import geoutils.utils.time_utils as tu
import geoutils.preprocessing.open_nc_file as onf
import numpy as np
import argparse
from importlib import reload
reload(tu)
reload(fut)
reload(onf)
reload(d5)
reload(dpe)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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


def get_request(dataset, variable, years, months, days, times, **kwargs):
    if dataset == 'era5':
        return d5.era5_request(variable=variable,
                               years=years,
                               months=months,
                               days=days,
                               times=times, **kwargs)
    if dataset == 'pan_europe':
        return dpe.pan_europe_request(variable=variable,
                                      years=years,
                                      months=months,
                                      days=days,
                                      times=times, **kwargs)


def get_dataset(dataset, request):
    if dataset == 'era5':
        return d5.get_dataset(request)
    elif dataset == 'pan_europe':
        return dpe.get_dataset(request)


def get_filename(dataset, variable, start_month, end_month, timestr, years,
                 **kwargs):
    if dataset == 'era5':
        plevel = kwargs.get('pressure_level', None)
        return d5.get_filename(variable=variable,
                               plevel=plevel,
                               start_month=start_month,
                               end_month=end_month, timestr=timestr,
                               years=years, **kwargs)
    elif dataset == 'pan_europe':
        gcm = kwargs.get('gcm', 'era5')
        ssp = kwargs.get('ssp', 'ssp245')
        technology = kwargs.get('technology', None)
        return dpe.get_filename(variable=variable,
                                # gcm=gcm, ssp=ssp,
                                # technology=technology,
                                start_month=start_month,
                                end_month=end_month, timestr=timestr,
                                years=years,
                                **kwargs)


def download_copernicus(variable,
                        starty=1959, endy=2022,
                        start_month='Jan', end_month='Dec',
                        start_day=1, end_day=31,
                        times='1h',
                        filename=None,
                        run=True,
                        dataset='era5',
                        folder='./',
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
    elif times == 'monthly':
        tstr = 'monthly'
    else:
        gut.myprint(f"User-defined time resolution {times}")
        tstr = ''

    years = np.arange(starty, endy+1, 1)  # include last year as well
    smonths = tu.get_month_number(start_month)
    emonths = tu.get_month_number(end_month)
    marray = np.arange(smonths, emonths+1, 1)
    months = tu.num2str_list(marray)
    days = np.arange(start_day, end_day+1, 1)
    sdays = tu.num2str_list(days)
    years = gut.arr2list(years)
    gut.myprint(f"We use the following variables: {variable}")
    gut.myprint(f"Download years {years}...")
    gut.myprint(f"Download months {months}...")
    gut.myprint(f"Download days {days}...")
    gut.myprint(f"Download times {times}...")

    fut.create_folder(folder)
    gut.myprint(f'Download to folder {folder}...')

    files = []
    for year in years:

        gut.myprint(f'Processing file for {year}')
        if filename is None:
            filename = get_filename(dataset, variable,
                                    start_month, end_month,
                                    timestr=tstr, years=year, **kwargs)
        savepath = f'{folder}/{filename}'

        if not fut.exist_file(savepath, verbose=True):

            fut.create_folder(savepath)

            # This runs the ERA5 downloader
            if run:
                gut.myprint(f"Download file {filename}...")

                request = get_request(dataset=dataset,
                                      variable=variable,
                                      years=year, months=months,
                                      days=sdays,
                                      times=times, **kwargs)
                dataset = get_dataset(dataset=dataset, request=request)

                print(request, dataset)
                try:
                    client = cdsapi.Client()
                    client.retrieve(dataset, request, savepath)
                except Exception as e:
                    gut.myprint(f"Error during download: {e}", color='red')
            else:
                gut.myprint(f"Dry run: {savepath}")

        files.append(savepath)
    return files
