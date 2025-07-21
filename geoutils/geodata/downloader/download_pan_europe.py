
import cdsapi
import sys
import geoutils.utils.general_utils as gut
import geoutils.utils.file_utils as fut
import geoutils.utils.time_utils as tu
import geoutils.preprocessing.open_nc_file as onf
import os
import numpy as np
import argparse
from importlib import reload
reload(tu)
reload(fut)
reload(onf)


def pan_europe_request(variable, years, months, days, times, **kwargs):
    if not isinstance(variable, list):
        variable = [variable]
    if not isinstance(years, (list, np.ndarray)):
        if not isinstance(years, (float, int)):
            gut.myprint(f"Invalid year format: {years}")
            raise ValueError(
                f"Invalid year format: {years}. Must be a (list of) number.")
        years = [years]

    gcm = kwargs.pop('gcm', None)
    ssp = kwargs.pop('ssp', None)
    technology = kwargs.pop('technology', None)
    if ssp is None and gcm != 'era5':
        raise ValueError("SSP must be provided for pan_europe_request.")

    if years[-1] > 2015:
        temporal_period = 'future_projections'
    else:
        temporal_period = 'historical'

    if len(months) > 1:
        months = ['12']

    request = {
        "pecd_version": "pecd4_2",
        "temporal_period": [temporal_period],
        "origin": [gcm],
        "emission_scenario": [ssp],
        'variable': [variable],
        'technology': [str(technology)],
        'year': years,
        'month': months,
        "spatial_resolution": ["0_25_degree"],
    }

    return request


def get_dataset(request):
    return 'sis-energy-pecd'


def get_filename(variable, gcm, ssp, start_month, end_month,
                 timestr, years,
                 **kwargs):
    if isinstance(years, (list, np.ndarray)):
        year = f'{years[0]}_{years[-1]}'
    else:
        year = f'{years}'
    dir = f'{variable}/{gcm}/{ssp}/'
    if start_month != 'Jan' or end_month != 'Dec':
        prefix = f'{variable}_{start_month}_{end_month}'
    else:
        prefix = f'{variable}_{gcm}_{ssp}'
    return f'{dir}/{prefix}_{timestr}_{year}.nc'
