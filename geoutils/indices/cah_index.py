''' File description

@Author  :   Felix Strnad
'''
import geoutils.utils.statistic_utils as sut
import geoutils.utils.general_utils as gut
import numpy as np
from importlib import reload

import geoutils.tsa.pca.rot_pca as rot_pca
import geoutils.utils.time_utils as tut
reload(tut)


# ======================================================================================
# Compute the  central Asian High (CAH) index as the
# mean Z200 spatially averaged over the region 45-55°N, 60-80°E
# ======================================================================================


def get_cah_index(z200, lev=200, var='z_an_month'):
    """Returns the cahi index based on the 200hPa GPH dataset.

    Args:
        z200 (xr.dataarray): Zonal winds fields.
        monthly (boolean): Averages time dimensions to monthly.
            Default to False.
        time_range(list, optional): Select Nino indices only in a given time-range.
            Defauts to None

    Returns:
        cahi_index (xr.Dataset): CAH indices.
    """
    gut.myprint(f'Computing CAH index with {var} on level {lev}hPa')
    da_low = z200[var].sel(lev=lev)
    box_low, box_low_std = tut.get_mean_time_series(
        da_low,
        lon_range=[50, 70],
        lat_range=[15, 25],
        # lon_range=[40, 90],
        # lat_range=[25, 35],
        time_roll=0
    )
    # bonin high
    da_bh = z200[var].sel(lev=lev)
    box_bh, box_bh_std = tut.get_mean_time_series(
        da_bh,
        lon_range=[110, 140],
        lat_range=[35, 55],
        # lon_range=[40, 90],
        # lat_range=[25, 35],
        time_roll=0,
        q=0.5  # Median Time series
    )

    cahi_idx = box_bh #- box_low
    cahi_idx.name = 'cahi'
    cahi_idx = cahi_idx.to_dataset()
    cahi_idx['std'] = box_low_std
    return cahi_idx


def get_cahi_strength(cahi_val=0):
    strength = 'none'
    if cahi_val < 0:
        strength = 'neg_cahi'
    if cahi_val > 0:
        strength = 'pos_cahi'

    return strength


def cah_pattern(z200):
    reload(sut)

    cah_index = get_cah_index(z200=z200, monthly=False)
    regressed_arr = sut.compute_correlation(
        data_array=z200, t_p=cah_index['cahi'])

    return regressed_arr
