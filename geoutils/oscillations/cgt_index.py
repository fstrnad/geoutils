''' File description

@Author  :   Felix Strnad
'''
import geoutils.utils.general_utils as gut
from importlib import reload
import cftime
import os
import numpy as np
import pandas as pd
import xarray as xr
import geoutils.tsa.pca.pca as pca

import geoutils.utils.time_utils as tut
PATH = os.path.dirname(os.path.abspath(__file__))
reload(tut)


# ======================================================================================
# Compute the  circumglobal teleconnection index (CGTI) as the
# mean Z200 spatially averaged over the region 35-40°N, 60-70°E
# ======================================================================================


def get_cgt_index(z200, monthly=False, time_range=None):
    """Returns the cgti index based on the 50hPa zonal winds dataset.

    Args:
        z200 (xr.dataarray): Zonal winds fields.
        monthly (boolean): Averages time dimensions to monthly.
            Default to False.
        time_range(list, optional): Select Nino indices only in a given time-range.
            Defauts to None

    Returns:
        cgti_index (xr.Dataset): Nino indices.
    """
    da = z200
    box_tropics, box_tropics_std = tut.get_mean_time_series(
        da, lon_range=[60, 70],
        lat_range=[35, 40],  # Defined as by Ding&Wang 2005
        time_roll=0
    )
    box_tropics.name = 'cgti'

    cgti_idx = box_tropics.to_dataset()

    if monthly:
        cgti_days = cgti_idx.time
        cgti_mm = tut.compute_timemean(cgti_idx, timemean='week')
        cgti_idx = cgti_mm.interp(time=cgti_days)

    return cgti_idx


def get_cgti_strength(cgti_val=0):
    strength = 'none'
    if cgti_val < 0:
        strength = 'neg_cgti'
    if cgti_val > 0:
        strength = 'pos_cgti'

    return strength


def cgt_pattern(z200):
    cgt_index = get_cgt_index(z200=z200, monthly=False)

    rot = 'varimax'
    pca_ = pca.SpatioTemporalPCA(z200,
                                 n_components=10,
                                 rotation=rot)

    return pca_.get_pca_loc_dict(q=None)