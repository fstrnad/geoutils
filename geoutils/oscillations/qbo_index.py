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

import geoutils.utils.time_utils as tut
PATH = os.path.dirname(os.path.abspath(__file__))
reload(tut)


# ======================================================================================
# qbo specific functions
# ======================================================================================


def get_qbo_index(u50, monthly=False, time_range=None):
    """Returns the qbo index based on the 50hPa zonal winds dataset.

    Args:
        u50 (xr.dataarray): Zonal winds anomalies.
        monthly (boolean): Averages time dimensions to monthly.
            Default to False.
        time_range(list, optional): Select Nino indices only in a given time-range.
            Defauts to None

    Returns:
        qbo_index (xr.Dataset): Nino indices.
    """
    da = u50
    box_tropics, box_tropics_std = tut.get_mean_time_series(
        da, lon_range=None,
        lat_range=[-10, 10],
        time_roll=0
    )
    box_tropics.name = 'qbo'

    qbo_idx = box_tropics.to_dataset()

    if monthly:
        # qbo_idx = qbo_idx.resample(time='M', label='left').mean()
        qbo_idx = tut.compute_timemean(qbo_idx, timemean='month')
        qbo_idx = qbo_idx.assign_coords(
            dict(time=qbo_idx['time'].data + np.timedelta64(1, 'D'))
        )

    if time_range is not None:
        # qbo_idx = qbo_idx.sel(time=slice(np.datetime64(time_range[0], "M"),
        #                                    np.datetime64(time_range[1], "M")))
        qbo_idx = tut.get_sel_time_range(
            ds=qbo_idx, time_range=time_range, freq='M',
            verbose=False)

    return qbo_idx


def get_qbo_strength(qbo_val=0):
    strength = 'none'
    if qbo_val < 0:
        strength = 'neg_qbo'
    if qbo_val > 0:
        strength = 'pos_qbo'

    return strength


def get_qbo_flavors(qbo_index,
                    month_range=['Jan', 'Dec'],
                    mean=True):
    """Get qbo flavors.

    Parameters:
    -----------
        min_diff (float): min_diff between nino3 and nino4 to get only the
                            extreme EP or CP
        threshold (float, str): Threshold to define winter as El Nino or La nqbo,
                                A float or 'std' are possible.
                                Default: 0.5.
    """

    qbo_classes = []
    sd, ed = tut.get_start_end_date(data=qbo_index)
    if tut.is_datetime360(qbo_index.time.data[0]):
        times = xr.cftime_range(start=sd,
                                end=ed,
                                freq='Y')
    else:
        times = np.arange(
            np.array(sd, dtype='datetime64[Y]'),
            np.array(ed, dtype='datetime64[Y]')
        )
    for yr in times:
        sm = tut.get_month_number(month_range[0])
        em = tut.get_month_number(month_range[1])
        y = yr.year if tut.is_datetime360(qbo_index.time.data[0]) else yr
        y_end = y+1 if em < sm else y
        if tut.is_datetime360(qbo_index.time.data[0]):
            time_range = [cftime.Datetime360Day(y, sm, 1),
                          cftime.Datetime360Day(y_end, em+1, 1)]
        else:
            time_range = [np.datetime64(f"{y}-{sm:02d}-01", "D"),
                          np.datetime64(f"{y_end}-{em+1:02d}-01", "D")-1]

        # Select time window
        qbo = tut.get_sel_time_range(
            ds=qbo_index['qbo'], time_range=time_range, verbose=False,
            freq='M')

        # Choose mean or min
        if mean:
            qbo = qbo.mean(dim='time', skipna=True)
        else:
            qbo = qbo.min(dim='time', skipna=True)

        buff_dic = {'start': time_range[0], 'end': time_range[1],
                    'dmi': float(qbo)}
        buff_dic['strength'] = get_qbo_strength(qbo)

        qbo_classes.append(buff_dic)

    qbo_classes = pd.DataFrame(qbo_classes)

    return qbo_classes


def get_qbo_flavors_obs(
    u50=None,
    time_range=None,
    ):

    qbo_index = get_qbo_index(u50, time_range=time_range)
    qbo_classes = get_qbo_flavors_obs(
        qbo_index,
        mean=True,
    )

    return qbo_classes


def get_qbo_years(qbo_classes, season_types, class_type='type'):
    qbo_years = []
    for season_type in season_types:
        try:
            qbo_years_season = np.array(
                [qbo_classes.loc[qbo_classes[class_type] == season_type]['start'],
                 qbo_classes.loc[qbo_classes[class_type] == season_type]['end']]
            ).T
        except:
            raise ValueError("No valid season_type chosen!")

        if len(qbo_years_season) == 0:
            gut.myprint(
                f'No Enso years for {class_type} of this season type {season_type}!')

        qbo_years.append(qbo_years_season)

    return np.concatenate(qbo_years, axis=0)
