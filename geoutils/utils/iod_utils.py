''' File description

@Author  :   Jakob Schlör
@Time    :   2022/08/02 19:22:51
@Contact :   jakob.schloer@uni-tuebingen.de
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
# IOD specific functions
# ======================================================================================


def get_iod_index(ssta, monthly=False, time_range=None):
    """Returns the time series of the Nino 1+2, 3, 3.4, 4 from SSTA dataset.

    Args:
        ssta (xr.dataarray): Sea surface temperature anomalies.
        monthly (boolean): Averages time dimensions to monthly.
            Default to False.
        time_range(list, optional): Select Nino indices only in a given time-range.
            Defauts to None

    Returns:
        iod_index (xr.Dataset): Nino indices.
    """
    da = ssta.copy()
    box_west, box_west_std = tut.get_mean_time_series(
        da, lon_range=[50, 70],
        lat_range=[-10, 10],
        time_roll=0
    )
    box_west.name = 'west'
    box_east, box_east_std = tut.get_mean_time_series(
        da, lon_range=[90, 110],
        lat_range=[-10, 0],
        time_roll=0
    )
    box_east.name = 'east'

    iod = box_west - box_east
    iod.name = 'dmi'  # Dipole Mode Index

    iod_idx = xr.merge([iod, box_west, box_east])

    if monthly:
        # iod_idx = iod_idx.resample(time='M', label='left').mean()
        iod_idx = tut.apply_timemean(iod_idx, timemean='month')
        iod_idx = iod_idx.assign_coords(
            dict(time=iod_idx['time'].data + np.timedelta64(1, 'D'))
        )

    if time_range is not None:
        # iod_idx = iod_idx.sel(time=slice(np.datetime64(time_range[0], "M"),
        #                                    np.datetime64(time_range[1], "M")))
        iod_idx = tut.get_sel_time_range(
            ds=iod_idx, time_range=time_range, freq='M',
            verbose=False)

    return iod_idx


def get_iod_strength(enso_val=0):
    strength = 'Normal'
    if enso_val < -0.5:
        strength = 'Weak_nIOD'
    if enso_val <= -1:
        strength = 'Moderate_nIOD'
    if enso_val <= -1.5:
        strength = 'Strong_nIOD'
    if enso_val > 0.5:
        strength = 'Weak_pIOD'
    if enso_val >= 1:
        strength = 'Moderate_pIOD'
    if enso_val >= 1.5:
        strength = 'Strong_pIOD'
    if enso_val > 2:
        strength = 'Very_Strong_pIOD'

    return strength


def get_iod_flavors(iod_index,
                    month_range=['Dec', 'Feb'],
                    mean=True, threshold=0.5,
                    drop_volcano_year=False):
    """Get IOD flavors.

    Parameters:
    -----------
        min_diff (float): min_diff between nino3 and nino4 to get only the
                            extreme EP or CP
        threshold (float, str): Threshold to define winter as El Nino or La nIOD,
                                A float or 'std' are possible.
                                Default: 0.5.
    """

    if threshold == 'std':
        threshold_iod = float(iod_index['dmi'].std(skipna=True))
    else:
        threshold_iod = float(threshold)

    # Identify El Nino and La nIOD types
    iod_classes = []
    sd, ed = tut.get_start_end_date(data=iod_index)
    if tut.is_datetime360(iod_index.time.data[0]):
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
        y = yr.year if tut.is_datetime360(iod_index.time.data[0]) else yr
        y_end = y+1 if em < sm else y
        if tut.is_datetime360(iod_index.time.data[0]):
            time_range = [cftime.Datetime360Day(y, sm, 1),
                          cftime.Datetime360Day(y_end, em+1, 1)]
        else:
            time_range = [np.datetime64(f"{y}-{sm:02d}-01", "D"),
                          np.datetime64(f"{y_end}-{em+1:02d}-01", "D")-1]

        # Select time window
        iod = tut.get_sel_time_range(
            ds=iod_index['dmi'], time_range=time_range, verbose=False,
            freq='M')

        # Choose mean or min
        if mean:
            iod = iod.mean(dim='time', skipna=True)
        else:
            iod = iod.min(dim='time', skipna=True)

        buff_dic = {'start': time_range[0], 'end': time_range[1],
                    'dmi': float(iod)}
        buff_dic['strength'] = get_iod_strength(iod)

        # pIOD
        if (iod.data >= threshold_iod):
            buff_dic['type'] = 'pIOD'
        # nIOD years
        elif (iod.data <= -threshold_iod):
            buff_dic['type'] = 'nIOD'
        # standard years
        else:
            buff_dic['type'] = 'Normal'

        iod_classes.append(buff_dic)

    iod_classes = pd.DataFrame(iod_classes)

    # Years of strong volcanic erruptions followed by an El Nino
    if drop_volcano_year:
        volcano_years_idx = iod_classes.loc[
            (iod_classes['start'] == '1955-12-01') |
            (iod_classes['start'] == '1956-12-01') |
            (iod_classes['start'] == '1957-12-01') |
            (iod_classes['start'] == '1963-12-01') |
            (iod_classes['start'] == '1980-12-01') |
            (iod_classes['start'] == '1982-12-01') |
            (iod_classes['start'] == '1991-12-01')
        ].index
        iod_classes = iod_classes.drop(index=volcano_years_idx)

    return iod_classes


def get_iod_flavors_obs(definition='box',
                        fname=None,
                        ssta=None,
                        vname='sst',
                        climatology='month',
                        month_range=[12, 2],
                        time_range=None,
                        ):
    """Classifies given month range into ENSO flavors.

    Args:
        definition (str, optional): Definition used for classification.
            Defaults to 'N3N4'.
        fname (str, optional): Each definition might require information of other
            datasets, i.e.:
                'N3N4' requires the global SST dataset.
                'EC' requires the global SST dataset for the EOF analysis.
                'N3N4_NOAA' requires the nino-indices by NOAA
                'Cons' requires a table of classifications.
            Defaults to None which uses the preset paths.
        vname (str): Varname of SST only required for 'N3N4' and 'EC'. Defaults to 'sst'.
        land_area_mask (xr.Dataarray): Land area fraction mask, i.e. 0 over oceans.
            Defaults to None.
        climatology (str, optional): Climatology to compute anomalies.
            Only required for 'N3N4' and 'EC'. Defaults to 'month'.
        month_range (list, optional): Month range. Defaults to [11,1].
        time_range (list, optional): Time period of interest.
            Defaults to None.

    Raises:
        ValueError: If wrong definition is defined.

    Returns:
        (pd.Dataframe) Containing the classification including the time-period.
    """
    if definition in ['box', 'EC']:
        if fname is None:
            fname = PATH + "/../../data/era5/sst/sea_surface_temperature_monthly_coarse_1950_2021.nc"
        # Process global SST data
        if ssta is None:
            da_sst = xr.open_dataset(fname)[vname]

            # Check dimensions
            da_sst = gut.check_dimensions(da_sst, sort=True)
            # Detrend data
            da_sst = tut.detrend_dim(da_sst, freq='M')
            # Anomalies
            ssta = tut.compute_anomalies(da_sst, group=climatology)

    if definition == 'box':
        iod_index = get_iod_index(ssta, time_range=time_range)
        iod_classes = get_iod_flavors_obs(
            iod_index,
            month_range=month_range,
            mean=True,
            threshold=0.5,
            drop_volcano_year=False
        )

    return iod_classes


def get_iod_years(iod_classes, season_types, class_type='type'):
    iod_years = []
    for season_type in season_types:
        try:
            iod_years_season = np.array(
                [iod_classes.loc[iod_classes[class_type] == season_type]['start'],
                 iod_classes.loc[iod_classes[class_type] == season_type]['end']]
            ).T
        except:
            raise ValueError("No valid season_type chosen!")

        if len(iod_years_season) == 0:
            gut.myprint(
                f'No Enso years for {class_type} of this season type {season_type}!')

        iod_years.append(iod_years_season)

    return np.concatenate(iod_years, axis=0)
