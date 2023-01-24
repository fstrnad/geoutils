''' File description

@Author  :   Jakob Schlör
@Time    :   2022/08/02 19:22:51
@Contact :   jakob.schloer@uni-tuebingen.de
'''
from importlib import reload
import cftime
import os
import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as stats

import geoutils.utils.general_utils as gut
import geoutils.utils.time_utils as tut
from sklearn.decomposition import PCA
PATH = os.path.dirname(os.path.abspath(__file__))
reload(tut)

# ======================================================================================
# SST process functions
# ======================================================================================


def cut_map(ds, lon_range=None, lat_range=None, shortest=True):
    """Cut an area in the map. Use always smallest range as default.
    It lon ranges accounts for regions (eg. Pacific) that are around the -180/180 region.

    Args:
    ----------
    lon_range: list [min, max]
        range of longitudes
    lat_range: list [min, max]
        range of latitudes
    shortest: boolean
        use shortest range in longitude (eg. -170, 170 range contains all points from
        170-180, -180- -170, not all between -170 and 170). Default is True.
    Return:
    -------
    ds_area: xr.dataset
        Dataset cut to range
    """
    if lon_range is not None:
        if (max(lon_range) - min(lon_range) <= 180) or shortest is False:
            ds = ds.sel(
                lon=slice(np.min(lon_range), np.max(lon_range)),
                lat=slice(np.min(lat_range), np.max(lat_range))
            )
        else:
            # To account for areas that lay at the border of -180 to 180
            ds = ds.sel(
                lon=ds.lon[(ds.lon < min(lon_range)) |
                           (ds.lon > max(lon_range))],
                lat=slice(np.min(lat_range), np.max(lat_range))
            )
    if lat_range is not None:
        ds = ds.sel(
            lat=slice(np.min(lat_range), np.max(lat_range))
        )

    return ds


def get_mean_time_series(da, lon_range, lat_range, time_roll=0):
    """Get mean time series of selected area.

    Parameters:
    -----------
    da: xr.DataArray
        Data
    lon_range: list
        [min, max] of longitudinal range
    lat_range: list
        [min, max] of latiduninal range
    """
    da_area = cut_map(da, lon_range, lat_range)
    ts_mean = da_area.mean(dim=('lon', 'lat'), skipna=True)
    ts_std = da_area.std(dim=('lon', 'lat'), skipna=True)
    if time_roll > 0:
        ts_mean = ts_mean.rolling(time=time_roll, center=True).mean()
        ts_std = ts_std.rolling(time=time_roll, center=True).mean()

    return ts_mean, ts_std


def get_antimeridian_coord(lons):
    """Change of coordinates from normal to antimeridian."""
    lons = np.array(lons)
#    lons_new = np.where(lons < 0, (lons % 180),(lons % 180 - 180))
    lons_new = np.where(lons < 0, (lons + 180), (lons - 180))
    return lons_new


def set_antimeridian2zero(ds, roll=True):
    """Set the antimeridian to zero.

    Easier to work with the pacific then.
    """
    if ds['lon'].data[0] <= -100 and roll is True:
        # Roll data such that the dateline is not at the corner of the dataset
        print("Roll longitudes.")
        ds = ds.roll(lon=(len(ds['lon']) // 2), roll_coords=True)

    # Change lon coordinates
    lons_new = get_antimeridian_coord(ds.lon)
    ds = ds.assign_coords(
        lon=lons_new
    )
    print('Set the dateline to the new longitude zero.')
    return ds


# ======================================================================================
# ENSO specific functions
# ======================================================================================


def get_nino_indices(ssta, monthly=False, time_range=None):
    """Returns the time series of the Nino 1+2, 3, 3.4, 4 from SSTA dataset.

    Args:
        ssta (xr.dataarray): Sea surface temperature anomalies.
        monthly (boolean): Averages time dimensions to monthly.
            Default to False.
        time_range(list, optional): Select Nino indices only in a given time-range.
            Defauts to None

    Returns:
        nino_indices (xr.Dataset): Nino indices.
    """
    da = ssta.copy()
    nino12, nino12_std = get_mean_time_series(
        da, lon_range=[-90, -80],
        lat_range=[-10, 0], time_roll=0
    )
    nino12.name = 'nino12'
    nino3, nino3_std = get_mean_time_series(
        da, lon_range=[-150, -90],
        lat_range=[-5, 5], time_roll=0
    )
    nino3.name = 'nino3'
    nino34, nino34_std = get_mean_time_series(
        da, lon_range=[-170, -120],
        lat_range=[-5, 5], time_roll=0
    )
    nino34.name = 'nino34'
    nino4, nino4_std = get_mean_time_series(
        da, lon_range=[160, -150],
        lat_range=[-5, 5], time_roll=0
    )
    nino4.name = 'nino4'

    nino_idx = xr.merge([nino12, nino3, nino34, nino4])

    if monthly:
        # nino_idx = nino_idx.resample(time='M', label='left').mean()
        nino_idx = tut.apply_timemean(nino_idx, timemean='month')
        nino_idx = nino_idx.assign_coords(
            dict(time=nino_idx['time'].data + np.timedelta64(1, 'D'))
        )

    if time_range is not None:
        # nino_idx = nino_idx.sel(time=slice(np.datetime64(time_range[0], "M"),
        #                                    np.datetime64(time_range[1], "M")))
        nino_idx = tut.get_sel_time_range(
            ds=nino_idx, time_range=time_range, freq='M',
            verbose=False)

    return nino_idx


def get_nino_indices_NOAA(
        fname="https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii",
        time_range=None, time_roll=0, group='month'):
    """Compute Nino indices from NOAA file.

    Args:
        fname ([type]): NOAA Nino calculation. Default to link.
        time_range ([type], optional): [description]. Defaults to None.
        time_roll (int, optional): [description]. Defaults to 3.
        base_period (list, optional): Base period to compute climatology.
            If None whole range is used. Default None.

    Returns:
        [type]: [description]
    """
    df = pd.read_csv(
        fname, skiprows=0, header=0, delim_whitespace=True
    )
    time = []
    for i, row in df.iterrows():
        time.append(np.datetime64(
            '{}-{:02d}'.format(int(row['YR']), int(row['MON'])), 'D'))

    nino_regions = xr.merge([
        xr.DataArray(data=df['NINO1+2'], name='nino12', coords={
                     "time": np.array(time)}, dims=["time"]),
        xr.DataArray(data=df['NINO3'], name='nino3', coords={
                     "time": np.array(time)}, dims=["time"]),
        xr.DataArray(data=df['NINO4'], name='nino4', coords={
                     "time": np.array(time)}, dims=["time"]),
        xr.DataArray(data=df['NINO3.4'], name='nino34', coords={
                     "time": np.array(time)}, dims=["time"]),
    ])

    # Choose 30 year climatology every 5 years

    min_date = np.array(nino_regions.time.data.min(), dtype='datetime64[M]')
    max_date = np.array('2020-12', dtype='datetime64[M]')
    time_steps = np.arange(min_date, max_date,
                           np.timedelta64(60, 'M'))
    nino_anomalies = []
    for ts in time_steps:
        if ts < (min_date + np.timedelta64(15*12, 'M')):
            base_period = np.array(
                [min_date, min_date + np.timedelta64(30*12-1, 'M')])
        elif ts > (max_date - np.timedelta64(15*12, 'M')):
            base_period = np.array(
                [max_date - np.timedelta64(30*12-1, 'M'), max_date])
        else:
            base_period = np.array([
                ts - np.timedelta64(15*12, 'M'),
                ts + np.timedelta64(15*12-1, 'M')
            ])

        buff = tut.compute_anomalies(
            nino_regions, group=group, base_period=base_period,
            verbose=False)
        buff = buff.sel(time=slice(ts, ts+np.timedelta64(5*12-1, 'M')))
        nino_anomalies.append(buff)

    nino_indices = xr.concat(nino_anomalies, dim='time')

    if time_roll > 0:
        nino_indices = nino_indices.rolling(
            time=time_roll, center=True).mean(skipna=True)
    if time_range is not None:
        nino_indices = nino_indices.sel(time=slice(np.datetime64(time_range[0], "M"),
                                                   np.datetime64(time_range[1], "M")))

    return nino_indices


def get_enso_strength(enso_val=0):
    strength = 'Normal'
    if enso_val < -0.5:
        strength = 'Weak_Nina'
    if enso_val <= -1:
        strength = 'Moderate_Nina'
    if enso_val <= -1.5:
        strength = 'Strong_Nina'
    if enso_val > 0.5:
        strength = 'Weak_Nino'
    if enso_val >= 1:
        strength = 'Moderate_Nino'
    if enso_val >= 1.5:
        strength = 'Strong_Nino'
    if enso_val > 2:
        strength = 'Very_Strong_Nino'

    return strength


def get_enso_flavors_N3N4(nino_indices,
                          month_range=['Dec', 'Feb'],
                          mean=True, threshold=0.5,
                          offset=0.0,
                          min_diff=0.0,
                          drop_volcano_year=False):
    """Get nino flavors from Niño‐3–Niño‐4 approach (Kug et al., 2009; Yeh et al.,2009).

    Parameters:
    -----------
        min_diff (float): min_diff between nino3 and nino4 to get only the
                            extreme EP or CP
        threshold (float, str): Threshold to define winter as El Nino or La Nina,
                                A float or 'std' are possible.
                                Default: 0.5.
    """
    if offset > 0.0:
        print("Warning! A new category of El Nino and La Ninas are introduced.")

    if threshold == 'std':
        threshold_nino3 = float(nino_indices['nino3'].std(skipna=True))
        threshold_nino4 = float(nino_indices['nino4'].std(skipna=True))
    else:
        threshold_nino3 = float(threshold)
        threshold_nino4 = float(threshold)

    # Identify El Nino and La Nina types
    enso_classes = []
    sd, ed = tut.get_start_end_date(data=nino_indices)
    if gut.is_datetime360(nino_indices.time.data[0]):
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
        y = yr.year if gut.is_datetime360(nino_indices.time.data[0]) else yr
        y_end = y+1 if em < sm else y
        if gut.is_datetime360(nino_indices.time.data[0]):
            time_range = [cftime.Datetime360Day(y, sm, 1),
                          cftime.Datetime360Day(y_end, em+1, 1)]
        else:
            time_range = [np.datetime64(f"{y}-{sm:02d}-01", "D"),
                          np.datetime64(f"{y_end}-{em+1:02d}-01", "D")-1]

        # Select time window
        nino34 = tut.get_sel_time_range(
            ds=nino_indices['nino34'], time_range=time_range, verbose=False,
            freq='M')
        nino3 = tut.get_sel_time_range(
            ds=nino_indices['nino3'], time_range=time_range, verbose=False,
            freq='M')
        nino4 = tut.get_sel_time_range(
            ds=nino_indices['nino4'], time_range=time_range, verbose=False,
            freq='M')

        # Choose mean or min
        if mean:
            nino34 = nino34.mean(dim='time', skipna=True)
            nino3 = nino3.mean(dim='time', skipna=True)
            nino4 = nino4.mean(dim='time', skipna=True)
        else:
            nino34 = nino34.min(dim='time', skipna=True)
            nino3 = nino3.min(dim='time', skipna=True)
            nino4 = nino4.min(dim='time', skipna=True)

        buff_dic = {'start': time_range[0], 'end': time_range[1],
                    'N34': float(nino34),
                    'N3': float(nino3),
                    'N4': float(nino4)}
        buff_dic['N3-N4'] = nino3.data - nino4.data
        buff_dic['strength'] = get_enso_strength(nino34)

        # El Nino years
        if ((nino3.data >= threshold_nino3) or (nino4.data >= threshold_nino4)):
            buff_dic['type'] = 'Nino'

            Nino_EP_label = 'Nino_EP_weak' if offset > 0 else 'Nino_EP'
            Nino_CP_label = 'Nino_CP_weak' if offset > 0 else 'Nino_CP'

            # EP type if DJF nino3 > 0.5 and nino3 > nino4
            if (nino3.data - min_diff) > nino4.data:
                buff_dic['flavor'] = Nino_EP_label
            # CP type if DJF nino4 > 0.5 and nino3 < nino4
            elif (nino4.data - min_diff) > nino3.data:
                buff_dic['flavor'] = Nino_CP_label

            # Strong El Ninos
            if offset > 0.0:
                if (nino3.data >= threshold_nino3 + offset) and (nino3.data - min_diff) > nino4.data:
                    buff_dic['flavor'] = "Nino_EP_strong"
                elif (nino4.data >= threshold_nino4 + offset) and (nino4.data - min_diff) > nino3.data:
                    buff_dic['flavor'] = 'Nino_CP_strong'

        # La Nina years
        elif ((nino3.data <= -threshold_nino3) or (nino4.data <= -threshold_nino4)):
            buff_dic['type'] = 'Nina'

            Nina_EP_label = 'Nina_EP_weak' if offset > 0 else 'Nina_EP'
            Nina_CP_label = 'Nina_CP_weak' if offset > 0 else 'Nina_CP'

            # EP type if DJF nino3 < -0.5 and nino3 < nino4
            if (nino3.data + min_diff) < nino4.data:
                buff_dic['flavor'] = Nina_EP_label
            # CP type if DJF nino4 < -0.5 and nino3 > nino4
            elif (nino4.data + min_diff) < nino3.data:
                buff_dic['flavor'] = Nina_CP_label

            # Strong La Nina
            if offset > 0.0:
                if (nino3.data <= -threshold_nino3 - offset) and (nino3.data + min_diff) < nino4.data:
                    buff_dic['flavor'] = "Nina_EP_strong"
                elif (nino4.data <= -threshold_nino4 - offset) and (nino4.data + min_diff) < nino3.data:
                    buff_dic['flavor'] = 'Nina_CP_strong'

        # standard years
        else:
            buff_dic['type'] = 'Normal'
            buff_dic['flavor'] = 'Normal'

        enso_classes.append(buff_dic)

    enso_classes = pd.DataFrame(enso_classes)

    # Years of strong volcanic erruptions followed by an El Nino
    if drop_volcano_year:
        volcano_years_idx = enso_classes.loc[
            (enso_classes['start'] == '1955-12-01') |
            (enso_classes['start'] == '1956-12-01') |
            (enso_classes['start'] == '1957-12-01') |
            (enso_classes['start'] == '1963-12-01') |
            (enso_classes['start'] == '1980-12-01') |
            (enso_classes['start'] == '1982-12-01') |
            (enso_classes['start'] == '1991-12-01')
        ].index
        enso_classes = enso_classes.drop(index=volcano_years_idx)

    return enso_classes


def get_enso_flavors_consensus(fname, time_range=None):
    # Read literature ENSO classification
    df_lit = pd.read_csv(
        fname, skiprows=0, header=0,
        skipinitialspace=True
    )
    df_lit['start'] = [pd.Timestamp(t) for t in df_lit['start']]
    df_lit['end'] = [pd.Timestamp(t) for t in df_lit['end']]

    nino_classes = []
    for i, row in df_lit.iterrows():
        if time_range is not None:
            if ((row['start'] < np.array(time_range[0], dtype='datetime64[D]'))
                    or (row['end'] > np.array(time_range[1], dtype='datetime64[D]'))):
                continue

        if row['Cons'] == 'E':
            flavor = 'Nino_EP'
        elif row['Cons'] == 'C':
            flavor = 'Nino_CP'
        else:
            continue
        nino_classes.append(
            {'start': row['start'], 'end': row['end'],
             'type': flavor}
        )

    return pd.DataFrame(nino_classes)


def EC_indices(ssta, pc_sign=[1, 1], time_range=None):
    """E and C indices (Takahashi et al., 2011).

    Args:
        ssta (xr.DataArray): Dataarray of SSTA in the region
            lat=[-10,10] and lon=[120E, 70W].
        pc_sign (list, optional): Sign of principal components which can be switched
            for consistency with e.g. Nino-indices. See ambigouosy of sign of PCA.
            For:
                ERA5 data set to [1,-1].
                CMIP6 models set to[-1,-1]
            Defaults to [1,1].

    Returns:
        e_index (xr.Dataarray)
        c_index (xr.Dataarray)
    """

    # Flatten and remove NaNs
    buff = ssta.stack(z=('lat', 'lon'))
    ids = ~np.isnan(buff.isel(time=0).data)
    X = buff.isel(z=ids)

    # PCA
    pca = PCA(n_components=2)
    pca.fit(X.data)

    # Modes
    ts_modes = []
    for i, comp in enumerate(pca.components_):
        ts = stats.zscore(X.data @ comp, axis=0)
        # Flip sign of mode due to ambiguousy of sign
        ts = pc_sign[i] * ts
        ts_modes.append(
            xr.DataArray(data=ts,
                         name=f'eof{i+1}',
                         coords={"time": X.time},
                         dims=["time"])
        )
    ts_mode = xr.merge(ts_modes)

    # Compute E and C index
    # Changed sign of eof2 due to sign flip of it
    e_index = (ts_mode['eof1'] - ts_mode['eof2']) / np.sqrt(2)
    e_index.name = 'E'

    c_index = (ts_mode['eof1'] + ts_mode['eof2']) / np.sqrt(2)
    c_index.name = 'C'

    # Cut time period
    if time_range is not None:
        e_index = e_index.sel(time=slice(
            np.datetime64(time_range[0], "M"), np.datetime64(
                time_range[1], "M")
        ))
        c_index = c_index.sel(time=slice(
            np.datetime64(time_range[0], "M"), np.datetime64(
                time_range[1], "M")
        ))

    return e_index, c_index


def get_enso_flavor_EC(e_index, c_index, month_range=[12, 2],
                       offset=0.0, mean=True, nino_indices=None):
    """Classify winters into their ENSO flavors based-on the E- and C-index.

    The E- and C-index was introduced by Takahashi et al. (2011).
    The following criterias are used:


    Args:
        e_index (xr.DataArray): E-index.
        c_index (xr.DataArray): C-index.
        month_range (list, optional): Month range where to consider the criteria.
            Defaults to [12,2].
        offset (float, optional): Offset to identify only extremes of the flavors.
            Defaults to 0.0.
        mean (boolean, optional): If True the mean of the range must exceed the threshold.
            Otherwise all months within the range must exceed the threshold.
            Defaults to True.

    Returns:
        enso_classes (pd.DataFrame): Dataframe containing the classification.
    """
    e_threshold = e_index.std(dim='time', skipna=True) + offset
    c_threshold = c_index.std(dim='time', skipna=True) + offset

    years = np.arange(
        np.array(e_index.time.min(), dtype='datetime64[Y]'),
        np.array(e_index.time.max(), dtype='datetime64[Y]')
    )
    enso_classes = []
    for y in years:
        time_range = [np.datetime64(f"{y}-{month_range[0]:02d}-01", "D"),
                      np.datetime64(f"{y+1}-{month_range[1]+1:02d}-01", "D")-1]
        # Either mean or min of DJF must exceed threshold
        e_range = e_index.sel(time=slice(*time_range))
        c_range = c_index.sel(time=slice(*time_range))
        if mean:
            e_range = e_range.mean(dim='time', skipna=True)
            c_range = c_range.mean(dim='time', skipna=True)

        # TODO: Nino indices for pre-selection might be obsolete
        # Preselect EN and LN conditions based on Nino34
        if nino_indices is not None:
            nino34 = nino_indices['nino34'].sel(
                time=slice(time_range[0], time_range[1]))
            nino3 = nino_indices['nino3'].sel(
                time=slice(time_range[0], time_range[1]))
            nino4 = nino_indices['nino4'].sel(
                time=slice(time_range[0], time_range[1]))

            # Normal conditions
            if ((nino34.min() >= -0.5 and nino34.max() <= 0.5)
                or (nino3.min() >= -0.5 and nino3.max() <= 0.5)
                    or (nino4.min() >= -0.5 and nino4.max() <= 0.5)):
                buff_dic = {'start': time_range[0], 'end': time_range[1],
                            'type': 'Normal', 'label': 0}
                enso_classes.append(buff_dic)
                continue

        # EPEN
        if e_range.min() >= e_threshold:
            buff_dic = {'start': time_range[0], 'end': time_range[1],
                        'type': 'Nino_EP', 'label': 1}
        # CPEN
        elif c_range.min() >= c_threshold:
            buff_dic = {'start': time_range[0], 'end': time_range[1],
                        'type': 'Nino_CP', 'label': 2}
        # EPLN
        elif e_range.max() <= -e_threshold:
            buff_dic = {'start': time_range[0], 'end': time_range[1],
                        'type': 'Nina_EP', 'label': 3}
        # CPLN
        elif c_range.max() <= -c_threshold:
            buff_dic = {'start': time_range[0], 'end': time_range[1],
                        'type': 'Nina_CP', 'label': 4}
        # Normal
        else:
            buff_dic = {'start': time_range[0], 'end': time_range[1],
                        'type': 'Normal', 'label': 0}

        enso_classes.append(buff_dic)

    return pd.DataFrame(enso_classes)


def get_enso_flavors_obs(definition='N3N4',
                         fname=None,
                         ssta=None,
                         vname='sst',
                         climatology='month',
                         month_range=[12, 2],
                         time_range=None, offset=0.0):
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
        offset (float, optional): Offset for 'extreme' events.
            Defaults to 0.0.

    Raises:
        ValueError: If wrong definition is defined.

    Returns:
        (pd.Dataframe) Containing the classification including the time-period.
    """
    if definition in ['N3N4', 'EC']:
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

    if definition == 'N3N4':
        nino_indices = get_nino_indices(ssta, time_range=time_range)
        enso_classes = get_enso_flavors_N3N4(
            nino_indices,
            month_range=month_range,
            mean=True,
            threshold=0.5, offset=offset,
            min_diff=0.1, drop_volcano_year=False
        )
    elif definition == 'EC':
        # Cut area for EOFs
        lon_range = [120, -80]
        lat_range = [-10, 10]
        ssta = cut_map(
            ds=ssta, lon_range=lon_range, lat_range=lat_range, shortest=True
        )

        # EC-index based-on EOFs
        e_index, c_index = EC_indices(
            ssta, pc_sign=[1, -1], time_range=time_range
        )
        enso_classes = get_enso_flavor_EC(
            e_index, c_index, month_range=month_range, mean=True,
            offset=offset
        )
    elif definition == 'N3N4_NOAA':
        # N3N4 approach
        if fname is None:
            fname = "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii"
        nino_indices = get_nino_indices_NOAA(
            fname, time_range=time_range, time_roll=0)
        enso_classes = get_enso_flavors_N3N4(
            nino_indices, month_range=month_range, mean=True, threshold=0.5,
            offset=offset, min_diff=0.1, drop_volcano_year=False
        )
    elif definition == 'Cons':
        # Consensus by Capotondi (2020)
        if fname is None:
            fname = PATH + "/../../input/enso_types_lit.dat"
        enso_classes = get_enso_flavors_consensus(
            fname,
            time_range=time_range
        )
    else:
        raise ValueError(f"Specified ENSO definition type {definition} is not defined! "
                         + "The following are defined: 'N3N4', 'Cons', 'EC'")

    return enso_classes


def get_enso_flavors_cmip(fname_sst, vname='ts', land_area_mask=None, climatology='month',
                          definition='N3N4', month_range=[12, 2],
                          time_range=None, offset=0.0, detrend_from=1950):
    """Classifies given month range into ENSO flavors.

    Args:
        fname_sst (str): Path to global SST dataset.
        vname (str): Varname of SST. Defaults to 'ts'.
        land_area_mask (xr.Dataarray): Land area fraction mask, i.e. 0 over oceans.
            Defaults to None.
        climatology (str, optional): Climatology to compute anomalies. Defaults to 'month'.
        definition (str, optional): Definition used for classification.
            Defaults to 'N3N4'.
        month_range (list, optional): Month range. Defaults to [11,1].
        time_range (list, optional): Time period of interest.
            Defaults to None.
        offset (float, optional): Offset for 'extreme' events.
            Defaults to 0.0.

    Raises:
        ValueError: If wrong definition is defined.

    Returns:
        (pd.Dataframe) Containing the classification including the time-period.
    """
    reload(tut)
    # Process global SST data
    da_sst = xr.open_dataset(fname_sst)[vname]
    # Mask only oceans
    if land_area_mask is not None or definition != 'N3N4':
        da_sst = da_sst.where(land_area_mask == 0.0)

    # Check dimensions
    da_sst = gut.check_dimensions(da_sst, sort=True)
    # Detrend data
    da_sst = tut.detrend_dim(da_sst, startyear=detrend_from)
    # Anomalies
    ssta = tut.compute_anomalies(da_sst, group=climatology)
    if definition == 'N3N4':
        nino_indices = get_nino_indices(ssta, time_range=time_range)
        enso_classes = get_enso_flavors_N3N4(
            nino_indices, month_range=month_range, mean=True, threshold=0.5,
            offset=offset, min_diff=0.1, drop_volcano_year=False
        )
    elif definition == 'EC':
        # Cut area for EOFs
        lon_range = [120, -80]
        lat_range = [-10, 10]
        ssta = cut_map(
            ds=ssta, lon_range=lon_range, lat_range=lat_range, shortest=True
        )

        # EC-index based-on EOFs
        e_index, c_index = EC_indices(
            ssta, pc_sign=[-1, -1], time_range=time_range
        )
        enso_classes = get_enso_flavor_EC(
            e_index, c_index, month_range=month_range, mean=True,
            offset=offset
        )
    else:
        raise ValueError(f"Specified ENSO definition type {definition} is not defined! "
                         + "The following are defined: 'N3N4', 'Cons', 'EC'")
    return enso_classes


def get_enso_years(enso_classes, season_types, class_type='type'):
    enso_years = []
    for season_type in season_types:
        try:
            enso_years_season = np.array(
                [enso_classes.loc[enso_classes[class_type] == season_type]['start'],
                 enso_classes.loc[enso_classes[class_type] == season_type]['end']]
            ).T
        except:
            raise ValueError("No valid season_type chosen!")

        if len(enso_years_season) == 0:
            gut.myprint(
                f'No Enso years for {class_type} of this season type {season_type}!')

        enso_years.append(enso_years_season)

    return np.concatenate(enso_years, axis=0)
