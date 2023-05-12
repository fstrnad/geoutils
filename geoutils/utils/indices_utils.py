import geoutils.utils.general_utils as gut
import os
import geoutils.tsa.time_series_analysis as tsa
import numpy as np
import pandas as pd
import xarray as xr
import geoutils.utils.time_utils as tu
from importlib import reload


def get_nino_indices(fname, time_range=None, time_roll=3, group='month'):
    """Compute Nino indices from NOAA file.

    Args:
        fname ([type]): [description]
        time_range ([type], optional): [description]. Defaults to None.
        time_roll (int, optional): [description]. Defaults to 3.
        base_period (list, optional): Base period to compute climatology.
            If None whole range is used. Default None.

    Returns:
        [type]: [description]
    """
    df = pd.read_csv(
        fname, skiprows=1, header=0, delim_whitespace=True
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

        buff = tu.compute_anomalies(
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


def get_oni_index(fname, time_range=None):
    """Get ONI index from oni.acii.txt file."""
    df = pd.read_csv(
        fname, skiprows=1, header=0, delim_whitespace=True
    )
    # create time
    df['MON'] = df.index % 12 + 1
    time = []
    for i, row in df.iterrows():
        time.append(np.datetime64(
            '{}-{:02d}'.format(int(row['YR']), int(row['MON'])), 'M'))

    oni = xr.DataArray(data=df['ANOM'], name='oni',
                       coords={"time": np.array(time)}, dims=["time"])

    if time_range is not None:
        oni = oni.sel(time=slice(np.datetime64(time_range[0], "M"),
                                 np.datetime64(time_range[1], "M")))
    return oni


def get_enso_time_snippets(df, flavor='Nino_EP', month_range=[12, 2]):
    """Extract ENSO flavors from file and create year ranges.

    Args:
        df (pd.Dataframe): DF storing a list of ENSO flavors and years.
        flavor (str, optional): Type of ENSO, i.e. Nino_EP, Nino_CP, Nina_CP, Nina_EP. Default to "Nino_EP"
        month_range (list, optional): Month range. Default to [12,2] corresponds to DJF.
    """
    time_period = []
    for t in df.loc[df['type'] == flavor]['year']:
        year = np.datetime64(f'{t}', 'Y')
        time_period.append(
            [np.datetime64(f"{np.datetime_as_string(year - 1)}-{month_range[0]:02d}-01", "D"),
             np.datetime64(f"{np.datetime_as_string(year)}-{month_range[1]+1:02d}-01", "D")]
        )

    return np.array(time_period)


def get_enso_flavors(nino_indices,
                     month_range=['Dec', 'Feb'],
                     mean=True, threshold=0.5, min_diff=0.0,
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
    if threshold == 'std':
        threshold_nino3 = float(nino_indices['nino3'].std(skipna=True))
        threshold_nino4 = float(nino_indices['nino4'].std(skipna=True))
    else:
        threshold_nino3 = float(threshold)
        threshold_nino4 = float(threshold)
    nino_indices = tu.get_month_range_data(dataset=nino_indices,
                                           start_month=month_range[0],
                                           end_month=month_range[1])
    # Identify El Nino and La Nina types
    nino_years = []
    standard_years = []
    times = np.arange(
        np.array(nino_indices.time.min(), dtype='datetime64[Y]'),
        np.array(nino_indices.time.max(), dtype='datetime64[Y]')
    )
    for y in times:
        sm_month_idx = tu.get_index_of_month(month_range[0]) + 1
        em_month_idx = tu.get_index_of_month(month_range[1]) + 2
        time_range = [np.datetime64(f"{y}-{sm_month_idx:02d}-01", "D"),
                      np.datetime64(f"{y+1}-{em_month_idx:02d}-01", "D")]
        # Select region
        nino34 = nino_indices['nino34'].sel(
            time=slice(time_range[0], time_range[1]))
        nino3 = nino_indices['nino3'].sel(
            time=slice(time_range[0], time_range[1]))
        nino4 = nino_indices['nino4'].sel(
            time=slice(time_range[0], time_range[1]))

        if mean:
            nino34 = nino34.mean(dim='time', skipna=True)
            nino3 = nino3.mean(dim='time', skipna=True)
            nino4 = nino4.mean(dim='time', skipna=True)
        else:
            nino34 = nino34.min(dim='time', skipna=True)
            nino3 = nino3.min(dim='time', skipna=True)
            nino4 = nino4.min(dim='time', skipna=True)

        # El Nino years
        if (
            nino34.data >= 0 and
            (nino3.data >= threshold_nino3 or nino4.data >= threshold_nino4)
        ):
            buff_dic = {'start': time_range[0], 'end': time_range[1],
                        'type': 'Nino', 'label': 4}
            buff_dic['N3-N4'] = nino3.data - nino4.data

            # EP type if DJF nino3 > 0.5 and nino3 > nino4
            if (nino3.data - min_diff) > nino4.data:
                buff_dic['type'] = 'Nino_EP'
                buff_dic['label'] = 0
            # CP type if DJF nino4 > 0.5 and nino3 < nino4
            elif (nino4.data - min_diff) > nino3.data:
                buff_dic['type'] = 'Nino_CP'
                buff_dic['label'] = 1

            nino_years.append(buff_dic)

        # La Nina years
        elif (nino34.data <= -threshold_nino3 and
              (nino3.data <= -threshold_nino3 or nino4.data <= -threshold_nino4)):
            buff_dic = {'start': time_range[0], 'end': time_range[1],
                        'type': 'Nina', 'label': 5}
            buff_dic['N3-N4'] = nino3.data - nino4.data

            # EP type if DJF nino3 < -0.5 and nino3 < nino4
            if (nino3.data + min_diff) < nino4.data:
                buff_dic['type'] = 'Nina_EP'
                buff_dic['label'] = 2
            # CP type if DJF nino4 < -0.5 and nino3 > nino4
            elif (nino4.data + min_diff) < nino3.data:
                buff_dic['type'] = 'Nina_CP'
                buff_dic['label'] = 3
            nino_years.append(buff_dic)

        # standard years
        else:
            buff_dic = {'start': time_range[0], 'end': time_range[1],
                        'type': 'Normal', 'label': 5}
            buff_dic['N3-N4'] = nino3.data - nino4.data
            standard_years.append(buff_dic)

    nino_years = pd.DataFrame(nino_years)
    standard_years = pd.DataFrame(standard_years)

    # Years of strong volcanic erruptions followed by an El Nino
    if drop_volcano_year:
        volcano_years_idx = nino_years.loc[
            (nino_years['start'] == '1955-12-01') |
            (nino_years['start'] == '1956-12-01') |
            (nino_years['start'] == '1957-12-01') |
            (nino_years['start'] == '1963-12-01') |
            (nino_years['start'] == '1980-12-01') |
            (nino_years['start'] == '1982-12-01') |
            (nino_years['start'] == '1991-12-01')
        ].index
        nino_years = nino_years.drop(index=volcano_years_idx)

    return nino_years, standard_years


def tps_enso_years(season_type,
                   month_range=['Dec', 'Sep'],
                   extreme_diff=None,
                   time_range=None,
                   threshold=0.5, min_diff=0.1,
                   drop_volcanic_year=False,
                   consensus=False,
                   print_info=False,
                   start_month='Jan',
                   end_month='Dec'):
    reload(tu)
    reload(gut)
    # Select file for nino indices
    PATH = os.path.dirname(os.path.abspath(__file__))

    fname = PATH + "/../../input/ersst5.nino.mth.91-20.ascii"
    # time snippets
    if consensus:  # consensus
        enso_dict = get_enso_flavors_consensus(
            PATH + "/../../input/enso_types_lit.dat",
            time_range=time_range
        )
    else:
        nino_indices = get_nino_indices(
            fname, time_range=time_range, time_roll=0)
        enso_dict = get_enso_years(nino_indices, month_range=month_range,
                                   mean=True, threshold=threshold,
                                   min_diff=min_diff, extreme_diff=extreme_diff,
                                   drop_volcano_year=drop_volcanic_year)

    if season_type in ["Nino_EP", "Nino_CP", "Nina_CP", "Nina_EP",
                       f'Nino_EP_{extreme_diff}', f'Nino_CP_{extreme_diff}',
                       f'Nina_EP_{extreme_diff}', f'Nina_CP_{extreme_diff}',
                       'nino', 'nina', 'Normal',
                       ]:
        # Select only ENSO years of specific type
        time_period = enso_dict[season_type]
        tps = tu.get_dates_of_time_ranges(time_ranges=time_period)
    elif season_type == 'full':
        tps = tu.get_dates_of_time_range(time_range=time_range)
    else:
        raise ValueError("No valid season_type chosen!")

    # tps = xr.DataArray(data=tps, dims=['time'], coords={'time': tps})
    tps = gut.create_xr_ds(data=tps, dims=['time'], coords={'time': tps})
    tps = tu.get_sel_time_range(tps, time_range=time_range, verbose=False)

    if start_month != 'Jan' or end_month != 'Dec':
        tps = tu.get_month_range_data(tps,
                                      start_month=start_month,
                                      end_month=end_month)

    gut.myprint(time_period, verbose=print_info)

    return tps


def get_time_period_enso(enso_flavors):
    for season_type in ["Nino_EP", "Nino_CP", "Nina_EP", "Nina_CP"]:
        # Select only ENSO years of specific type
        time_period = np.array([
            enso_flavors.loc[(enso_flavors['type'] == season_type)]['start'],
            enso_flavors.loc[(enso_flavors['type'] == season_type)]['end'],
        ]).T
        if season_type == 'Nino_EP':
            Nino_EP_years = time_period
        elif season_type == 'Nino_CP':
            Nino_CP_years = time_period
        elif season_type == 'Nina_EP':
            Nina_EP_years = time_period
        elif season_type == 'Nina_CP':
            Nina_CP_years = time_period

    return {'Nino_EP': Nino_EP_years,
            'Nino_CP': Nino_CP_years,
            'Nina_EP': Nina_EP_years,
            'Nina_CP': Nina_CP_years,
            }


def get_enso_years(nino_indices, month_range=[12, 2],
                   mean=True, threshold=0.5, min_diff=0.0,
                   extreme_diff=0.5,
                   drop_volcano_year=False):
    enso_flavors, standard_ = get_enso_flavors(nino_indices=nino_indices,
                                               month_range=month_range,
                                               mean=mean,
                                               threshold=threshold, min_diff=min_diff,
                                               drop_volcano_year=drop_volcano_year
                                               )
    enso_years_dict = get_time_period_enso(enso_flavors=enso_flavors)
    Nino_EP_years = enso_years_dict['Nino_EP']
    Nino_CP_years = enso_years_dict['Nino_CP']
    Nina_EP_years = enso_years_dict['Nina_EP']
    Nina_CP_years = enso_years_dict['Nina_CP']

    nino_years = np.concatenate([Nino_EP_years, Nino_CP_years])
    nina_years = np.concatenate([Nina_EP_years, Nina_CP_years])

    standard_years = np.array([
        standard_.loc[(standard_['type'] == 'Normal')]['start'],
        standard_.loc[(standard_['type'] == 'Normal')]['end'],
    ]).T

    nino_extr = {f'Nino_EP_{extreme_diff}': None, f'Nino_CP_{extreme_diff}': None,
                 f'Nina_EP_{extreme_diff}': None, f'Nina_CP_{extreme_diff}': None}

    if extreme_diff is not None:
        if extreme_diff < 0.2:
            raise ValueError(
                f'Difference not extreme enough {extreme_diff} < 0.2!')
        enso_flavors_ext, _ = get_enso_flavors(nino_indices=nino_indices,
                                               month_range=month_range,
                                               mean=mean,
                                               threshold=threshold, min_diff=extreme_diff,
                                               drop_volcano_year=drop_volcano_year
                                               )
        enso_years_dict_extreme = get_time_period_enso(
            enso_flavors=enso_flavors_ext)
        nino_extr = {f'Nino_EP_{extreme_diff}': enso_years_dict_extreme['Nino_EP'],
                     f'Nino_CP_{extreme_diff}': enso_years_dict_extreme['Nino_CP'],
                     f'Nina_EP_{extreme_diff}': enso_years_dict_extreme['Nina_EP'],
                     f'Nina_CP_{extreme_diff}': enso_years_dict_extreme['Nina_CP']}

    return dict(
        **{'nino': nino_years,
           'nina': nina_years,
           'Normal': standard_years,
           'Nino_EP': Nino_EP_years,
           'Nino_CP': Nino_CP_years,
           'Nina_EP': Nina_EP_years,
           'Nina_CP': Nina_CP_years,
           },
        **nino_extr
    )


def get_nao(fname, time_range=None, timemean=None):
    reload(tu)
    df = pd.read_csv(
        fname, skiprows=0, names=['YEAR', 'MON', 'DAY', 'NAO'], delim_whitespace=True
    )
    time = []
    for i, row in df.iterrows():
        tp = np.datetime64(
            f"{int(row['YEAR'])}-{int(row['MON']):02}-{int(row['DAY']):02}",
            'D')
        time.append(tp)

    da = xr.DataArray(data=df['NAO'], name='nao', coords={"time": np.array(time)},
                      dims=["time"])

    if time_range is not None:
        da = da.sel(time=slice(np.datetime64(time_range[0], "D"),
                               np.datetime64(time_range[1], "D")))

    if timemean is not None:
        da = tu.apply_timemean(ds=da,
                               timemean=timemean)
    return da


def enso_events_noaa(oni):
    """Get time period of ENSO events based on the definition by NOAA,
        i.e. the ONI index is larger/smaller than +/- 0.5 for
        at least 5 consecutive months.

    Args:
        oni (xr.DataArray): ONI index

    Returns:
        enso_time_period (dict)
    """
    nina = []
    nino = []
    t_idx = 0
    while t_idx < len(oni.time)-1:
        i = 1
        if oni[t_idx] >= 0.5:
            while oni[t_idx + i] >= 0.5 and (t_idx+i < len(oni.time)-1):
                i += 1
            if i >= 4:
                nino.append(
                    [oni[t_idx].time.data, oni[t_idx+i].time.data]
                )
        elif oni[t_idx] <= -0.5:
            while oni[t_idx+i] <= -0.5 and (t_idx+i < len(oni.time)-1):
                i += 1
            if i >= 4:
                nina.append(
                    [oni[t_idx].time.data, oni[t_idx+i].time.data]
                )
        t_idx += i

    return dict(nino=np.array(nino, dtype='datetime64[M]'),
                nina=np.array(nina, dtype='datetime64[M]'))


def enso_flavor_oni(oni, nino_indices, num_months=3, mean=True, min_diff=0.0):
    """Identify ENSO flavors based on the Nino3-Nino4 approach, however
    preselect the ENSO events based on the NOAA definition, i.e. the ONI index.

    Args:
        oni ([type]): [description]
        nino_indices ([type]): [description]
        num_months (int, optional): [description]. Defaults to 3.
        mean (bool, optional): [description]. Defaults to True.
        min_diff (float, optional): [description]. Defaults to 0.0.

    Returns:
        [type]: [description]
    """

    enso_events = enso_events_noaa(oni)
    for key, time_period in enso_events.items():
        enso_years = []
        for tp in time_period:
            # Select months around the max/min of ONI in ENSO event
            buff = oni.sel(time=slice(tp[0], tp[1]))
            if key == 'nino':
                idx_max = buff.argmax(dim='time').data
            elif key == 'nina':
                idx_max = buff.argmin(dim='time').data
            else:
                ValueError

            time_range = [np.array(buff['time'][idx_max - int(num_months/2)].data,
                                   dtype='datetime64[M]'),
                          np.array(buff['time'][idx_max + int(num_months/2)].data,
                                   dtype='datetime64[M]')]

            # Check if new range is out of old range
            assert time_range[0] >= tp[0]
            assert time_range[1] <= tp[1]

            # Take nino3 and nino4 in that time period
            nino3 = nino_indices['nino3'].sel(
                time=slice(time_range[0], time_range[1]))
            nino4 = nino_indices['nino4'].sel(
                time=slice(time_range[0], time_range[1]))

            # Average or min over this period
            if mean:
                nino3 = nino3.mean(dim='time', skipna=True)
                nino4 = nino4.mean(dim='time', skipna=True)
            else:
                nino3 = nino3.min(dim='time', skipna=True)
                nino4 = nino4.min(dim='time', skipna=True)

            # Use Nino3-Nino4 criterion
            buff_dic = {'start': time_range[0], 'end': time_range[1],
                        'type': key}
            buff_dic['N3-N4'] = nino3.data - nino4.data

            if key == 'nino':
                # EP type if DJF nino3 > 0.5 and nino3 > nino4
                if (nino3.data - min_diff) > nino4.data:
                    buff_dic['type'] = f'{key}_EP'
                # CP type if DJF nino4 > 0.5 and nino3 < nino4
                elif (nino4.data - min_diff) > nino3.data:
                    buff_dic['type'] = f'{key}_CP'
            elif key == 'nina':
                # EP type if DJF nino3 < -0.5 and nino3 < nino4
                if (nino3.data + min_diff) < nino4.data:
                    buff_dic['type'] = f'{key}_EP'
                # CP type if DJF nino4 < -0.5 and nino3 > nino4
                elif (nino4.data + min_diff) < nino3.data:
                    buff_dic['type'] = f'{key}_CP'

            enso_years.append(buff_dic)

        return pd.DataFrame(enso_years)


def get_enso_flavors_consensus(fname, time_range=None):
    # Read literature ENSO classification
    df_lit = pd.read_csv(
        fname, skiprows=0, header=0,
        skipinitialspace=True
    )
    df_lit['start'] = [pd.Timestamp(t) for t in df_lit['start']]
    df_lit['end'] = [pd.Timestamp(t) for t in df_lit['end']]

    # remove rows which are not within the time range
    if time_range is not None:
        idx_drop = []
        for i, row in df_lit.iterrows():
            if ((row['start'] < np.array(time_range[0], dtype='datetime64[D]'))
                    or (row['end'] > np.array(time_range[1], dtype='datetime64[D]'))):
                idx_drop.append(i)

        df_lit = df_lit.drop(index=idx_drop)

    enso_years = dict()
    enso_years['Nino_EP'] = np.array([
        df_lit.loc[(df_lit['Cons'] == 'E')]['start'], df_lit.loc[(
            df_lit['Cons'] == 'E')]['end']
    ], dtype='datetime64[D]').T
    enso_years['Nino_CP'] = np.array([
        df_lit.loc[(df_lit['Cons'] == 'C')]['start'], df_lit.loc[(
            df_lit['Cons'] == 'C')]['end']
    ], dtype='datetime64[D]').T

    return enso_years


# ############################ MJO #####################################
def get_mjo_index(time_range=['1981-01-01', '2020-01-01'],
                  start_month='Jan', end_month='Dec'):
    # RMM Index
    rmm_index = xr.open_dataset('/home/strnad/data/MJO-RMM/rmm_index.nc')
    rmm_index = tu.get_sel_time_range(rmm_index, time_range=time_range)
    rmm_index = tu.get_month_range_data(rmm_index,
                                        start_month=start_month,
                                        end_month=end_month)
    return rmm_index


def get_mjophase_tps(phase_number,
                     time_range=['1981-01-01', '2020-01-01'],
                     start_month='Jan', end_month='Dec',
                     active=None,
                     ):
    reload(tsa)
    reload(tu)
    rmm_index = get_mjo_index(time_range=time_range, start_month=start_month,
                              end_month=end_month)
    ampl = rmm_index['amplitude']
    tps = tsa.get_tps4val(ts=rmm_index['phase'], val=phase_number)
    if active is not None:
        if active:
            tps = tu.get_sel_tps_ds(
                ds=tps, tps=ampl.where(ampl >= 1, drop=True).time)
        else:
            tps = tu.get_sel_tps_ds(
                ds=tps, tps=ampl.where(ampl < 1, drop=True).time)
    return tps


def get_bsisophase_tps(phase_number,
                       time_range=['1981-01-01', '2020-01-01'],
                       start_month='Jan', end_month='Dec',
                       active=None,
                       bsiso_name='BSISO1',
                       ampl_th=1.5
                       ):
    reload(tsa)
    reload(tu)
    bsiso_index = get_bsiso_index(time_range=time_range, start_month=start_month,
                                  end_month=end_month,
                                  )
    ampl = bsiso_index[bsiso_name]
    tps = tsa.get_tps4val(
        ts=bsiso_index[f'{bsiso_name}-phase'], val=phase_number)
    if active is not None:
        if active:
            tps = tu.get_sel_tps_ds(
                ds=tps, tps=ampl.where(ampl >= ampl_th, drop=True).time)
        else:
            tps = tu.get_sel_tps_ds(
                ds=tps, tps=ampl.where(ampl < ampl_th, drop=True).time)
    return tps


@np.vectorize
def get_phase_of_angle(angle):
    """Gives the (MJO) phase for an angle between 0-360° (ie. in degree).

    Args:
        angle (float): angle in degree

    Raises:
        ValueError: if angle > 360°

    Returns:
        phase: phase
    """

    # +360 for angle where mod returns negative numbers
    angle = (angle + 360) % 360

    if 0 <= angle < 45:
        phase = 3
    elif 45 <= angle < 90:
        phase = 4
    elif 90 <= angle < 135:
        phase = 5
    elif 135 <= angle < 180:
        phase = 6
    elif 180 <= angle < 225:
        phase = 7
    elif 225 <= angle < 270:
        phase = 8
    elif 270 <= angle < 315:
        phase = 1
    elif 315 <= angle < 360:
        phase = 2
    else:
        raise ValueError(f'angle {angle} not between 0-360')

    return int(phase)


def get_bsiso_index(time_range=['1980-01-01', '2020-01-01'],
                    start_month='Jan', end_month='Dec',
                    index_def='Lee'):
    # BSISO Index
    if index_def == 'Lee':
        bsiso_index = xr.open_dataset('/home/strnad/data/bsiso/BSISO.nc')
    elif index_def == 'Kikuchi':
        bsiso_index = xr.open_dataset('/home/strnad/data/kikuchi_bsiso/BSISO_index.nc')

    bsiso_index = tu.get_time_range_data(bsiso_index, time_range=time_range)
    bsiso_index = tu.get_month_range_data(bsiso_index,
                                          start_month=start_month,
                                          end_month=end_month)
    return bsiso_index
