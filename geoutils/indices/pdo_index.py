# %%
import geoutils.plotting.plots as cplt
import geoutils.utils.statistic_utils as sut
import geoutils.indices.indices_utils as iut
import numpy as np
import xarray as xr
import geoutils.utils.time_utils as tu
from importlib import reload
import pandas as pd


def get_pdo_index(start_month='Jan', end_month='Dec'):
    savepath_pdo = '/home/strnad/data/pdo/pdo_days.nc'
    pdo_index = xr.open_dataset(savepath_pdo)['pdo']
    pdo_index = tu.get_month_range_data(pdo_index,
                                        start_month=start_month,
                                        end_month=end_month,
                                        verbose=False)
    return pdo_index


def get_pdo_types(start_month='Jan', end_month='Dec'):
    pdo_index = get_pdo_index(start_month=start_month, end_month=end_month)
    # All ENSO time points
    pos_tps = pdo_index[pdo_index > 0]
    neg_tps = pdo_index[pdo_index <= 0]
    if len(pos_tps) + len(neg_tps) != len(pdo_index):
        raise ValueError('Some time points are missing')
    pdo_types = {
        'positive': pos_tps,
        'negative': neg_tps,
    }
    return pdo_types


def get_enso_tps(start_month='Jan', end_month='Dec', enso_type='nino'):
    pdo_types = get_pdo_types(start_month=start_month, end_month=end_month)

    if enso_type == 'nino':
        enso_tps = pdo_types['El Nino']
    elif enso_type == 'nina':
        enso_tps = pdo_types['La Nina']
    elif enso_type == 'neutral':
        enso_tps = pdo_types['neutral']
    else:
        raise ValueError(f'Unknown enso_type: {enso_type}')

    return enso_tps


if __name__ == '__main__':

    # Only available as monthly data
    pdo_file = '/home/strnad/data/pdo/ersst.v5.pdo.dat'

    data = pd.read_csv(pdo_file, delim_whitespace=True,
                       skiprows=1,
                       names=tu.months
                       )

    reload(tu)
    tps_pdo_months = tu.get_dates_of_time_range(
        time_range=['1854-01', '2023-12'],
        freq='M')  # Downloaded 05.02.2024

    only_data = data.to_numpy().flatten()
    # %%
    pdo_index = xr.DataArray(
        data=only_data,
        dims=['time'],
        coords=dict(
            time=tps_pdo_months
        ),
        name='pdo'
    )
    savepath_pdo = '/home/strnad/data/pdo/pdo_index.nc'
    pdo_index.to_netcdf(savepath_pdo)
    # %%
    # Interpolate to days
    tps_pdo_days = tu.get_dates_of_time_range(
        time_range=['1854-01', '2023-12'],
        freq='D')
    pdo_index_days = pdo_index.interp(time=tps_pdo_days)

    savepath_pdo = '/home/strnad/data/pdo/pdo_days.nc'
    pdo_index_days.to_netcdf(savepath_pdo)

    # %%
    reload(cplt)
    pdo_types = get_pdo_types()
    pos_tps = pdo_types['positive']
    neg_tps = pdo_types['negative']
    cplt.plot_xy(x_arr=[pos_tps.time, neg_tps.time],
                 y_arr=[pos_tps, neg_tps],
                 label_arr=['Pos PDO', 'Neg PDO'],
                 xlabel='Time',
                 ylabel='PDO Index',)
