# %%
import xarray as xr
import pandas as pd
import numpy as np
import geoutils.utils.time_utils as tu
import geoutils.utils.general_utils as gut
from importlib import reload
reload(tu)
reload(gut)
folder = '/home/strnad/data/air/'
fname = '/home/strnad/data/air/All-India-Rainfall_1901_2019.csv'
# fname = f'{folder}/air_index.csv'


def get_air_index(start_month='Jan', end_month='Dec',
                  time_range=None):
    savepath_air = f'{folder}/ari_days.nc'
    air_index = xr.open_dataset(savepath_air)['air']
    air_index = tu.get_month_range_data(air_index,
                                        start_month=start_month,
                                        end_month=end_month,
                                        verbose=False)
    if time_range is not None:
        air_index = tu.get_time_range_data(air_index,
                                           time_range=time_range,
                                           verbose=False)

    air_index.name = 'air'
    air_index = air_index.to_dataset()

    return air_index


def get_air_strength(start_month='Jun', end_month='Sep',
                     time_range=None,
                     air_index_val=0,
                     definition='std',
                     quantile=0.8,
                     verbose=False):
    air_index = get_air_index(start_month=start_month,
                              end_month=end_month,
                              time_range=time_range)
    if definition == 'std':
        std = float(air_index.std()['air'])
        mean = float(air_index.mean()['air'])
        gut.myprint(f'mean {mean}, std: {std}', verbose=verbose)
        reduced_air_index = air_index.where(air_index < mean-std,
                                            drop=True)
        enhanced_air_index = air_index.where(air_index > mean+std,
                                             drop=True)
    elif definition == 'thresh':
        reduced_air_index = air_index.where(air_index < -air_index_val,
                                            drop=True)
        enhanced_air_index = air_index.where(air_index > air_index_val,
                                             drop=True)
    elif definition == 'quantile':
        gut.myprint(
            f'Get strength based on quantile: {quantile}', verbose=verbose)
        reduced_air_index = air_index.where(air_index < air_index.quantile(1-quantile),
                                            drop=True)
        enhanced_air_index = air_index.where(air_index > air_index.quantile(quantile),
                                             drop=True)
    else:
        raise ValueError('Invalid definition for air_index strength')

    gut.myprint(f'# anomalous enhanced air_index times: {len(enhanced_air_index.time)}',
                verbose=verbose)
    gut.myprint(f'# anomalous reduced air_index times: {len(reduced_air_index.time)}',
                verbose=verbose)
    normal_days = tu.remove_time_points(air_index,
                                        [reduced_air_index, enhanced_air_index])

    air_dict = {'enhanced': enhanced_air_index.time,
                'reduced': reduced_air_index.time,
                'normal': normal_days.time}

    return air_dict


if __name__ == '__main__':

    df = pd.read_csv(fname,
                     #  skiprows=1,
                     header=0,
                     # delim_whitespace=True
                     )
    data = df[['JUN', 'JUL', 'AUG', 'SEP']].to_numpy().flatten()
    dates = tu.get_dates_of_time_range(['1901-01-01', '2020-01-01'], freq='M')
    dates = tu.get_month_range_data(dates, 'Jun', 'Sep')
    air = xr.DataArray(data=data, name='air',
                       coords={"time": dates},
                       dims=["time"])
    # air_index = air_index['JUN-SEP'].loc[1998:2018]

    savepath_ari = f'{folder}/airi.nc'
    air.to_netcdf(savepath_ari)
    # %%
    # Interpolate to days

    tps_air_days = tu.get_dates_of_time_range(['1901-01-01', '2020-01-01'],
                                              freq='D')
    tps_air_days = tu.get_month_range_data(tps_air_days, 'Jun', 'Sep')
    air_index_days = air.interp(time=tps_air_days,
                                kwargs={"fill_value": "extrapolate"})

    savepath_ari = f'{folder}/ari_days.nc'
    air_index_days.to_netcdf(savepath_ari)


# %%
