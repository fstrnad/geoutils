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
