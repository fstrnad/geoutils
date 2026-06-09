# %%
import geoutils.plotting.plots as cplt
import geoutils.utils.statistic_utils as sut
import geoutils.indices.indices_utils as iut
import numpy as np
import xarray as xr
import geoutils.utils.time_utils as tu
from importlib import reload
import pandas as pd


mei_file = '/home/strnad/data/nao/norm.nao.monthly.b5001.current.txt'  # Only available as monthly data

data = pd.read_csv(mei_file, delim_whitespace=True,
                   skiprows=1,
                   names=tu.months
                   )
# %%
# Downloaded 24.01.2023
reload(tu)
time_range = ['1950-01', '2022-12']
tps_nao_idx_months = tu.get_dates_of_time_range(
    time_range=time_range,
    freq='M')

only_data = data.to_numpy().flatten()
# %%
# %%
# Save as monthly time series
nao_index = xr.DataArray(
    data=only_data,
    dims=['time'],
    coords=dict(
        time=tps_nao_idx_months
    ),
    name='NAO'
)
savepath_nao = '/home/strnad/data/nao/nao_index_months.nc'
nao_index.to_netcdf(savepath_nao)
# %%
# Interpolate to days
tps_nao_days = tu.get_dates_of_time_range(
    time_range=time_range,
    freq='D')
nao_index_days = nao_index.interp(time=tps_nao_days)

savepath_nao = '/home/strnad/data/nao/nao_index_days.nc'
nao_index_days.to_netcdf(savepath_nao)

# %%
# Consistency check of monthly and daily time series
reload(cplt)

cplt.plot_xy(x_arr=[nao_index.time,
                    nao_index_days.time],
             y_arr=[nao_index, nao_index_days],
             label_arr=['NAO Months', 'NAO Days'])