# %%
import geoutils.plotting.plots as cplt
import geoutils.utils.statistic_utils as sut
import geoutils.utils.indices_utils as iut
import numpy as np
import xarray as xr
import geoutils.utils.time_utils as tu
from importlib import reload
import pandas as pd


mei_file = '/home/strnad/data/meiv2/meiv2.data'  # Only available as monthly data

data = pd.read_csv(mei_file, delim_whitespace=True,
                   skiprows=1,
                   names=tu.months
                   )

# %%
reload(tu)
tps_mei_months = tu.get_dates_of_time_range(
    time_range=['1979-01', '2022-12'],
    freq='M')  # Downloaded 20.10.2022


only_data = data.to_numpy().flatten()
# %%
# %%
meiv2_index = xr.DataArray(
    data=only_data,
    dims=['time'],
    coords=dict(
        time=tps_mei_months
    ),
    name='MEIV2'
)
savepath_meiv2 = '/home/strnad/data/meiv2/meiv2.nc'
meiv2_index.to_netcdf(savepath_meiv2)
# %%
# Interpolate to days
tps_mei_days = tu.get_dates_of_time_range(
    time_range=['1979-01', '2022-12'],
    freq='D')
meiv2_index_days = meiv2_index.interp(time=tps_mei_days)

savepath_meiv2 = '/home/strnad/data/meiv2/meiv2_days.nc'
meiv2_index_days.to_netcdf(savepath_meiv2)

# %%
reload(cplt)

cplt.plot_xy(x_arr=[meiv2_index.time,
                    meiv2_index_days.time],
             y_arr=[meiv2_index, meiv2_index_days],
             label_arr=['MEIv2', 'MEIv2 Days'])
