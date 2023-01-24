# %%
import climnet.utils.statistic_utils as sut
import climnet.utils.indices_utils as iut
import numpy as np
import xarray as xr
import climnet.utils.time_utils as tu
from importlib import reload
import pandas as pd


bsiso_file = '/home/strnad/data/bsiso/BSISO.INDEX.NORM.data'
bsiso_file = '/home/strnad/data/bsiso/BSISO.INDEX.NORM.data_new'

data = pd.read_csv(bsiso_file, delim_whitespace=True, header=0)

# %%
tps_bsiso = tu.get_dates_of_time_range(
    time_range=['1981-01-01', '2022-06-11'])  # Downloaded 14.06.2022
len(tps_bsiso)

# %%
bsiso1_1_index = xr.DataArray(
    data=data['BSISO1-1'],
    dims=['time'],
    coords=dict(
        time=tps_bsiso
    ),
    name='BSISO1-1'
)
savepath_bsiso1_1 = '/home/strnad/data/bsiso/BSISO1-1.nc'
bsiso1_1_index.to_netcdf(savepath_bsiso1_1)
# %%
bsiso1_2_index = xr.DataArray(
    data=data['BSISO1-2'],
    dims=['time'],
    coords=dict(
        time=tps_bsiso
    ),
    name='BSISO1-2'
)
savepath_bsiso1_2 = '/home/strnad/data/bsiso/BSISO1-2.nc'
bsiso1_2_index.to_netcdf(savepath_bsiso1_2)
# %%
# Construct phase BSISO 1
reload(iut)

angle_1 = np.rad2deg(np.arctan2(sut.standardize(data['BSISO1-2']),  # X-axis is PC2
                                sut.standardize(data['BSISO1-1'])   # Y-axis is PC1
                                ))
# angle_1 = np.rad2deg(np.arctan2(data['BSISO1-2'],
#                                 data['BSISO1-1']
#                                 ))
phase_1 = iut.get_phase_of_angle(angle_1)
bsiso1_phase = xr.DataArray(
    data=phase_1,
    dims=['time'],
    coords=dict(
        time=tps_bsiso
    ),
    name='BSISO1-phase'
)
savepath_bsiso1_phase = '/home/strnad/data/bsiso/BSISO1-phase.nc'
bsiso1_phase.to_netcdf(savepath_bsiso1_phase)


# %%
bsiso1_index = xr.DataArray(
    data=data['BSISO1'],
    dims=['time'],
    coords=dict(
        time=tps_bsiso
    ),
    name='BSISO1'
)
savepath_bsiso1 = '/home/strnad/data/bsiso/BSISO1.nc'
bsiso1_index.to_netcdf(savepath_bsiso1)

# %%
bsiso2_1_index = xr.DataArray(
    data=data['BSISO2-1'],
    dims=['time'],
    coords=dict(
        time=tps_bsiso
    ),
    name='BSISO2-1'
)
savepath_bsiso2_1 = '/home/strnad/data/bsiso/BSISO2-1.nc'
bsiso2_1_index.to_netcdf(savepath_bsiso2_1)
# %%
bsiso2_2_index = xr.DataArray(
    data=data['BSISO2-2'],
    dims=['time'],
    coords=dict(
        time=tps_bsiso
    ),
    name='BSISO2-2'
)
savepath_bsiso2_2 = '/home/strnad/data/bsiso/BSISO2-2.nc'
bsiso2_2_index.to_netcdf(savepath_bsiso2_2)

# %%
# Construct phase BSISO 2
reload(iut)

angle_2 = np.rad2deg(np.arctan2(sut.standardize(data['BSISO2-2']),
                                sut.standardize(data['BSISO2-1'])
                                ))
phase_2 = iut.get_phase_of_angle(angle_2)
bsiso2_phase = xr.DataArray(
    data=phase_2,
    dims=['time'],
    coords=dict(
        time=tps_bsiso
    ),
    name='BSISO2-phase'
)
savepath_bsiso2_phase = '/home/strnad/data/bsiso/BSISO2-phase.nc'
bsiso2_phase.to_netcdf(savepath_bsiso2_phase)
# %%
bsiso2_index = xr.DataArray(
    data=data['BSISO2'],
    dims=['time'],
    coords=dict(
        time=tps_bsiso
    ),
    name='BSISO2'
)
savepath_bsiso2 = '/home/strnad/data/bsiso/BSISO2.nc'
bsiso2_index.to_netcdf(savepath_bsiso2)

# %%
bsiso_index = xr.merge([bsiso1_1_index, bsiso1_2_index, bsiso1_index,
                        bsiso2_1_index, bsiso2_2_index, bsiso2_index,
                        bsiso1_phase, bsiso2_phase])
savepath_bsiso = '/home/strnad/data/bsiso/BSISO.nc'
bsiso_index.to_netcdf(savepath_bsiso)
# %%
