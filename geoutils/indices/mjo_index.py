import xarray as xr
import climnet.utils.time_utils as tu
from importlib import reload
# %%
# RMM
rmm_file1 = '/home/strnad/data/MJO-RMM/data1.nc'
rmm_file2 = '/home/strnad/data/MJO-RMM/data2.nc'
rmm_file_ampl = '/home/strnad/data/MJO-RMM/data_ampl.nc'
rmm_file_phase = '/home/strnad/data/MJO-RMM/data_phase.nc'
v1 = 'RMM1'
v2 = 'RMM2'
va = 'amplitude'
vp = 'phase'
rmm1_index_raw = xr.open_dataset(rmm_file1)
rmm2_index_raw = xr.open_dataset(rmm_file2)
rmm_ampl_index_raw = xr.open_dataset(rmm_file_ampl)
rmm_phase_index_raw = xr.open_dataset(rmm_file_phase)

# %%
reload(tu)
tps_mjo = tu.get_dates_of_time_range(time_range=['1974-06-01', '2022-02-21']) # Downloaded 23.02.2022


rmm1_index = xr.DataArray(
    data=rmm1_index_raw[v1].data,
    dims=['time'],
    coords=dict(
        time=tps_mjo
    ),
    name=v1
)
savepath_rmm1 = '/home/strnad/data/MJO-RMM/rmm1.nc'
rmm1_index.to_netcdf(savepath_rmm1)


# %%
rmm2_index = xr.DataArray(
    data=rmm2_index_raw[v2].data,
    dims=['time'],
    coords=dict(
        time=tps_mjo
    ),
    name=v2
)
savepath_rmm2 = '/home/strnad/data/MJO-RMM/rmm2.nc'
rmm1_index.to_netcdf(savepath_rmm2)
# %%
# Ampl
rmm_index_ampl = xr.DataArray(
    data=rmm_ampl_index_raw[va].data,
    dims=['time'],
    coords=dict(
        time=tps_mjo
    ),
    name=va
)
savepath_rmm_ampl = '/home/strnad/data/MJO-RMM/rmm_ampl.nc'
rmm_index_ampl.to_netcdf(savepath_rmm_ampl)
# %%
# Phase
rmm_index_phase = xr.DataArray(
    data=rmm_phase_index_raw[vp].data,
    dims=['time'],
    coords=dict(
        time=tps_mjo
    ),
    name=vp
)
savepath_rmm_phase = '/home/strnad/data/MJO-RMM/rmm_ampl.nc'
rmm_index_phase.to_netcdf(savepath_rmm_phase)



# %%
rmm_index = xr.merge([rmm1_index, rmm2_index, rmm_index_ampl, rmm_index_phase])
savepath_rmm = '/home/strnad/data/MJO-RMM/rmm_index.nc'
rmm_index.to_netcdf(savepath_rmm)
# %%
