import xarray as xr
import geoutils.utils.time_utils as tu
import geoutils.tsa.time_series_analysis as tsa

from importlib import reload

# ############################ MJO #####################################


def get_mjo_index(time_range=['1981-01-01', '2020-01-01'],
                  start_month='Jan', end_month='Dec'):
    # RMM Index
    rmm_index = xr.open_dataset('/home/strnad/data/MJO-RMM/rmm_index.nc')
    rmm_index = tu.get_time_range_data(rmm_index, time_range=time_range)
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

# %%
if __name__ == "__main__":
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
    tps_mjo = tu.get_dates_of_time_range(
        time_range=['1974-06-01', '2022-02-21'])  # Downloaded 23.02.2022

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
    rmm_index = xr.merge(
        [rmm1_index, rmm2_index, rmm_index_ampl, rmm_index_phase])
    savepath_rmm = '/home/strnad/data/MJO-RMM/rmm_index.nc'
    rmm_index.to_netcdf(savepath_rmm)
    # %%
