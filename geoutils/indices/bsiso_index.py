# %%
import geoutils.utils.file_utils as fut
import geoutils.utils.statistic_utils as sut
import numpy as np
import xarray as xr
import geoutils.utils.time_utils as tu
import geoutils.indices.indices_utils as iut
import geoutils.tsa.time_series_analysis as tsa

from importlib import reload
import pandas as pd


def get_bsiso_index(time_range=['1980-01-01', '2020-01-01'],
                    start_month='Jan', end_month='Dec',
                    index_def='Kikuchi'):
    # BSISO Index
    if index_def == 'Lee':
        bsiso_index = xr.open_dataset('/home/strnad/data/bsiso/BSISO.nc')
    elif index_def == 'Kikuchi':
        bsiso_index = xr.open_dataset(
            '/home/strnad/data/kikuchi_bsiso/BSISO_index.nc')

    bsiso_index = tu.get_time_range_data(bsiso_index, time_range=time_range)
    bsiso_index = tu.get_month_range_data(bsiso_index,
                                          start_month=start_month,
                                          end_month=end_month)
    return bsiso_index


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


if __name__ == '__main__':
    # %%
    # Lee et al. 2013
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
        name='BSISO1'
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
        name='BSISO2'
    )
    savepath_bsiso1_2 = '/home/strnad/data/bsiso/BSISO1-2.nc'
    bsiso1_2_index.to_netcdf(savepath_bsiso1_2)
    # %%
    # Construct phase BSISO 1

    angle_1 = np.rad2deg(np.arctan2(sut.standardize(data['BSISO1-2']),  # X-axis is PC2
                                    # Y-axis is PC1
                                    sut.standardize(data['BSISO1-1'])
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
        name='BSISO-phase'
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
        name='BSISO-ampl'
    )
    savepath_bsiso1 = '/home/strnad/data/bsiso/BSISO_ampl.nc'
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
        name='BSISO2-ampl'
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
    # Kickuchi et al. 2012 index (https://iprc.soest.hawaii.edu/users/kazuyosh/Bimodal_ISO.html)

    bsiso_file = '/home/strnad/data/kikuchi_bsiso/BSISO_25-90bpfil_pc.extension.txt'

    raw_data = pd.read_csv(bsiso_file, delim_whitespace=True, header=0)

    # %%
    tps_bsiso = tu.get_dates_of_time_range(
        time_range=['1979-03-22', '2021-10-22'])  # Downloaded 14.06.2022
    tps_bsiso = tu.get_month_range_data(tps_bsiso,
                                        start_month='Jan',
                                        end_month='Dec'
                                        )

    len(tps_bsiso)
    bsiso1_index = xr.DataArray(
        data=raw_data['PCx'],
        dims=['time'],
        coords=dict(
            time=tps_bsiso
        ),
        name='BSISO1'
    )
    bsiso2_index = xr.DataArray(
        data=raw_data['PCy'],
        dims=['time'],
        coords=dict(
            time=tps_bsiso
        ),
        name='BSISO2'
    )
    bsiso_phase = xr.DataArray(
        data=raw_data['phase'],
        dims=['time'],
        coords=dict(
            time=tps_bsiso
        ),
        name='BSISO-phase'
    )
    bsiso_ampl = xr.DataArray(
        data=raw_data['Amp(nrm)'],
        dims=['time'],
        coords=dict(
            time=tps_bsiso
        ),
        name='BSISO-ampl'
    )

    bsiso_index = xr.merge(
        [bsiso1_index, bsiso2_index, bsiso_phase, bsiso_ampl])
    savepath_bsiso1 = '/home/strnad/data/kikuchi_bsiso/BSISO_index.nc'
    fut.save_ds(ds=bsiso_index,
                filepath=savepath_bsiso1)

    # %%
