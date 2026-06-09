# %%
import geoutils.geodata.base_dataset as bds
import geoutils.utils.file_utils as fut
import geoutils.utils.statistic_utils as sut
import numpy as np
import xarray as xr
import geoutils.utils.time_utils as tu
import geoutils.indices.indices_utils as iut
import geoutils.tsa.time_series_analysis as tsa
import geoutils.plotting.plots as cplt

from importlib import reload
import pandas as pd


def get_mjo_index(time_range=['1981-01-01', '2020-01-01'],
                  start_month='Jan', end_month='Dec',
                  verbose=False):
    # RMM Index
    rmm_index = xr.open_dataset('/home/strnad/data/MJO-RMM/rmm_index.nc')
    rmm_index = tu.get_time_range_data(rmm_index, time_range=time_range)
    rmm_index = tu.get_month_range_data(rmm_index,
                                        start_month=start_month,
                                        end_month=end_month,
                                        verbose=verbose)
    return rmm_index


def get_mjophase_tps(phase_number,
                     time_range=['1981-01-01', '2020-01-01'],
                     start_month='Jan', end_month='Dec',
                     active=None, verbose=False
                     ):
    reload(tsa)
    reload(tu)
    rmm_index = get_mjo_index(time_range=time_range, start_month=start_month,
                              end_month=end_month, verbose=verbose)
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


if __name__ == '__main__':
    mjo_file = '/home/strnad/data/mjo/mjo.INDEX.NORM.data'
    mjo_file = '/home/strnad/data/mjo/mjo.INDEX.NORM.data_new'

    data = pd.read_csv(mjo_file, delim_whitespace=True, header=0)

    # %%
    tps_mjo = tu.get_dates_of_time_range(
        time_range=['1981-01-01', '2022-06-11'])  # Downloaded 14.06.2022
    len(tps_mjo)

    # %%
    mjo1_1_index = xr.DataArray(
        data=data['mjo1-1'],
        dims=['time'],
        coords=dict(
            time=tps_mjo
        ),
        name='mjo1'
    )
    savepath_mjo1_1 = '/home/strnad/data/mjo/mjo1-1.nc'
    mjo1_1_index.to_netcdf(savepath_mjo1_1)
    # %%
    mjo1_2_index = xr.DataArray(
        data=data['mjo1-2'],
        dims=['time'],
        coords=dict(
            time=tps_mjo
        ),
        name='mjo2'
    )
    savepath_mjo1_2 = '/home/strnad/data/mjo/mjo1-2.nc'
    mjo1_2_index.to_netcdf(savepath_mjo1_2)
    # %%
    # Construct phase mjo 1

    angle_1 = np.rad2deg(np.arctan2(sut.standardize(data['mjo1-2']),  # X-axis is PC2
                                    # Y-axis is PC1
                                    sut.standardize(data['mjo1-1'])
                                    ))
    # angle_1 = np.rad2deg(np.arctan2(data['mjo1-2'],
    #                                 data['mjo1-1']
    #                                 ))
    phase_1 = iut.get_phase_of_angle(angle_1)
    mjo1_phase = xr.DataArray(
        data=phase_1,
        dims=['time'],
        coords=dict(
            time=tps_mjo
        ),
        name='mjo-phase'
    )
    savepath_mjo1_phase = '/home/strnad/data/mjo/mjo1-phase.nc'
    mjo1_phase.to_netcdf(savepath_mjo1_phase)

    # %%
    mjo1_index = xr.DataArray(
        data=data['mjo1'],
        dims=['time'],
        coords=dict(
            time=tps_mjo
        ),
        name='mjo-ampl'
    )
    savepath_mjo1 = '/home/strnad/data/mjo/mjo_ampl.nc'
    mjo1_index.to_netcdf(savepath_mjo1)

    # %%
    mjo2_1_index = xr.DataArray(
        data=data['mjo2-1'],
        dims=['time'],
        coords=dict(
            time=tps_mjo
        ),
        name='mjo2-1'
    )
    savepath_mjo2_1 = '/home/strnad/data/mjo/mjo2-1.nc'
    mjo2_1_index.to_netcdf(savepath_mjo2_1)
    # %%
    mjo2_2_index = xr.DataArray(
        data=data['mjo2-2'],
        dims=['time'],
        coords=dict(
            time=tps_mjo
        ),
        name='mjo2-2'
    )
    savepath_mjo2_2 = '/home/strnad/data/mjo/mjo2-2.nc'
    mjo2_2_index.to_netcdf(savepath_mjo2_2)

    # %%
    # Construct phase mjo 2
    reload(iut)

    angle_2 = np.rad2deg(np.arctan2(sut.standardize(data['mjo2-2']),
                                    sut.standardize(data['mjo2-1'])
                                    ))
    phase_2 = iut.get_phase_of_angle(angle_2)
    mjo2_phase = xr.DataArray(
        data=phase_2,
        dims=['time'],
        coords=dict(
            time=tps_mjo
        ),
        name='mjo2-phase'
    )
    savepath_mjo2_phase = '/home/strnad/data/mjo/mjo2-phase.nc'
    mjo2_phase.to_netcdf(savepath_mjo2_phase)
    # %%
    mjo2_index = xr.DataArray(
        data=data['mjo2'],
        dims=['time'],
        coords=dict(
            time=tps_mjo
        ),
        name='mjo2-ampl'
    )
    savepath_mjo2 = '/home/strnad/data/mjo/mjo2.nc'
    mjo2_index.to_netcdf(savepath_mjo2)

    # %%
    mjo_index = xr.merge([mjo1_1_index, mjo1_2_index, mjo1_index,
                          mjo2_1_index, mjo2_2_index, mjo2_index,
                          mjo1_phase, mjo2_phase])
    savepath_mjo = '/home/strnad/data/mjo/mjo.nc'
    mjo_index.to_netcdf(savepath_mjo)
    # %%
    # Kickuchi et al. 2012 index (https://iprc.soest.hawaii.edu/users/kazuyosh/Bimodal_ISO.html)

    mjo_file = '/home/strnad/data/kikuchi_mjo/mjo_25-90bpfil_pc.extension.txt'

    raw_data = pd.read_csv(mjo_file, delim_whitespace=True, header=0)

    # %%
    tps_mjo = tu.get_dates_of_time_range(
        time_range=['1979-03-22', '2021-10-22'])  # Downloaded 14.06.2022
    tps_mjo = tu.get_month_range_data(tps_mjo,
                                      start_month='Jan',
                                      end_month='Dec'
                                      )

    len(tps_mjo)
    mjo1_index = xr.DataArray(
        data=raw_data['PCx'],
        dims=['time'],
        coords=dict(
            time=tps_mjo
        ),
        name='mjo1'
    )
    mjo2_index = xr.DataArray(
        data=raw_data['PCy'],
        dims=['time'],
        coords=dict(
            time=tps_mjo
        ),
        name='mjo2'
    )
    mjo_phase = xr.DataArray(
        data=raw_data['phase'],
        dims=['time'],
        coords=dict(
            time=tps_mjo
        ),
        name='mjo-phase'
    )
    mjo_ampl = xr.DataArray(
        data=raw_data['Amp(nrm)'],
        dims=['time'],
        coords=dict(
            time=tps_mjo
        ),
        name='mjo-ampl'
    )

    mjo_index = xr.merge(
        [mjo1_index, mjo2_index, mjo_phase, mjo_ampl])
    savepath_mjo1 = '/home/strnad/data/kikuchi_mjo/mjo_index.nc'
    fut.save_ds(ds=mjo_index,
                filepath=savepath_mjo1)

    # %%
    # check for consistency
    reload(bds)
    reload(cplt)
    data_dir = "/home/strnad/data/"
    dataset_file = data_dir + \
        f"climate_data/2.5/era5_ttr_{2.5}_ds.nc"

    ds_olr_25 = bds.BaseDataset(data_nc=dataset_file,
                                can=True,
                                an_types=['dayofyear', 'month', 'JJAS'],

                                )
    # %%
    nrows = 3
    ncols = 3
    an_type = 'month'
    var_type = f'olr_an_{an_type}'
    label = rf'Anomalies OLR (wrt {an_type}) [W/m$^2$]'
    im = cplt.create_multi_plot(nrows=nrows,
                                ncols=ncols,
                                figsize=(18, 10),
                                projection='PlateCarree',
                                hspace=0.,
                                wspace=0.15,
                                orientation='horizontal',
                                lon_range=[30, -140],
                                dateline=True,
                                lat_range=[-30, 40],
                                end_idx=8,)

    for idx, phase in enumerate(np.arange(1, 9)):
        tps = get_mjophase_tps(
            phase_number=phase,
            # start_month='Jun',
            # end_month='Sep',
            active=True)

        composite_pr = tu.get_sel_tps_ds(ds=ds_olr_25.ds, tps=tps)
        mean_pr = composite_pr.mean(dim='time')

        vmax = 20
        vmin = -vmax

        im_comp = cplt.plot_map(mean_pr[var_type],
                                ax=im['ax'][idx],
                                plot_type='contourf',
                                cmap='RdBu_r',
                                centercolor='white',
                                levels=12,
                                vmin=vmin, vmax=vmax,
                                title=f"Phase {phase}",  # ({len(tps)} days)",
                                extend='both',
                                )
    cplt.add_colorbar(im=im_comp,
                      fig=im['fig'],
                      width=0.8,
                      height=0.015,
                      x_pos=0.1,
                      y_pos=0.09,
                      label=label)
    plot_dir = "/home/strnad/data/plots/mjo/"
    savepath = plot_dir + \
        f"definitions/mjo_phases_olr_{an_type}.png"
    cplt.save_fig(savepath=savepath, fig=im['fig'])
