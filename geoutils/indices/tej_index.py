''' File description

@Author  :   Felix Strnad
'''
# %%
import numpy as np
import geoutils.utils.statistic_utils as sut
import geoutils.utils.general_utils as gut
import geoutils.utils.time_utils as tu
import geoutils.geodata.base_dataset as bds
import geoutils.plotting.plots as cplt
from importlib import reload

import geoutils.utils.time_utils as tut
reload(tut)


# ======================================================================================
# Compute the  tropical easterly jet (TEJ) index (TEJI) as the
# mean zonal winds U200 spatially averaged over the region 5-15°N, 10°W-20°E
# ======================================================================================


def get_tej_index(u200,
                  #   lon_range=[-0, 30],
                  #   lat_range=[5, 15],
                  lon_range=[0, 70],  # definition by Huang et al. 2019
                  lat_range=[0, 15],
                  northward_extension=False,
                  start_month='Jan',
                  end_month='Dec',):
    """Returns the tej index based on the 200hPa U-wind dataset.

    Args:
        u200 (xr.dataarray): Zonal winds fields.
        monthly (boolean): Averages time dimensions to monthly.
            Default to False.
        time_range(list, optional): Select Nino indices only in a given time-range.
            Defauts to None

    Returns:
        tej_index (xr.Dataset): Nino indices.
    """
    da = u200
    box_tropics, box_tropics_std = tut.get_mean_time_series(
        da,
        lon_range=lon_range,
        lat_range=lat_range,
        time_roll=0
    )
    if northward_extension:
        box_india, box_tropics_std = tut.get_mean_time_series(
            da,
            lon_range=[70, 80],
            lat_range=[10, 30],
            time_roll=0
        )
    else:
        box_india = 0
    box_sum = box_tropics + box_india
    box_sum.name = 'tej'

    tej_idx = box_sum.to_dataset()
    tej_idx = tu.get_month_range_data(tej_idx,
                                      start_month=start_month,
                                      end_month=end_month,
                                      verbose=False)

    return tej_idx


def get_tej_strength(u200, tej_val=0,
                     quantile=0.8,
                     northward_extension=True,
                     definition='std',
                     start_month='Jan',
                     end_month='Dec',
                     get_index=True,):
    tej = get_tej_index(u200=u200,
                        northward_extension=northward_extension,
                        start_month=start_month,
                        end_month=end_month)

    if definition == 'std':
        std = float(tej.std()['tej'])
        mean = float(tej.mean()['tej'])
        gut.myprint(f'mean {mean}, std: {std}')
        enhanced_tej = tej.where(tej < mean-np.sqrt(std), drop=True)
        reduced_tej = tej.where(tej > mean+np.sqrt(std), drop=True)
    elif definition == 'thresh':
        enhanced_tej = tej.where(tej < -tej_val, drop=True)
        reduced_tej = tej.where(tej > tej_val, drop=True)
    elif definition == 'quantile':
        gut.myprint(f'Get strength based on quantile: {quantile}')
        enhanced_tej = tej.where(tej < tej.quantile(1-quantile), drop=True)
        reduced_tej = tej.where(tej > tej.quantile(quantile), drop=True)
    else:
        raise ValueError('Invalid definition for tej strength')

    gut.myprint(f'# anomalous enhanced TEJ times: {len(enhanced_tej.time)}')
    gut.myprint(f'# anomalous reduced TEJ times: {len(reduced_tej.time)}')
    if get_index:
        return dict(enhanced=enhanced_tej.time,
                    reduced=reduced_tej.time,
                    index=tej)
    else:
        return dict(enhanced=enhanced_tej.time,
                    reduced=reduced_tej.time)


def tej_pattern(u200):
    reload(sut)

    tej_index = get_tej_index(u200=u200, monthly=False)
    regressed_arr = sut.compute_correlation(
        data_array=u200, t_p=tej_index['tej'])

    return regressed_arr


# %%
if __name__ == '__main__':
    reload(cplt)
    data_dir = "/home/strnad/data/"
    plot_dir = "/home/strnad/data/plots/tej/"

    lev = 200
    grid_step = 1
    dataset_file = data_dir + \
        f"climate_data/{grid_step}/era5_u_{grid_step}_{lev}_ds.nc"

    u_def = bds.BaseDataset(data_nc=dataset_file,
                            can=True,
                            an_types=['dayofyear', 'month'],
                            )

    grid_step = 2.5
    dataset_file = data_dir + \
        f"climate_data/{grid_step}/era5_v_{grid_step}_{lev}_ds.nc"
    ds_v200 = bds.BaseDataset(data_nc=dataset_file,
                              can=True,
                              an_types=['dayofyear', 'month'],
                              )
    dataset_file = data_dir + \
        f"climate_data/{grid_step}/era5_u_{grid_step}_{lev}_ds.nc"
    ds_u200 = bds.BaseDataset(data_nc=dataset_file,
                              can=True,
                              an_types=['dayofyear', 'month'],
                              )

    grid_step_z = 2.5
    dataset_file = data_dir + \
        f"climate_data/{grid_step_z}/era5_z_{grid_step_z}_{lev}_ds.nc"
    ds_z200 = bds.BaseDataset(data_nc=dataset_file,
                              can=True,
                              an_types=['dayofyear', 'month'],
                              )
    # %%
    dataset_file = data_dir + \
        f"/climate_data/2.5/era5_t2m_{2.5}_ds.nc"

    ds_sat = bds.BaseDataset(data_nc=dataset_file,
                             can=True,
                             an_types=['JJAS', 'month'],
                             month_range=['Jun', 'Sep'],
                             #  lon_range=lon_range_cut,
                             #  lat_range=lat_range_cut,
                             )
    # %%
    reload(cplt)
    an_type = 'month'
    var_type = f'u_an_{an_type}'

    tej_tps = get_tej_strength(u200=u_def.ds[var_type],
                               definition='quantile',
                               tej_val=3,  # or 3 for clearer results
                               start_month='Jun',
                               end_month='Sep',
                               northward_extension=False,
                               )
    tej_dict = dict(
        enhanced=tej_tps['enhanced'],
        # reduced=tej_tps['reduced'],
    )

    nrows = len(tej_dict)
    ncols = 2
    im = cplt.create_multi_plot(nrows=nrows, ncols=ncols,
                                projection='PlateCarree',
                                lon_range=[-20, 180],
                                lat_range=[-20, 70],
                                wspace=0.1,
                                hspace=0.8,
                                dateline=False)
    for idx, (tej_type, this_tps) in enumerate(tej_dict.items()):
        gut.myprint(f'Plotting {tej_type} {len(this_tps)} time steps')
        mean_tps_u, sig_u = tu.get_mean_tps(ds_u200.ds[f'u_an_{an_type}'],
                                            this_tps.time)
        vmax = 6
        vmin = -vmax

        im_comp = cplt.plot_map(mean_tps_u,
                                ax=im['ax'][idx*ncols + 0],
                                plot_type='contourf',
                                cmap='PuOr',
                                centercolor='white',
                                levels=12,
                                vmin=vmin, vmax=vmax,
                                title=f"U200 Anomalies",
                                label=rf'U-wind Anomalies {lev} hPa (wrt {an_type}) [m/s]',
                                vertical_title=f'{tej_type} TEJ',
                                )

        mean_tps_v, sig_v = tu.get_mean_tps(ds_v200.ds[f'v_an_{an_type}'],
                                            this_tps.time)
        # vmax = 5
        # vmin = -vmax

        # im_comp = cplt.plot_map(mean_tps_v,
        #                         ax=im['ax'][idx*ncols + 1],
        #                         plot_type='contourf',
        #                         cmap='PuOr',
        #                         centercolor='white',
        #                         levels=12,
        #                         vmin=vmin, vmax=vmax,
        #                         title=f"V200 Anomalies",
        #                         label=rf'V-wind Anomalies {lev} hPa (wrt {an_type}) [m/s]',
        #                         )

        mean_tps, sig_z = tu.get_mean_tps(ds_sat.ds[f't2m_an_{an_type}'],
                                          this_tps.time)
        vmax = 1.5
        vmin = -vmax
        im_comp = cplt.plot_map(mean_tps,
                                ax=im['ax'][idx*ncols +1],
                                plot_type='contourf',
                                cmap='RdYlBu_r',
                                centercolor='white',
                                levels=12,
                                vmin=vmin, vmax=vmax,
                                title=f"SAT Anomalies + Wind Field 200hPa",
                                label=rf'Anomalies GP (wrt {an_type}) [$m^2/s^2$]',
                                )
        dict_w = cplt.plot_wind_field(ax=im_comp['ax'],
                                      u=mean_tps_u,
                                      v=mean_tps_v,
                                      #   u=mean_u,
                                      #   v=mean_v,
                                      scale=50,
                                      steps=2,
                                      key_length=2,
                                      )

        plot_dir = "/home/strnad/data/plots/tej/"
        savepath = plot_dir + \
            f"definitions/u200_uv200_{an_type}_tej_types.png"
        cplt.save_fig(savepath=savepath, fig=im['fig'])
    # %%
    im = cplt.create_multi_plot(nrows=1,
                                ncols=2,
                                orientation='horizontal',
                                # hspace=0.7,
                                wspace=0.2,
                                projection='PlateCarree',
                                lat_range=[-50, 70],
                                # lon_range=[0, -60],
                                # dateline=True,
                                central_longitude=30
                                )
    vmin_sst = -1
    vmax_sst = -vmin_sst
    an_type = 'month'
    var_type = f't2m_an_{an_type}'
    label_sst = f'SST Anomalies (wrt {an_type}) [°C]'
    for idx, (group, sel_tps) in enumerate(tej_dict.items()):

        mean, mask = tu.get_mean_tps(ds_sat.ds[var_type], tps=sel_tps)

        im_sst = cplt.plot_map(mean,
                               ax=im['ax'][idx],
                               title=f'{group} TEJ',
                               cmap='RdBu_r',
                               plot_type='contourf',
                               levels=14,
                               centercolor='white',
                               vmin=vmin_sst, vmax=vmax_sst,
                               extend='both',
                               orientation='horizontal',
                               significance_mask=mask,
                               hatch_type='..',
                               label=label_sst,
                               )

    savepath = plot_dir + \
        f"definitions/sst_{an_type}_tej_types.png"
    cplt.save_fig(savepath=savepath, fig=im['fig'])

#%%