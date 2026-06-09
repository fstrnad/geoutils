''' File description

@Author  :   Felix Strnad
'''
# %%
import geoutils.utils.statistic_utils as sut
import geoutils.utils.general_utils as gut
import geoutils.utils.time_utils as tu
import geoutils.geodata.base_dataset as bds
import geoutils.plotting.plots as cplt
from importlib import reload
import geoutils.utils.time_utils as tut
reload(tut)


# ======================================================================================
# Compute the  circumglobal teleconnection index (CGTI) as the
# mean Z200 spatially averaged over the region 35-40°N, 60-70°E
# ======================================================================================


def get_cgt_index(z200, start_month='Jan', end_month='Dec',):
    """Returns the cgti index based on the 200hPa GPH dataset.

    Args:
        z200 (xr.dataarray): Zonal winds fields.
        monthly (boolean): Averages time dimensions to monthly.
            Default to False.
        time_range(list, optional): Select Nino indices only in a given time-range.
            Defauts to None

    Returns:
        cgti_index (xr.Dataset): Nino indices.
    """
    da = z200
    box_tropics, box_tropics_std = tut.get_mean_time_series(
        da, lon_range=[60, 70],  # Defined by Ding&Wang 2005
        lat_range=[35, 40],  # Defined as by Ding&Wang 2005  (compare DW2007)
        time_roll=0
    )
    box_tropics.name = 'cgti'

    cgti_idx = box_tropics.to_dataset()

    cgti_idx = tu.get_month_range_data(cgti_idx,
                                       start_month=start_month,
                                       end_month=end_month)
    return cgti_idx


def get_cgti_strength(z200,
                      cgti_val=0,
                      quantile=0.8,
                      start_month='Jun',
                      end_month='Sep',
                      definition='std'):
    cgti = get_cgt_index(z200=z200,
                         start_month=start_month,
                         end_month=end_month,)
    if cgti_val != 0:
        definition = 'thresh'
    if quantile != 0.8:
        definition = 'quantile'

    if definition == 'std':
        std = cgti.std()
        pos_cgti = cgti.where(cgti > std, drop=True)
        neg_cgti = cgti.where(cgti < -std, drop=True)
    elif definition == 'thresh':
        pos_cgti = cgti.where(cgti > cgti_val, drop=True)
        neg_cgti = cgti.where(cgti < -cgti_val, drop=True)
    elif definition == 'quantile':
        pos_cgti = cgti.where(
            (cgti > cgti.quantile(quantile)).compute(), drop=True)
        neg_cgti = cgti.where(
            (cgti < cgti.quantile(1 - quantile)).compute(), drop=True)
    else:
        raise ValueError('Invalid definition for cgti strength')

    return dict(pos=pos_cgti.time,
                neg=neg_cgti.time,
                index=cgti)


def cgt_pattern(z200):
    reload(sut)

    cgt_index = get_cgt_index(z200=z200, monthly=False)
    regressed_arr = sut.compute_correlation(
        data_array=z200, t_p=cgt_index['cgti'])

    return regressed_arr


# %%
if __name__ == '__main__':
    reload(bds)
    data_dir = "/home/strnad/data/"
    plot_dir = "/home/strnad/data/plots/cgti/"

    # Load the data
    grid_step = 2.5
    lev = 200

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
    # %%
    reload(bds)
    grid_step_z = 2.5
    dataset_file = data_dir + \
        f"climate_data/{grid_step_z}/era5_z_{grid_step_z}_{lev}_ds.nc"
    ds_z200 = bds.BaseDataset(data_nc=dataset_file,
                              can=True,
                              an_types=['dayofyear', 'month', 'season'],
                              )
    # %%
    dataset_file = data_dir + \
        f"/climate_data/2.5/era5_sst_{2.5}_ds.nc"

    ds_sst = bds.BaseDataset(data_nc=dataset_file,
                             can=True,
                             an_types=['JJAS', 'month'],
                             month_range=['Jun', 'Sep'],
                             #  lon_range=lon_range_cut,
                             #  lat_range=lat_range_cut,
                             )

    # %%
    an_type = 'month'
    var_type = f'z_an_{an_type}'
    cgti_dict = get_cgti_strength(ds_z200.ds[var_type],
                                  definition='quantile',
                                  quantile=0.9,
                                #   cgti_val=10,
                                  start_month='Jun',
                                  end_month='Sep',
                                  )
    # %%
    reload(cplt)
    cgt_types = ['pos', 'neg']
    nrows = len(cgt_types)
    ncols = 1
    im = cplt.create_multi_plot(nrows=nrows, ncols=ncols,
                                projection='PlateCarree',
                                lon_range=[-20, 180],
                                lat_range=[-0, 80],
                                wspace=0.1,
                                hspace=0.4,
                                dateline=False)
    for idx, (cgti_type) in enumerate(cgt_types):
        this_tps = cgti_dict[cgti_type]
        gut.myprint(f'Plotting {cgti_type} {len(this_tps)} time steps')
        mean_tps_u, sig_u = tu.get_mean_tps(ds_u200.ds[f'u_an_{an_type}'],
                                            this_tps.time)
        # vmax = 6
        # vmin = -vmax

        # im_comp = cplt.plot_map(mean_tps_u,
        #                         ax=im['ax'][idx*ncols + 0],
        #                         plot_type='contourf',
        #                         cmap='PuOr',
        #                         centercolor='white',
        #                         levels=12,
        #                         vmin=vmin, vmax=vmax,
        #                         title=f"U200 Anomalies",
        #                         label=rf'U-wind Anomalies {lev} hPa (wrt {an_type}) [m/s]',
        #                         vertical_title=f'{cgti_type} cgti',
        #                         )

        mean_tps_v, sig_v = tu.get_mean_tps(ds_v200.ds[f'v_an_{an_type}'],
                                            this_tps.time)
        vmax = 6
        vmin = -vmax

        im_comp = cplt.plot_map(mean_tps_v*sig_v,
                                ax=im['ax'][idx*ncols + 0],
                                plot_type='contourf',
                                cmap='PuOr',
                                centercolor='white',
                                levels=12,
                                vmin=vmin, vmax=vmax,
                                # title=f"V200 Anomalies",
                                vertical_title=f'{cgti_type}. CGT',
                                label=rf'V-wind ({lev} hPa) Anomalies (wrt {an_type}) [m/s]',
                                )

        # mean_tps, sig_z = tu.get_mean_tps(ds_z200.ds[f'z_an_{an_type}'],
        #                                   this_tps.time)
        # vmax = 5.e2
        # vmin = -vmax
        # im_comp = cplt.plot_map(mean_tps,
        #                         ax=im['ax'][idx*ncols + 1],
        #                         plot_type='contourf',
        #                         cmap='RdYlBu_r',
        #                         centercolor='white',
        #                         levels=12,
        #                         vmin=vmin, vmax=vmax,
        #                         title=f"z200 Anomalies",
        #                         label=rf'Anomalies GP (wrt {
        #                             an_type}) [$m^2/s^2$]',
        #                         )
        # dict_w = cplt.plot_wind_field(ax=im_comp['ax'],
        #                               u=mean_tps_u,
        #                               v=mean_tps_v,
        #                               #   u=mean_u,
        #                               #   v=mean_v,
        #                               scale=50,
        #                               steps=2,
        #                               key_length=2,
        #                               )

        savepath = plot_dir + \
            f"definitions/v200_{an_type}_cgti_types.png"
        cplt.save_fig(savepath=savepath, fig=im['fig'])

    # %%
    # SST
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
    var_type = f'an_{an_type}'
    label_sst = f'SST Anomalies (wrt {an_type}) [°C]'
    for idx, (group, sel_tps) in enumerate(cgti_dict.items()):

        mean, mask = tu.get_mean_tps(ds_sst.ds[var_type], tps=sel_tps)

        im_sst = cplt.plot_map(mean,
                               ax=im['ax'][idx],
                               title=f'{group} CGTI',
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
        f"definitions/sst_{an_type}_cgti_types.png"
    cplt.save_fig(savepath=savepath, fig=im['fig'])
