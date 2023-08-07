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

import geoutils.tsa.pca.pca as pca
import geoutils.utils.time_utils as tut
reload(tut)


# ======================================================================================
# Compute the  circumglobal teleconnection index (CGTI) as the
# mean Z200 spatially averaged over the region 35-40°N, 60-70°E
# ======================================================================================


def get_cgt_index(z200, ):
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

    return cgti_idx


def get_cgti_strength(z200, cgti_val=0, definition='std'):
    cgti = get_cgt_index(z200=z200)
    if definition == 'std':
        std = cgti.std()
        pos_cgti = cgti.where(cgti > std, drop=True)
        neg_cgti = cgti.where(cgti < -std, drop=True)
    elif definition == 'thresh':
        pos_cgti = cgti.where(cgti > cgti_val, drop=True)
        neg_cgti = cgti.where(cgti < -cgti_val, drop=True)
    else:
        raise ValueError('Invalid definition for cgti strength')

    return dict(pos=pos_cgti.time,
                neg=neg_cgti.time)


def cgt_eofs(z200_ds):

    rot = 'None'
    pca_ = pca.SpatioTemporalPCA(z200_ds,
                                 var_name='an_dayofyear',
                                 n_components=10,
                                 rotation=rot)

    pca_dict = pca_.get_pca_loc_dict(q=None)
    return pca_dict


def cgt_pattern(z200):
    reload(sut)

    cgt_index = get_cgt_index(z200=z200, monthly=False)
    regressed_arr = sut.compute_correlation(
        data_array=z200, t_p=cgt_index['cgti'])

    return regressed_arr


# %%
if __name__ == '__main__':
    reload(cplt)
    data_dir = "/home/strnad/data/"
    lev = 200
    dataset_file = data_dir + \
        f"climate_data/1/era5_z_{1}_{lev}_ds.nc"

    ds_z200 = bds.BaseDataset(data_nc=dataset_file,
                              can=True,
                              an_types=['dayofyear', 'month', 'JJAS'],
                              )
    dataset_file = data_dir + \
        f"climate_data/2.5/era5_v_{2.5}_{lev}_ds.nc"
    ds_v200 = bds.BaseDataset(data_nc=dataset_file,
                              can=True,
                              an_types=['dayofyear', 'month', 'JJAS'],
                              )
    dataset_file = data_dir + \
        f"climate_data/2.5/era5_u_{2.5}_{lev}_ds.nc"
    ds_u200 = bds.BaseDataset(data_nc=dataset_file,
                              can=True,
                              an_types=['dayofyear', 'month', 'JJAS'],
                              )
    # %%
    an_type = 'dayofyear'
    var_type = f'an_{an_type}'
    ctype = 'pos'
    z200 = tu.get_month_range_data(ds_z200.ds[var_type], 'Jun',
                                   'Sep')
    v200 = tu.get_month_range_data(ds_v200.ds[var_type], 'Jun',
                                   'Sep')
    u200 = tu.get_month_range_data(ds_u200.ds[var_type], 'Jun',
                                   'Sep')

    cgt_tps = get_cgti_strength(z200=z200,)

    im = cplt.create_multi_plot(nrows=1, ncols=3,
                                title=f'Anomalies SRP ({ctype}, {200}hPa)',
                                projection='PlateCarree',
                                lon_range=[-50, 180],
                                lat_range=[-30, 70],
                                wspace=0.13)

    mean_tps_v = tu.get_sel_tps_ds(v200,
                                   cgt_tps[ctype].time).mean('time')
    vmax = 6
    vmin = -vmax

    im_comp = cplt.plot_map(mean_tps_v,
                            ax=im['ax'][0],
                            plot_type='contourf',
                            cmap='PuOr',
                            centercolor='white',
                            levels=12,
                            vmin=vmin, vmax=vmax,
                            title=f"V200 Anomalies",
                            label=rf'V-wind Anomalies {lev} hPa (wrt {an_type}) [m/s]',
                            )

    mean_tps_u = tu.get_sel_tps_ds(u200,
                                   cgt_tps[ctype].time).mean('time')
    vmax = 6
    vmin = -vmax

    im_comp = cplt.plot_map(mean_tps_u,
                            ax=im['ax'][1],
                            plot_type='contourf',
                            cmap='PuOr',
                            centercolor='white',
                            levels=12,
                            vmin=vmin, vmax=vmax,
                            title=f"U200 Anomalies",
                            label=rf'U-wind Anomalies {lev} hPa (wrt {an_type}) [m/s]',
                            )

    mean_tps = tu.get_sel_tps_ds(z200,
                                 cgt_tps[ctype].time).mean('time')
    vmax = 1.5e2
    vmin = -vmax
    im_comp = cplt.plot_map(mean_tps,
                            ax=im['ax'][2],
                            plot_type='contourf',
                            cmap='RdYlBu_r',
                            centercolor='white',
                            levels=12,
                            vmin=vmin, vmax=vmax,
                            title=f"Z200 Anomalies",
                            label=rf'Anomalies GPH (wrt {an_type}) [m]',
                            )
    dict_w = cplt.plot_wind_field(ax=im['ax'][2],
                                  u=mean_tps_u,
                                  v=mean_tps_v,
                                  #   u=mean_u,
                                  #   v=mean_v,
                                  scale=200,
                                  steps=3,
                                  key_length=5,
                                  )

    plot_dir = "/home/strnad/data/plots/cgti/"
    savepath = plot_dir + \
        f"definitions/z200_uv200_{an_type}_cgti_{ctype}.png"
    cplt.save_fig(savepath=savepath, fig=im['fig'])

    # %%
    # Climatology
    reload(cplt)
    an_type = 'dayofyear'
    var_type = f'an_{an_type}'
    ctype = 'pos'
    z200 = tu.get_month_range_data(ds_z200.ds, 'Jun',
                                   'Sep')
    v200 = tu.get_month_range_data(ds_v200.ds['v'], 'Jun',
                                   'Sep')
    u200 = tu.get_month_range_data(ds_u200.ds['u'], 'Jun',
                                   'Sep')

    cgt_tps = get_cgti_strength(z200=z200[var_type],)

    im = cplt.create_multi_plot(nrows=1, ncols=3,
                                projection='PlateCarree',
                                lon_range=[-50, 180],
                                lat_range=[-30, 70],
                                # central_longitude=80,
                                title=f'Climatology SRP ({ctype}, {200}hPa)',
                                wspace=0.13)

    mean_tps_v = tu.get_sel_tps_ds(v200,
                                   cgt_tps[ctype].time).mean('time')
    vmax = 15
    vmin = -vmax

    im_comp = cplt.plot_map(mean_tps_v,
                            ax=im['ax'][0],
                            plot_type='contourf',
                            cmap='RdBu',
                            levels=12,
                            vmin=vmin, vmax=vmax,
                            title=f"V200",
                            label=rf'V-winds {lev} hPa [m/s]',
                            )

    mean_tps_u = tu.get_sel_tps_ds(u200,
                                   cgt_tps[ctype].time).mean('time')
    vmax = 40
    vmin = -vmax

    im_comp = cplt.plot_map(mean_tps_u,
                            ax=im['ax'][1],
                            plot_type='contourf',
                            cmap='RdBu',
                            levels=12,
                            vmin=vmin, vmax=vmax,
                            title=f"U200 ",
                            label=rf'U-winds {lev} hPa [m/s]',
                            )

    mean_tps = tu.get_sel_tps_ds(z200['z'],
                                 cgt_tps[ctype].time).mean('time')
    vmax = 12.55e3
    vmin = 12.3e3
    im_comp = cplt.plot_map(mean_tps,
                            ax=im['ax'][2],
                            plot_type='contourf',
                            cmap='Blues',
                            levels=12,
                            vmin=vmin, vmax=vmax,
                            title=f"Z200 Anomalies",
                            label=rf'Anomalies GPH (wrt {an_type}) [m]',
                            )

    dict_w = cplt.plot_wind_field(ax=im['ax'][2],
                                  u=mean_tps_u,
                                  v=mean_tps_v,
                                  #   u=mean_u,
                                  #   v=mean_v,
                                  scale=1000,
                                  steps=3,
                                  key_length=10,
                                  )

    plot_dir = "/home/strnad/data/plots/cgti/"
    savepath = plot_dir + \
        f"definitions/climatology_z200_uv200_{an_type}_cgti_{ctype}.png"
    cplt.save_fig(savepath=savepath, fig=im['fig'])
