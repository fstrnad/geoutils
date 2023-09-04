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
# Compute the  tropical easterly jet (TEJ) index (TEJI) as the
# mean zonal winds U200 spatially averaged over the region 5-15°N, 10°W-20°E
# ======================================================================================


def get_tej_index(u200,
                  lon_range=[-0, 30],
                  lat_range=[5, 15]):
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
    box_tropics.name = 'tej'

    tej_idx = box_tropics.to_dataset()

    return tej_idx


def get_tej_strength(u200, tej_val=0, definition='std'):
    tej = get_tej_index(u200=u200)
    if definition == 'std':
        std = tej.std()
        pos_tej = tej.where(tej < -std, drop=True)
        neg_tej = tej.where(tej > std, drop=True)
    elif definition == 'thresh':
        pos_tej = tej.where(tej < -tej_val, drop=True)
        neg_tej = tej.where(tej > tej_val, drop=True)
    else:
        raise ValueError('Invalid definition for tej strength')

    return dict(pos=pos_tej.time,
                neg=neg_tej.time)


def tej_eofs(u200_ds):

    rot = 'None'
    pca_ = pca.SpatioTemporalPCA(u200_ds,
                                 var_name='an_dayofyear',
                                 n_components=10,
                                 rotation=rot)

    pca_dict = pca_.get_pca_loc_dict(q=None)
    return pca_dict


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
    lev = 200
    dataset_file = data_dir + \
        f"climate_data/1/era5_u_{1}_{lev}_ds.nc"

    ds_u200 = bds.BaseDataset(data_nc=dataset_file,
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
    dataset_file = data_dir + \
        f"climate_data/2.5/era5_z_{2.5}_{lev}_ds.nc"
    ds_z200 = bds.BaseDataset(data_nc=dataset_file,
                              can=True,
                              an_types=['dayofyear', 'month', 'JJAS'],
                              )
    # %%
    reload(cplt)
    an_type = 'month'
    var_type = f'an_{an_type}'
    ctype = 'pos'
    u200 = tu.get_month_range_data(ds_u200.ds[var_type], 'Jun',
                                   'Sep')
    v200 = tu.get_month_range_data(ds_v200.ds[var_type], 'Jun',
                                   'Sep')
    z200 = tu.get_month_range_data(ds_z200.ds[var_type], 'Jun',
                                   'Sep')

    tej_tps = get_tej_strength(u200=u200,)

    im = cplt.create_multi_plot(nrows=1, ncols=3,
                                title=f'Phases of the Tropical Easterly Jet  ({ctype}, {200}hPa)',
                                projection='PlateCarree',
                                lon_range=[-50, 180],
                                lat_range=[-30, 70],
                                wspace=0.13)

    mean_tps_u = tu.get_sel_tps_ds(u200,
                                   tej_tps[ctype].time).mean('time')
    vmax = 8
    vmin = -vmax

    im_comp = cplt.plot_map(mean_tps_u,
                            ax=im['ax'][0],
                            plot_type='contourf',
                            cmap='PuOr',
                            centercolor='white',
                            levels=12,
                            vmin=vmin, vmax=vmax,
                            title=f"U200 Anomalies",
                            label=rf'U-wind Anomalies {lev} hPa (wrt {an_type}) [m/s]',
                            )

    lon_range = [-0, 30]
    lat_range = [5, 15]
    cplt.plot_rectangle(ax=im['ax'][0],
                        lon_range=lon_range,
                        lat_range=lat_range,
                        lw=5,
                        color='magenta')

    mean_tps_v = tu.get_sel_tps_ds(v200,
                                   tej_tps[ctype].time).mean('time')
    vmax = 5
    vmin = -vmax

    im_comp = cplt.plot_map(mean_tps_v,
                            ax=im['ax'][1],
                            plot_type='contourf',
                            cmap='PuOr',
                            centercolor='white',
                            levels=12,
                            vmin=vmin, vmax=vmax,
                            title=f"V200 Anomalies",
                            label=rf'V-wind Anomalies {lev} hPa (wrt {an_type}) [m/s]',
                            )

    mean_tps = tu.get_sel_tps_ds(z200,
                                 tej_tps[ctype].time).mean('time')
    vmax = .7e2
    vmin = -vmax
    im_comp = cplt.plot_map(mean_tps,
                            ax=im['ax'][2],
                            plot_type='contourf',
                            cmap='RdYlBu_r',
                            centercolor='white',
                            levels=12,
                            vmin=vmin, vmax=vmax,
                            title=f"z200 Anomalies",
                            label=rf'Anomalies GPH (wrt {an_type}) [m]',
                            )
    dict_w = cplt.plot_wind_field(ax=im['ax'][2],
                                  u=mean_tps_u,
                                  v=mean_tps_v,
                                  #   u=mean_u,
                                  #   v=mean_v,
                                  scale=100,
                                  steps=3,
                                  key_length=5,
                                  )

    plot_dir = "/home/strnad/data/plots/tej/"
    savepath = plot_dir + \
        f"definitions/u200_uv200_{an_type}_tej_{ctype}.png"
    cplt.save_fig(savepath=savepath, fig=im['fig'])

    # %%
