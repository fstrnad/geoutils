# %%

import numpy as np
import xarray as xr
import geoutils.utils.statistic_utils as sut
import geoutils.utils.spatial_utils as sput
import geoutils.utils.general_utils as gut
import geoutils.utils.time_utils as tu
import geoutils.tsa.pca.rot_pca as rot_pca
import geoutils.plotting.plots as cplt
import geoutils.geodata.base_dataset as bds
import copy

from importlib import reload

reload(tu)
reload(cplt)
# ======================================================================================
# Compute the  Silk Road Pattern index (SRP) as the
# 1st EOF of v200 over the region 20-60°N, 30-130°E
# ======================================================================================


def get_srp_index(v200, var_name='an_dayofyear',
                  timemean=None, idx=0,
                  start_month='Jan', end_month='Dec',):
    """Returns the cgti index based on the 50hPa zonal winds dataset.

    Args:
        v200 (baseDataset): BaseDataset of meridional wind at 200hPa.
        monthly (boolean): Averages time dimensions to monthly.
            Default to False.
        time_range(list, optional): Select Nino indices only in a given time-range.
            Defauts to None

    Returns:
        cgti_index (xr.Dataset): Nino indices.
    """
    v200_srp = copy.deepcopy(v200)
    v200_srp.cut_map(lon_range=[30, 130],  # Kosaka et al. 2009
                     lat_range=[20, 60],
                     dateline=False,
                     set_ds=True)

    rot = 'None'
    pca_ = rot_pca.SpatioTemporalPCA(v200_srp,
                                 var_name=var_name,
                                 n_components=10,
                                 rotation=rot)

    pca_dict = pca_.get_pca_loc_dict(q=None)

    # Take first principal component (idx = 0)
    srp_index = pca_dict[idx]['ts']
    srp_pattern = pca_dict[idx]['map']
    srp_index = tu.get_month_range_data(srp_index,
                                        start_month=start_month,
                                        end_month=end_month)
    return {'index': srp_index,
            'map': srp_pattern, }


def get_srp_strength(v200, srp_val=0,
                     definition='std',
                     quantile=0.8,
                     start_month='Jan',
                     end_month='Dec'):
    """
    Calculate the strength of the Silk Road Pattern (SRP) based on the given parameters.

    Parameters:
    v200 (array-like): The v200 data.
    srp_val (float, optional): The threshold value for defining enhanced and reduced SRP. Default is 0.
    definition (str, optional): The definition method for calculating SRP strength. Options are 'std' and 'thresh'. Default is 'std'.
    start_month (str, optional): The start month for calculating SRP. Default is 'Jan'.
    end_month (str, optional): The end month for calculating SRP. Default is 'Dec'.

    Returns:
    dict: A dictionary containing the times of anomalous enhanced SRP and anomalous reduced SRP.
    """
    srp = get_srp_index(v200=v200, start_month=start_month,
                        end_month=end_month)['index']
    if definition == 'std':
        std = float(srp.std())
        mean = float(srp.mean())
        gut.myprint(f'mean {mean}, std: {std}')
        enhanced_srp = srp.where(srp >= mean-np.sqrt(std), drop=True)
        reduced_srp = srp.where(srp < mean+np.sqrt(std), drop=True)
    elif definition == 'thresh':
        enhanced_srp = srp.where(srp >= -srp_val, drop=True)
        reduced_srp = srp.where(srp < srp_val, drop=True)
    elif definition == 'quantile':
        gut.myprint(f'Get strength based on quantile: {quantile}')
        enhanced_srp = srp.where(srp >= srp.quantile(quantile), drop=True)
        reduced_srp = srp.where(srp < srp.quantile(1-quantile), drop=True)
    else:
        raise ValueError('Invalid definition for srp strength')

    gut.myprint(f'# anomalous enhanced srp times: {len(enhanced_srp.time)}')
    gut.myprint(f'# anomalous reduced srp times: {len(reduced_srp.time)}')

    return dict(enhanced=enhanced_srp.time,
                reduced=reduced_srp.time)


# %%
if __name__ == '__main__':
    data_dir = "/home/strnad/data/"
    plot_dir = "/home/strnad/data/plots/srp/"

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

    grid_step_z = 2.5
    dataset_file = data_dir + \
        f"climate_data/{grid_step_z}/era5_z_{grid_step_z}_{lev}_ds.nc"
    ds_z200 = bds.BaseDataset(data_nc=dataset_file,
                              can=True,
                              an_types=['dayofyear', 'month'],
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
    srp_index = get_srp_index(ds_v200,
                              timemean=None,
                              var_name='an_dayofyear',
                              )
    # %%
    srp_dict = get_srp_strength(ds_v200,
                                definition='quantile',
                                srp_val=1,
                                start_month='Jun',
                                end_month='Sep',
                                )
    # %%

    an_type = 'month'
    var_type = f'an_{an_type}'
    nrows = len(srp_dict)
    ncols = 3
    im = cplt.create_multi_plot(nrows=nrows, ncols=ncols,
                                projection='PlateCarree',
                                lon_range=[-20, 180],
                                lat_range=[-20, 70],
                                wspace=0.1,
                                hspace=0.8,
                                dateline=False)
    for idx, (srp_type, this_tps) in enumerate(srp_dict.items()):
        gut.myprint(f'Plotting {srp_type} {len(this_tps)} time steps')
        mean_tps_u, sig_u = tu.get_mean_tps(ds_u200.ds[f'an_{an_type}'],
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
                                vertical_title=f'{srp_type} srp',
                                )

        mean_tps_v, sig_v = tu.get_mean_tps(ds_v200.ds[f'an_{an_type}'],
                                            this_tps.time)
        vmax = 6
        vmin = -vmax

        im_comp = cplt.plot_map(mean_tps_v,
                                ax=im['ax'][idx*ncols + 1],
                                plot_type='contourf',
                                cmap='PuOr',
                                centercolor='white',
                                levels=12,
                                vmin=vmin, vmax=vmax,
                                title=f"V200 Anomalies",
                                label=rf'V-wind Anomalies {lev} hPa (wrt {an_type}) [m/s]',
                                )

        mean_tps, sig_z = tu.get_mean_tps(ds_z200.ds[f'an_{an_type}'],
                                          this_tps.time)
        vmax = 5.e2
        vmin = -vmax
        im_comp = cplt.plot_map(mean_tps,
                                ax=im['ax'][idx*ncols + 2],
                                plot_type='contourf',
                                cmap='RdYlBu_r',
                                centercolor='white',
                                levels=12,
                                vmin=vmin, vmax=vmax,
                                title=f"z200 Anomalies",
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

        savepath = plot_dir + \
            f"definitions/u200_uv200_{an_type}_srp_types.png"
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
                                lon_range=[0, -60],
                                dateline=True,
                                )
    vmin_sst = -1
    vmax_sst = -vmin_sst
    an_type = 'month'
    var_type = f'an_{an_type}'
    label_sst = f'SST Anomalies (wrt {an_type}) [°C]'
    for idx, (group, sel_tps) in enumerate(srp_dict.items()):

        mean, mask = tu.get_mean_tps(ds_sst.ds[var_type], tps=sel_tps)

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
        f"definitions/sst_{an_type}_srp_types.png"
    cplt.save_fig(savepath=savepath, fig=im['fig'])