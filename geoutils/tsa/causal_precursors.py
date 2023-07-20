import xarray as xr
import numpy as np
import geoutils.plotting.plots as cplt
import geoutils.utils.time_utils as tu
import geoutils.utils.spatial_utils as sput
import geoutils.utils.general_utils as gut
import copy
from importlib import reload
reload(tu)
reload(sput)
reload(cplt)


def get_corr_precursors(ds, var, target_ts, lag=-1,
                        timemean=None,
                        plevel=None,
                        min_num_locations=15,
                        plot=True,
                        savepath=None,):

    if lag > 0:
        raise ValueError(f'Lag must be negative, but is {lag}!')

    ds_corr = copy.deepcopy(ds)
    if plevel is None:
        ds_arr, t_ = tu.equalize_time_points(
            ds_corr.ds[var], target_ts)
    else:
        ds_arr, t_ = tu.equalize_time_points(
            ds_corr.ds[var].sel(lev=plevel), target_ts)
    gut.myprint(f'Compute correlation between {var} and target_ts')
    corr_ = sput.compute_correlation(
        data_array=tu.compute_timemean(
            ds_arr, timemean=timemean, dropna=True),
        ts=tu.compute_timemean(t_, timemean=timemean,
                               dropna=True),
        lag_arr=np.array([lag]),
        correlation_type='spearman')

    gut.myprint(f'Compute significant correlation regions in dataset')
    lag_corr_mask = xr.where(corr_['p'].sel(lag=lag) < 0.01,
                             1,
                             0)
    _ = ds_corr.init_mask(init_mask=True,
                          mask_ds=lag_corr_mask)
    lagged_sign_data = ds_corr.get_data_spatially_seperated_regions(
        var=var,
        min_num_locations=min_num_locations,)
    if plevel is not None:
        for idx, data in enumerate(lagged_sign_data):
            lagged_sign_data[idx]['data'] = data['data'].sel(lev=plevel)

    if plot:
        vmax = .25
        vmin = -vmax
        im = cplt.plot_map(
            corr_['corr'].sel(lag=lag),
            significance_mask=corr_['p'].sel(lag=lag) < 0.01,
            sig_plot_type='contour',
            projection='Robinson',
            cmap='RdBu',
            levels=14,
            plot_type='contourf',
            label=f'Correlation',
            vertical_title=f'{var} Lag={lag} {timemean}',
            vmin=vmin,
            vmax=vmax,
            orientation='vertical',
            round_dec=2,
        )
        if savepath is not None:
            cplt.save_fig(savepath, fig=im['fig'])
        im = None
        for group in lagged_sign_data:
            im = cplt.plot_map(
                np.array(gut.zip_2_lists(group['data']['lon'],
                                         group['data']['lat'])),
                ax=im['ax'] if im is not None else None,
                z=None,
                title='Locations of significant correlation',
                projection='Robinson' if im is None else None,
                cmap='Reds',
                plot_type='points',
                orientation='vertical',
                size=2,
                round_dec=2,
            )

    del ds_corr
    return lagged_sign_data


def get_variable_dict(lagged_data, var, lag=-1):
    variable_dict = {}
    for idx, spatial_area in enumerate(lagged_data):
        this_data = spatial_area['data']
        av_ts = this_data.mean(dim='ids')
        locs = gut.zip_2_lists(
            spatial_area['data'].lon, spatial_area['data'].lat)
        mean_loc = sput.get_mean_std_loc(locs)
        variable_dict[f'{var}_{idx}_{lag}'] = dict(
            ts=av_ts,
            loc=mean_loc,
            idx=idx,
            data=this_data,
        )

    return variable_dict


def get_significant_target_links(q_matrix, pcmci, pc_alpha):
    var_names = np.array(pcmci.var_names)
    idx_target = np.where(var_names == 'target')[0][0]
    sig_links = (q_matrix <= pc_alpha)
    sig_links_names = var_names[sig_links[:, idx_target, -1]]
    idx_sig_links_target = np.where(sig_links_names == 'target')[0]
    sig_links_names = np.delete(sig_links_names, idx_sig_links_target)
    gut.myprint(f'Significant links: {sig_links_names}')
    return sig_links_names
