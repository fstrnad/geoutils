import numpy as np
import xarray as xr
import scipy as sp
import math
from tqdm import tqdm
from importlib import reload
import geoutils.utils.time_utils as tu
import geoutils.utils.spatial_utils as sput
import geoutils.utils.general_utils as gut
reload(gut)
reload(tu)
reload(sput)


def get_day_progression_arr(data, tps, start,
                            sps=None, eps=None,
                            var=None, end=None,
                            q=None, step=1,
                            average_ts=False,
                            verbose=False):
    """Gets a day progression for a xr.Dataset for specific time points.

    Args:
        ds (xr.DataSet): dataset of variables
        tps (xr.dataarray): dataarray that contains time points
        start (int): how many time points before to start
        end (int, optional): how many time points to end. Defaults to None.
        step (int, optional): step of progression. Defaults to 1.
        average_ts(bool, optional): Takes all days between two steps into account and averages over .

    Returns:
        dict: dictionary that contains xr.Dataarrays of the means
    """
    composite_arrs = dict()
    s_step = 1
    e_step = -1
    if end is None:
        end = start
    if sps is None:
        sps = tps
        s_step = 0
    if eps is None:
        eps = tps
        e_step = 0

    days = np.arange(-start, end+step, step)
    composite_arrs = []
    for thisstep in days:
        if thisstep < 0:
            this_tps = tu.add_time_step_tps(
                sps.time, time_step=thisstep + s_step)
        elif thisstep > 0:
            this_tps = tu.add_time_step_tps(
                eps.time, time_step=thisstep + e_step)
        else:
            this_tps = tps.time  # day 0

        if average_ts and thisstep != 0:
            # the sign is because average is for the preceeding periode
            # signum of thisstep
            av_step = step * -1 * math.copysign(1, thisstep)
            this_tps = tu.get_periods_tps(tps=this_tps, step=av_step)

        this_comp_ts = tu.get_sel_tps_ds(ds=data, tps=this_tps)
        if var == 'evs':
            this_comp_ts = xr.where(
                this_comp_ts[var] == 1, this_comp_ts[var], np.nan
            )
            mean_ts = this_comp_ts.sum(dim='time')
        elif var is not None:
            if q is None:
                mean_ts = this_comp_ts[var].mean(dim='time')
            else:
                mean_ts = this_comp_ts[var].quantile(q=q,
                                                     dim='time')
        else:
            mean_ts = this_comp_ts.mean(dim='time')

        mean_ts = mean_ts.expand_dims(
            {'day': 1}).assign_coords({'day': [thisstep]})

        composite_arrs.append(mean_ts)

    gut.myprint(
        'Merge selected composite days into 1 xr.DataSet...', verbose=verbose)
    composite_arrs = xr.merge(composite_arrs)

    return composite_arrs


def get_day_arr(ds, tps):

    composite_arrs = []
    for day, tp in enumerate(tps):
        tp_ds = tu.get_sel_tps_ds(ds, tps=[tp]).mean(dim='time')
        tp_ds = tp_ds.expand_dims(
            {'day': 1}).assign_coords({'day': [day]})
        composite_arrs.append(tp_ds)

    gut.myprint('Merge selected composite days into 1 xr.DataSet...')
    composite_arrs = xr.merge(composite_arrs)

    return composite_arrs


def get_hovmoeller(ds, tps, sps=None, eps=None, num_days=0,
                   start=1,
                   var=None, step=1,
                   lat_range=None, lon_range=None,
                   zonal=True,
                   dateline=False):
    reload(sput)
    if num_days > 0:
        composite_arrs = get_day_progression_arr(ds=ds,
                                                 tps=tps,
                                                 sps=sps, eps=eps,
                                                 start=start,
                                                 end=num_days,
                                                 step=step,
                                                 var=var,
                                                 )
    else:
        composite_arrs = get_day_arr(ds=ds, tps=tps)
    composite_arrs = sput.cut_map(ds=composite_arrs,
                                  lon_range=lon_range,
                                  lat_range=lat_range,
                                  dateline=dateline)
    if zonal:
        hov_means = sput.compute_zonal_mean(ds=composite_arrs)
    else:
        hov_means = sput.compute_meridional_mean(ds=composite_arrs)

    return hov_means


def get_hovmoeller_single_tps(ds, tps, num_days,
                              start=1,
                              var=None, step=1,
                              lat_range=None, lon_range=None,
                              zonal=True,
                              gf=(0, 0),
                              dateline=False):
    hov_data = []
    if gf[0] != 0 or gf[1] != 0:
        gut.myprint(f'Apply Gaussian Filter with sigma = {gf}!')
        sigma = [gf[1], gf[0]]  # sigma_y, sigma_x

    for tp in tps:
        this_hov_data = get_hovmoeller(ds=ds, tps=tp,
                                       num_days=num_days,
                                       start=start,
                                       var=var, step=step,
                                       lat_range=lat_range,
                                       lon_range=lon_range,
                                       zonal=zonal, dateline=dateline)
        if gf[0] != 0 or gf[1] != 0:
            tmp_data = sp.ndimage.filters.gaussian_filter(
                this_hov_data[var].data, sigma, mode='constant')
            this_hov_data = xr.DataArray(data=tmp_data,
                                         dims=this_hov_data.dims,
                                         coords=this_hov_data.coords)
        hov_data.append(this_hov_data)
    single_hov_dates = xr.concat(hov_data, tps)

    return single_hov_dates


def get_box_propagation(ds, loc_dict, tps,
                        sps=None, eps=None,
                        num_days=1, regions=None,
                        normalize=True,
                        var='evs', step=1,
                        q=0.9,
                        q_prog=None,
                        norm_grid_fac=2):  # four borders
    reload(sput)
    coll_data = dict()
    if regions is None:
        regions = list(loc_dict.keys())
    for region in tqdm(regions):
        # EE TS
        if 'points' in list(loc_dict[region]['data'].dims):
            pids = loc_dict[region]['pids']
            pr_data = ds.sel(points=pids)
        else:
            pr_data = loc_dict[region]['data']
        # pr_data = loc_dict[region]['data']
        print(pr_data[var].data.shape)
        composite_arrs = get_day_progression_arr(data=pr_data,
                                                 tps=tps,
                                                 sps=sps, eps=eps,
                                                 start=num_days,
                                                 end=num_days,
                                                 step=step,
                                                 var=var,
                                                 q=q_prog
                                                 )
        coll_data[region] = composite_arrs[var]

    days = composite_arrs.day.data
    box_data = np.zeros((len(regions), len(days)))
    for i, region in enumerate(regions):
        this_data = coll_data[region]
        gut.myprint(region)

        if normalize:
            data = loc_dict[region]['data']
            tot_num_days = len(tps)
            num_cells = len(data.points) if 'points' in data.dims else len(
                data.lon)*len(data.lat)
            # num_cells = 10  # Get Results per 100 cells
            # norm = num_cells * tot_num_days
            norm = tot_num_days*num_cells / \
                (norm_grid_fac*100)  # Get Results per 100 cells
        else:
            norm = 1
        for j, day in enumerate(days):
            if var != 'evs':
                if q is not None:
                    box_data[i][j] = float(this_data.sel(day=day).quantile(
                        dim=['lon', 'lat'],
                        q=q).data)
                else:
                    box_data[i][j] = float(this_data.sel(day=day).mean(
                        dim=['lon', 'lat']).data)
            else:
                this_box_data = float(this_data.sel(
                    day=day).sum(dim='points').data) / norm
                box_data[i][j] = this_box_data
    box_data = gut.mk_grid_array(data=box_data,
                                 x_coords=days,
                                 y_coords=regions,)

    return box_data


def get_quantile_progression_arr(ds, tps, start,
                                 sps=None, eps=None,
                                 var=None, end=None,
                                 q=None, step=1,
                                 average_ts=False,
                                 verbose=False,
                                 q_th=0.05,
                                 th=None):
    progression_arr = get_day_progression_arr(
        ds=ds, tps=tps,
        start=start,
        sps=sps, eps=eps,
        var=var, end=end,
        q=q, step=step,
        average_ts=average_ts,
        verbose=verbose)
    day_arr = xr.zeros_like(progression_arr.sel(day=0))
    day_arr = xr.where(day_arr == 1, 0, np.nan)
    for idx, (day) in enumerate(progression_arr.day):
        day = int(day)
        mean_ts = progression_arr.sel(day=day)
        th_mask = xr.ones_like(mean_ts)
        if th is not None:
            th_mask = xr.where(mean_ts <= th, 1, 0) if th < 0 else xr.where(
                mean_ts >= th, 1, 0)
        q_mask = xr.ones_like(mean_ts)
        if q_th is not None:
            q_val = mean_ts.quantile(q=q_th)
            q_mask = xr.where(mean_ts <= q_val, 1, 0) if q_th < 0.5 else xr.where(
                mean_ts > q_val, 1, 0)
        mask = q_mask * th_mask

        # This overwrites old values
        day_arr = xr.where(mask, day, day_arr)

    return day_arr