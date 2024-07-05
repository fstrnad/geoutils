import numpy as np
import xarray as xr
import geoutils.utils.general_utils as gut
import geoutils.utils.statistic_utils as sut
import geoutils.utils.spatial_utils as sput
import geoutils.utils.time_utils as tu
from importlib import reload
reload(gut)
reload(sut)
reload(sput)


def map2flatten(x_map: xr.Dataset, reindex_z=False) -> list:
    """Flatten dataset/dataarray and remove NaNs.

    Args:
        x_map (xr.Dataset/ xr.DataArray): Dataset or DataArray to flatten.

    Returns:
        x_flat (xr.DataArray): Flattened dataarray without NaNs
        ids_notNaN (xr.DataArray): Boolean array where values are on the grid.
    """
    if isinstance(x_map, xr.core.dataset.Dataset):
        vars = gut.get_vars(ds=x_map)
        x_stack_vars = [x_map[var] for var in vars]
        x_stack_vars = xr.concat(x_stack_vars, dim='var')
        x_stack_vars = x_stack_vars.assign_coords(
            {'var': list(x_map.data_vars)})
        x_flatten = x_stack_vars.stack(z=('var', 'lat', 'lon'))
    else:
        x_flatten = sput.stack_lat_lon(x_map, reindex=reindex_z)

    # Flatten and remove NaNs
    if 'time' in x_flatten.dims:
        if len(x_flatten.time) > 1:
            idx_notNaN = ~np.isnan(x_flatten.isel(time=0))
        else:
            # single time point
            x_flatten.drop_dims('time')
            idx_notNaN = ~np.isnan(x_flatten)
    else:

        idx_notNaN = ~np.isnan(x_flatten)

    x_proc = x_flatten.isel(z=idx_notNaN.variable)

    return x_proc, idx_notNaN


def flattened2map(x_flat: np.ndarray, ids_notNaN: xr.DataArray, times: np.ndarray = None) -> xr.Dataset:
    """Transform flattened array without NaNs to gridded data with NaNs.

    Args:
        x_flat (np.ndarray): Flattened array of size (n_times, n_points) or (n_points).
        ids_notNaN (xr.DataArray): Boolean dataarray of size (n_points).
        times (np.ndarray): Time coordinate of xarray if x_flat has time dimension.

    Returns:
        xr.Dataset: Gridded data.
    """
    if len(x_flat.shape) == 1:
        x_map = xr.full_like(ids_notNaN, np.nan, dtype=float)
        x_map[ids_notNaN.data] = x_flat
    else:
        temp = np.ones((x_flat.shape[0], ids_notNaN.shape[0])) * np.nan
        temp[:, ids_notNaN.data] = x_flat
        if times is None:
            times = np.arange(x_flat.shape[0])
        x_map = xr.DataArray(data=temp, coords={
                             'time': times, 'z': ids_notNaN['z']})

    x_map = sput.unstack_lat_lon(x_map, dim='z')
    dims = gut.get_dims(x_map)
    if 'var' in dims:  # For xr.Datasset only
        da_list = [xr.DataArray(x_map.isel(var=i), name=var)
                    for i, var in enumerate(x_map['var'].data)]
        x_map = xr.merge(da_list, compat='override')
        x_map = x_map.drop(('var'))

    return x_map


def spatio_temporal_latent_volume(sppca, x_encode, tps=None, steps=0,
                                  ts=None,
                                  min_corr=0, num_eofs=None):
    """Generates a space-time volume of latent variables. The volume is a 2D array with a vector of steps*components for each time point.

    Args:
        sppca (object): spatio-temporal PCA object.
        x_encode (xr.Dataarray): input data.
        tps (xr.Dataarray, optional): time points. If not provided all time points of x_encode are used. Defaults to None.
        steps (int, optional): number of steps forward in time. Defaults to 0.

    Returns:
        np.ndarray: Numpy array of shape (n_time_points, n_components*steps)
    """
    if tps is None:
        tps = x_encode.time
    if tu.is_tp_smaller(date1=x_encode.time[-1], date2=tps.time[-1]):
        _, tps = tu.equalize_time_points(ts1=x_encode, ts2=tps, verbose=False)
        if ts is not None:
            _, ts = tu.equalize_time_points(
                ts1=x_encode, ts2=ts, verbose=False)
    if ts is not None:
        sel_eofs = get_reduced_eofs(x=x_encode, sppca=sppca, ts=ts,
                                    num_eofs=num_eofs, min_corr=min_corr)
    else:
        sel_eofs = sppca.get_principal_components().eof.data[:num_eofs]
    gut.myprint(f'Select eofs {sel_eofs}!')
    if steps < 1:
        x_encode = tu.get_sel_tps_ds(ds=x_encode, tps=tps)
        data = sppca.transform_reduced(x=x_encode, reduzed_eofs=sel_eofs)
        num_steps = 1
    else:
        steps_arr = np.arange(1, steps+1)
        num_steps = len(steps_arr)
        data = np.array([])
        for idx, step in enumerate(steps_arr):
            this_tps = tu.add_time_step_tps(tps=tps, time_step=step)
            x_encode_step = tu.get_sel_tps_ds(ds=x_encode, tps=this_tps)
            this_data = sppca.transform_reduced(
                x=x_encode_step, reduzed_eofs=sel_eofs)
            data = np.append(data, this_data, axis=1) if idx > 0 else this_data

    n_components = len(sel_eofs)

    z_events = xr.DataArray(
        data=data,
        coords={'time': tps.time,
                'eof': np.arange(0, (n_components*num_steps))}
    )

    return z_events


def get_max_correlation(pcs, x, corr_type='pearson', return_corrs=False):
    corr_function = sut.corr_function(corr_type=corr_type)
    max_corr = []
    eof_nums = pcs.eof.data
    for pc in pcs.eof:
        ts = pcs.sel(eof=pc)
        ts, x_ts = tu.equalize_time_points(ts1=ts, ts2=x, verbose=False)
        corr = corr_function(ts, x_ts)[0]  # select only the correlation value
        max_corr.append(corr)
    max_corr = np.abs(max_corr)
    if return_corrs:
        return gut.reverse_array(eof_nums[np.argsort(max_corr)]), gut.reverse_array(np.sort(max_corr))
    return gut.reverse_array(eof_nums[np.argsort(max_corr)])


def get_reduced_eofs(x, sppca,
                     ts,
                     num_eofs=None, min_corr=0):
    eof_nums = sppca.get_eof_nums()
    if len(eof_nums) < num_eofs:
        raise ValueError(
            f'Number of EOFs {len(eof_nums)} is smaller than selected num_eofs {num_eofs}')
    if ts is None:
        return eof_nums
    else:
        eof_nums, max_corr = get_max_correlation(
            sppca.get_principal_components(), x=ts, return_corrs=True)
        sel_eofs_corr = eof_nums[max_corr >
                                 min_corr] if min_corr > 0 else eof_nums
        sel_eofs_num = eof_nums[:num_eofs] if num_eofs is not None else eof_nums
        # Get ids that are maximal num_eofs and at least correlation min_corr
        sel_eofs = np.intersect1d(sel_eofs_corr, sel_eofs_num)
        sel_corr_vals = max_corr[:len(sel_eofs)]
        gut.myprint(f'Reduce to eofs {sel_eofs} with corr {sel_corr_vals}!')
        return sel_eofs


