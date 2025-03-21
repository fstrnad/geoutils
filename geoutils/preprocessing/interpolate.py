import xarray as xr
import numpy as np
import geoutils.utils.general_utils as gut
import geoutils.utils.spatial_utils as sput
import geoutils.preprocessing.open_nc_file as of

from importlib import reload
reload(gut)
reload(sput)
reload(of)


def interpolate_grid(dataarray, grid_step=None,
                     input_grid=None,
                     method="nearest",
                     min_lon=None, max_lon=None,
                     min_lat=None, max_lat=None,
                     grid_step_lon=None,
                     grid_step_lat=None,
                     use_esmf=True,):
    """Common grid for all datasets.
    """
    if grid_step is None and input_grid is None:
        raise ValueError("Either grid step or input_grid must be given!")
    if grid_step is not None and input_grid is not None:
        raise ValueError(
            "Either grid step or input_grid must be given, not both!")

    dataarray, dims = of.check_dimensions(dataarray,
                                          check_clim_dims=False,
                                          transpose_dims=False,
                                          verbose=False,
                                          )
    if input_grid is not None:
        init_lat = input_grid["lat"]
        init_lon = input_grid["lon"]
    else:
        native_grid_step, _, _ = sput.get_grid_step(dataarray)
        if native_grid_step == grid_step:
            gut.myprint(
                f"Native grid step is the same as the desired grid step: {grid_step}!")
            gut.myprint(f"Returning the same dataarray!")
            return dataarray

        if native_grid_step < grid_step:
            # coarse graining by next neighbor
            method = "nearest" if not use_esmf else 'nearest_s2d'

        init_lat, init_lon = generate_new_grid(
            dataarray, grid_step,
            min_lon, max_lon, min_lat, max_lat,
            grid_step_lon, grid_step_lat)

    if use_esmf:
        import xesmf as xe
        method = 'nearest_s2d' if method == 'nearest' else method
        grid = xr.Dataset(
            {
                "lat": (['lat'], init_lat, {"units": "degrees_north"}),
                "lon": (['lon'], init_lon, {"units": "degrees_east"}),
            }
        )
        gut.myprint(f"Using ESMF regridder with method {method}!")
        regridder = xe.Regridder(dataarray, grid,
                                 method=method,
                                 #    extrap_method="nearest_s2d",
                                 periodic=True)
        dr_out = regridder(dataarray,
                           skipna=True,   # keeps as many pixels as possible
                           na_thres=1.0,  # lowering this results in more NaNs in ds_out
                           keep_attrs=True)
    else:
        grid = {"lat": init_lat, "lon": init_lon}

        dr_out = dataarray.interp(coords=grid,
                                  method=method,
                                  kwargs={"fill_value": "extrapolate"}
                                  )  # Extrapolate if outside of the range

    # time dimension is not preserved
    # dr_out = gut.assign_new_coords(dr_out, 'time', times_in)

    return dr_out


def generate_new_grid(dataarray, grid_step,
                      min_lon, max_lon, min_lat, max_lat,
                      grid_step_lon, grid_step_lat):
    correct_max_lon = True if max_lon is None else False
    correct_min_lon = True if min_lon is None else False
    correct_min_lat = True if min_lat is None else False
    correct_max_lat = True if max_lat is None else False

    min_lat = float(
        np.min(dataarray["lat"])) if min_lat is None else min_lat
    min_lon = float(
        np.min(dataarray["lon"])) if min_lon is None else min_lon

    max_lat = float(
        np.max(dataarray["lat"])) if max_lat is None else max_lat
    max_lon = float(
        np.max(dataarray["lon"])) if max_lon is None else max_lon
    diff_lon = np.abs(max_lon - min_lon)
    if diff_lon-0.01 < 352:  # To avoid scenarios with big gap
        gut.myprint(f'WARNING: Max lon smaller than {180}!')
    if max_lon < 179 and max_lon > 175:  # To avoid scenarios with big gap
        gut.myprint(f'WARNING! Set max lon from {max_lon} to 179.75!')
        max_lon = 179.75 if correct_max_lon else max_lon
    if diff_lon > 352 and diff_lon < 360 and min_lon >= 0:
        gut.myprint(
            f'WARNING! Set max lon from {max_lon} to 359.75 and {min_lon} to 0!')
        min_lon = 0 if correct_min_lon else min_lon
        max_lon = 359.75 if correct_max_lon else max_lon

    if min_lon == -180 and max_lon == 180:  # To avoid scenarios with big gap
        gut.myprint(f'WARNING! Set min lon from {min_lon} to -179.75')
        min_lon = 179.75 if correct_min_lon else min_lon

    if max_lat < 89 and max_lat > 85:  # To avoid scenarios with big gap
        max_lat = 89.5 if correct_max_lat else max_lat
        if max_lat == 89.5:
            gut.myprint(
                f'WARNING! Set max lat from {max_lat} to 89.5!')

    if min_lat > -89 and min_lat < -85:  # To avoid scenarios with big gap
        gut.myprint(f'WARNING! Set min lat from {min_lat} to -89.5!')
        min_lat = -89.5 if correct_min_lat else min_lat

    grid_step_lon = grid_step if grid_step_lon is None else grid_step_lon
    grid_step_lat = grid_step if grid_step_lat is None else grid_step_lat

    init_lat = gut.custom_arange(start=min_lat,
                                 end=max_lat,
                                 step=grid_step_lat)
    init_lon = gut.custom_arange(start=min_lon,
                                 end=max_lon,
                                 step=grid_step_lon)
    gut.myprint(
        f"Interpolte grid from {min(init_lon)} to {max(init_lon)},{min(init_lat)} to {
            max(init_lat)}  with grid_step {grid_step}!",
    )

    nlat = len(init_lat)
    if nlat % 2:
        # Odd number of latitudes includes the poles.
        gut.myprint(
            f"WARNING: Poles might be included: {min_lat} and {min_lat}!",
            color='yellow'
        )

    return init_lat, init_lon


def coarse_spatial_grid(da, coarsen_factor, method="mean",
                        coords='nearest'):
    """
    Coarse an xarray DataArray by spatial averaging.

    Parameters:
    - da (xarray.DataArray): The input DataArray to coarsen.
    - coarsen_factor (int): The factor by which to coarsen the resolution (e.g., 4 for 32x32 -> 8x8).

    Returns:
    - xarray.DataArray: The coarsened DataArray.
    """
    # Ensure coarsen_factor is an integer and that the input DataArray has the expected dimensions
    if not isinstance(coarsen_factor, int):
        raise ValueError("Coarsen factor must be an integer")

    dims = gut.get_dims(da)
    if len(dims) < 2:
        raise ValueError(
            "Input DataArray must have at least two dimensions (lon and lat)")

    if coarsen_factor == 0:
        return da

    # Coarse the DataArray based on the coarsen_factor
    if method == "mean":
        coarsened_da = da.coarsen(
            lon=coarsen_factor, lat=coarsen_factor, boundary="trim",
            side='left').mean()
    elif method == "sum":
        coarsened_da = da.coarsen(
            lon=coarsen_factor, lat=coarsen_factor, boundary="trim").sum()
    elif method == 'median':
        coarsened_da = da.coarsen(
            lon=coarsen_factor, lat=coarsen_factor, boundary="trim").median()
    else:
        raise ValueError(
            "Invalid method; must be one of 'mean', 'sum', or 'median'")

    if coords == 'nearest':
        coarsened_da = coarsened_da.assign_coords(
            lon=da.lon[::coarsen_factor].values,
            lat=da.lat[::coarsen_factor].values
        )
    elif coords == 'center':
        coarsened_da = coarsened_da.assign_coords(
            lon=da.lon[coarsen_factor//2::coarsen_factor].values,
            lat=da.lat[coarsen_factor//2::coarsen_factor].values
        )
    else:
        raise ValueError(
            "Invalid coords; must be one of 'nearest' or 'center'")

    return coarsened_da
