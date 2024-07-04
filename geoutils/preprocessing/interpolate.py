import xarray as xr
import numpy as np
import geoutils.utils.general_utils as gut
import geoutils.preprocessing.open_nc_file as of

from importlib import reload
reload(gut)
reload(of)


def interpolate_grid(dataarray, grid_step,
                     method="nearest",
                     min_lon=None, max_lon=None,
                     min_lat=None, max_lat=None,
                     grid_step_lon=None,
                     grid_step_lat=None):
    """Common grid for all datasets.
    """
    if grid_step is None:
        raise ValueError("Grid step must be defined!")
    dataarray, dims = of.check_dimensions(dataarray,
                                          check_clim_dims=False,
                                          transpose_dims=False,
                                          verbose=False,
                                          )

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

    nlat = len(init_lat)
    if nlat % 2:
        # Odd number of latitudes includes the poles.
        gut.myprint(
            f"WARNING: Poles might be included: {min_lat} and {min_lat}!",
            color='red'
        )

    gut.myprint(
        f"Interpolte grid from {min(init_lon)} to {max(init_lon)},{min(init_lat)} to {max(init_lat)}!",
    )
    grid = {"lat": init_lat, "lon": init_lon}

    da = dataarray.interp(grid, method=method,
                          kwargs={"fill_value": "extrapolate"}
                          )  # Extrapolate if outside of the range
    return da
