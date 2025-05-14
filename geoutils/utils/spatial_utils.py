from typing import List, Tuple
from sklearn.neighbors import KernelDensity
import scipy.stats as st
import numpy as np
import xarray as xr
import scipy.interpolate as interp
import geoutils.utils.time_utils as tu
import geoutils.utils.general_utils as gut
import geoutils.utils.statistic_utils as sut
from importlib import reload
reload(tu)
reload(sut)
reload(gut)

RADIUS_EARTH = 6371  # radius of earth in km


def assert_has_grid_dims(da):
    """
    Assert that a given xarray DataArray has a lon-lat dimension.

    Parameters
    ----------
    da : xarray.DataArray
        The input DataArray to check.

    Raises
    ------
    ValueError
        If the input DataArray does not have a lon-lat dimension.
    """

    if 'lon' not in da.dims or 'lat' not in da.dims:
        raise ValueError(
            f"The input DataArray '{da.name}' does not have a time dimension.")


def def_sel_ids_ds(ds, ids):
    """Returns an dataset of climnet dataset class with the selected ids.
    note that ids are transformed to points first.

    Args:
        ds (climnet.dataset): dataset class
        ids (list): list of int ids

    Returns:
        ds: dataset of climnet class of selected points.
    """
    pids = ds.get_points_for_idx(ids)
    ds_sel = ds.ds.sel(points=pids)
    return ds_sel


def get_mean_std_loc(locs):
    lons = locs[:, 0]
    lats = locs[:, 1]
    std_lon = np.std(lons)
    std_lat = np.std(lats)
    mean_lon = np.mean(lons)
    mean_lat = np.mean(lats)

    return {'mean_lon': mean_lon,
            'mean_lat': mean_lat,
            'mean_loc': np.array([mean_lon, mean_lat]),
            'std_lon': std_lon,
            'std_lat': std_lat}


def location_stats(locations):
    """
    A function that takes a list of latitude, longitude pairs and returns
    a dictionary containing the mean, standard deviation, and quantile values
    from 0.05 to 0.95 in steps of 0.05 for both the latitudes and longitudes.

    Args:
    - locations: list of tuples, where each tuple contains a longitude,latitude pair

    Returns:
    - dict with the following keys:
        - lat_mean : float, mean latitude
        - lon_mean : float, mean longitude
        - lat_std : float, standard deviation of latitudes
        - lon_std : float, standard deviation of longitudes
        - lat_qX : float, single quantile value for X percentile, where X is a two-digit integer representing the percentile (e.g., 'lat_q90' for the 90th percentile of latitudes)
        - lon_qX : float, single quantile value for X percentile, where X is a two-digit integer representing the percentile (e.g., 'lon_q90' for the 90th percentile of longitudes)
    """

    lats = [loc[1] for loc in locations]
    lons = [loc[0] for loc in locations]
    lat_mean = np.mean(lats)
    lon_mean = np.mean(lons)
    lat_std = np.std(lats)
    lon_std = np.std(lons)

    # Create key-value pairs for quantiles
    lat_quantiles = {}
    lon_quantiles = {}
    quantile_values = np.arange(0.05, 1, 0.05)
    for q in quantile_values:
        # Convert quantile values to integers for use in dictionary keys
        q_int = int(q * 100)
        lat_quantiles[f"lat_q{q_int}"] = np.quantile(lats, q=q)
        lon_quantiles[f"lon_q{q_int}"] = np.quantile(lons, q=q)

    return {'lat_mean': lat_mean, 'lon_mean': lon_mean,
            'lat_std': lat_std, 'lon_std': lon_std,
            **lat_quantiles, **lon_quantiles}


def get_mean_lon_lat_range(lon_range, lat_range):
    return np.mean(lon_range), np.mean(lat_range)


def in_lon_lat_range(lon, lat, lon_range, lat_range, dateline=False):
    """Check if lon, lat is in lon_range and lat_range

    Args:
        lon (float): longitude
        lat (float): latitude
        lon_range (list): list of longitude range
        lat_range (list): list of latitude range

    Returns:
        boolean: True if lon, lat is in lon_range and lat_range
    """
    if dateline is False:
        if lon_range[1] < lon_range[0]:
            lon_range = lon_range[::-1]
        lon_in_range = (lon >= lon_range[0]) & (lon <= lon_range[1])
    else:
        lon_range = lon2_360(lon_range)
        lon_in_range = (lon >= lon_range[0]) | (lon <= lon_range[1])

    if lat_range[0] < lat_range[1]:
        lat_in_range = (lat >= lat_range[0]) & (lat <= lat_range[1])
    else:
        lat_in_range = (lat >= lat_range[0]) | (lat <= lat_range[1])

    return lon_in_range & lat_in_range


def compute_rm(da, rm_val, dim='time', sm='Jan', em='Dec'):
    """This function computes a rolling mean on an input
    xarray dataarray. The first and last rm_val elements
    of the time series are deleted.py

    Args:
        da (xr.dataarray): dataarray of the input dataset
        rm_val (int): size of the rolling mean

    Returns:
        [type]: [description]
    """
    reload(tu)
    times = da.time
    if dim != 'time':
        da_rm = da.rolling(time=rm_val, center=True).mean(
            skipna=True).dropna(dim, how='all')
    else:
        start_year, end_year = tu.get_sy_ey_time(times)
        all_year_arr = []
        for idx, year in enumerate(np.arange(start_year, end_year)):
            start_date, end_date = tu.get_start_end_date(
                sy=year,
                ey=year,
                sm=sm,
                em=em)
            arr_1_year = da.sel(time=slice(start_date, end_date))
            arr_1_year_rm = arr_1_year.rolling(
                time=rm_val, center=True, min_periods=1).mean(skipna=True).dropna(dim=dim,
                                                                                  how='all')
            all_year_arr.append(arr_1_year_rm)
        da_rm_ds = xr.merge(all_year_arr)

        for name, da in da_rm_ds.data_vars.items():
            var_name = name
        da_rm = da_rm_ds[var_name]

    return da_rm


def get_map_for_def_map(data_map, def_map):
    if data_map.shape[1:] != def_map.shape[:]:
        raise ValueError("Error! Not same shape of def map and data map!")
    nan_map = xr.where(def_map > 0, data_map, np.nan)
    return nan_map


def create_new_coordinates(times, lons, lats):
    """Generates an xr.Dataarray coordinates.

    Args:
        times (np.datatimearray): np array of datetimes
        lons (list): list of longitudes.
        lats (list): list of latitudes.

    Returns:
        dict: dictionary of coordinates.
    """
    coordinates = dict(time=times,
                       points=np.arange(0, len(lons), 1),
                       lon=("points", lons),
                       lat=("points", lats))
    return coordinates


def interp_fib2gaus(dataarray, grid_step=2.5):
    """Interpolate dataarray on Fibonacci grid to Gaussian grid.

    Args:
        dataarray (xr.DataArray): Dataarray with fibonacci grid
        grid_step (float, optional): Grid step of new Gaussian grid. Defaults to 2.5.

    Returns:
        dataarray_gaus: Dataarray interpolated on Gaussian grid.
    """
    # Create gaussian grid
    lon_gaus = np.arange(np.round_(dataarray.coords['lon'].min()),
                         dataarray.coords['lon'].max(),
                         grid_step)
    lat_gaus = np.arange(np.round_(dataarray.coords['lat'].min()),
                         dataarray.coords['lat'].max(),
                         grid_step)
    lon_mesh, lat_mesh = np.meshgrid(lon_gaus, lat_gaus)
    new_points = np.array([lon_mesh.flatten(), lat_mesh.flatten()]).T
    origin_points = np.array(
        [dataarray.coords['lon'], dataarray.coords['lat']]).T

    # Interpolate
    if len(dataarray.data.shape) < 2:  # If there is not time dimension
        origin_values = dataarray.data.flatten()
        assert len(origin_values) == origin_points.shape[0]
        new_values = interp.griddata(origin_points, origin_values, new_points,
                                     method='nearest')
        new_data = new_values.reshape(len(lat_gaus), len(lon_gaus))
        coordinates = dict(lon=lon_gaus, lat=lat_gaus)
        dims = ['lat', 'lon']
    else:  # with time dimension
        new_data = []
        for idx, t in enumerate(dataarray.time):
            origin_values = dataarray.sel(time=t.data).data.flatten()
            assert len(origin_values) == origin_points.shape[0]
            new_values = interp.griddata(origin_points, origin_values,
                                         new_points,
                                         method='nearest')
            new_data.append(
                new_values.reshape(len(lat_gaus), len(lon_gaus))
            )
        coordinates = dict(time=dataarray.time.data,
                           lon=lon_gaus, lat=lat_gaus)
        dims = ['time', 'lat', 'lon']
        new_data = np.array(new_data)

    da_gaus = xr.DataArray(
        data=new_data,
        dims=dims,
        coords=coordinates,
        name=dataarray.name)

    return da_gaus


def interp2gaus(dataarray, grid_step=2.5):
    """Interpolate dataarray on arbitrary grid to Gaussian grid.

    Args:
        dataarray (xr.DataArray): Dataarray with fibonacci grid
        grid_step (float, optional): Grid step of new Gaussian grid. Defaults to 2.5.

    Returns:
        dataarray_gaus: Dataarray interpolated on Gaussian grid.
    """
    min_lat = float(np.min(dataarray["lat"]))
    min_lon = float(np.min(dataarray["lon"]))

    max_lat = float(np.max(dataarray["lat"]))
    max_lon = float(np.max(dataarray["lon"]))
    if np.abs(180 - max_lon)-0.01 > grid_step:  # To avoid scenarios with big gap
        max_lon = 180

    # Create gaussian grid
    lon_gaus = gut.crange(min_lon, max_lon, grid_step)
    lat_gaus = gut.crange(min_lat, max_lat, grid_step)

    lon_mesh, lat_mesh = np.meshgrid(lon_gaus, lat_gaus)
    new_points = np.array([lon_mesh.flatten(), lat_mesh.flatten()]).T
    origin_points = np.array(
        [dataarray.coords["lon"], dataarray.coords["lat"]]).T

    # Interpolate
    if len(dataarray.data.shape) < 2:  # If there is not time dimension
        origin_values = dataarray.data.flatten()
        assert len(origin_values) == origin_points.shape[0]
        new_values = interp.griddata(origin_points, origin_values, new_points,
                                     method='nearest')
        new_data = new_values.reshape(len(lat_gaus), len(lon_gaus))
        coordinates = dict(lon=lon_gaus, lat=lat_gaus)
        dims = ['lat', 'lon']
    else:  # with time dimension
        new_data = []
        for idx, t in enumerate(dataarray.time):
            origin_values = dataarray.sel(time=t.data).data.flatten()
            assert len(origin_values) == origin_points.shape[0]
            new_values = interp.griddata(origin_points, origin_values,
                                         new_points,
                                         method='nearest')
            new_data.append(
                new_values.reshape(len(lat_gaus), len(lon_gaus))
            )
        coordinates = dict(time=dataarray.time.data,
                           lon=lon_gaus, lat=lat_gaus)
        dims = ['time', 'lat', 'lon']
        new_data = np.array(new_data)

    da_gaus = xr.DataArray(
        data=new_data,
        dims=dims,
        coords=coordinates,
        name=dataarray.name)

    return {'intpol': da_gaus,
            'origin_points': origin_points,
            'new_points': new_points,
            'lon_gaus': lon_gaus,
            'lat_gaus': lat_gaus,
            }


def cut_map(ds, lon_range=[-180, 180],
            lat_range=[-90, 90],
            dateline=False,
            lon_name='lon',
            lat_name='lat',
            verbose=False):
    """
    Works only for rectangular data!
    Cut an area in the map. Use always smallest range as default.
    It lon ranges accounts for regions (eg. Pacific) that are around the -180/180 region.

    Args:
    ----------
    lon_range: list [min, max]
        range of longitudes
    lat_range: list [min, max]
        range of latitudes
    dateline: boolean
        use dateline range in longitude (eg. -170, 170 range) contains all points from
        170-180, -180- -170, not all between -170 and 170. Default is False.
    Return:
    -------
    ds_area: xr.dataset
        Dataset cut to range
    """
    if lon_range is None:
        lon_range = [-180, 180]
    if lat_range is None:
        lat_range = [-90, 90]

    gut.myprint(f'Cut map to lon: {lon_range} lat: {lat_range}',
                verbose=verbose)

    if dateline:
        # To account for areas that lay at the border of -180 to 180
        # ds_cut = ds.sel(
        #     lon=ds.lon[(ds.lon < np.min(lon_range)) |
        #                (ds.lon > np.max(lon_range))],
        #     lat=slice(np.min(lat_range), np.max(lat_range))
        # )
        lon_range = lon2_360(lon_range)
        lon_range = slice(np.min(lon_range), np.max(lon_range))
        # requires trafo of coordinates to 0-360
        da = ds.assign_coords(lon=(((ds.lon) % 360)))
        da = da.roll(lon=int(len(da['lon']) / 2), roll_coords=True)
        # requires for plotting as well gl_plt=False and central_longitude=180

    else:
        da = ds
        lon_range = slice(np.min(lon_range), np.max(lon_range))

    lats = da.lat
    if lats[0] < lats[1]:
        ds_cut = da.sel(
            {lon_name: lon_range,
                lat_name: slice(np.min(lat_range), np.max(lat_range))},
        )
    else:
        ds_cut = da.sel({
            lon_name: lon_range,
            lat_name: slice(np.max(lat_range), np.min(lat_range))},
        )

    return ds_cut


def compute_zonal_mean(ds, ):
    zonal_mean = ds.mean(dim='lat', skipna=True)

    return zonal_mean


def compute_meridional_mean(ds, return_std=False):
    lats_rad = np.deg2rad(ds['lat'])
    weighted_meridional_mean = ds.mean(
        dim='lon', skipna=True) * np.cos(lats_rad)

    if return_std:
        weighted_meridional_std = ds.std(
            dim='lon', skipna=True)
        return weighted_meridional_mean, weighted_meridional_std
    else:
        return weighted_meridional_mean


def compute_meridional_quantile(ds, q):
    lats_rad = np.deg2rad(ds['lat'])
    weighted_meridional_mean = ds.quantile(
        q=q, dim='lon', skipna=True)  # * np.cos(lats_rad)

    return weighted_meridional_mean


def horizontal_average(ds, dim='lon', average_type='mean'):
    if dim not in ds.dims:
        raise ValueError(f'Dimension {dim} not in dataset!')
    if average_type == 'mean':
        return ds.mean(dim=dim, skipna=True)
    elif average_type == 'median':
        return ds.median(dim=dim, skipna=True)
    elif average_type == 'std':
        return ds.std(dim=dim, skipna=True)
    elif average_type == 'max':
        ds = ds.max(dim=dim)
    else:
        raise ValueError(
            f'Average type {average_type} not implemented!')
    return ds


def get_vertical_ds(wind_dict, tps,
                    vname='v',
                    wname='w',
                    vtype='zonal',
                    lon_range=[-180, 180],
                    lat_range=[-90, 90]):
    v_comp = []
    w_comp = []
    plevels = list(wind_dict.keys())
    # get_longitudinal averages
    for plevel, ds in wind_dict.items():
        composites = tu.get_sel_tps_ds(ds=ds, tps=tps).mean(dim='time')
        composites = cut_map(composites,
                             lon_range=lon_range,
                             lat_range=lat_range)
        if vtype == 'zonal':
            l_means, _ = compute_meridional_mean(composites)
            x_coord_name = 'lat'
        elif vtype == 'meridional':
            l_means = compute_zonal_mean(composites)
            x_coord_name = 'lon'

        v_comp.append(l_means[vname])
        w_comp.append(l_means[wname])
    if vtype == 'zonal':
        lon_lats = composites['lat']
    elif vtype == 'meridional':
        lon_lats = composites['lon']

    xr_v = gut.mk_grid_array(data=v_comp,
                             x_coords=lon_lats, y_coords=plevels,
                             x_coord_name=x_coord_name,
                             y_coord_name='plevel',
                             name=vname)

    xr_w = gut.mk_grid_array(data=w_comp,
                             x_coords=lon_lats, y_coords=plevels,
                             x_coord_name=x_coord_name,
                             y_coord_name='plevel',
                             name=wname)
    xr_vw = xr.merge([xr_v, xr_w])
    return xr_vw


def get_map4indices(ds, indices):
    da = xr.ones_like(ds.ds['anomalies']) * np.NAN
    da[:, indices] = ds.ds['anomalies'].sel(points=indices).data

    return da


def get_locations_in_range(def_map,
                           lon_range=[-180, 180],
                           lat_range=[-90, 90],
                           dateline=False):
    """Returns a map with nans at the positions that are not within lon_range-lat_range

    Args:
        def_map (xr.DataArray): dataarray, must contain lat and lon definition
        lon_range (list): list of lons
        lat_range (list): list of lats

    Returns:
        xr.DataArray: masked xr.DataArray.
    """
    if not dateline:
        if np.abs(np.max(lon_range) - np.min(lon_range)) > 180 and lon_range != [-180, 180]:
            gut.myprint(
                f'WARNING! Range larger 180° {lon_range} but not dateline!')
        mask = (
            (def_map['lat'] >= np.min(lat_range))
            & (def_map['lat'] <= np.max(lat_range))
            & (def_map['lon'] >= np.min(lon_range))
            & (def_map['lon'] <= np.max(lon_range))
        )
    else:   # To account for areas that lay at the border of -180 to 180
        mask = (
            (def_map['lat'] >= np.min(lat_range))
            & (def_map['lat'] <= np.max(lat_range))
            & ((def_map['lon'] <= np.min(lon_range)) | (def_map['lon'] >= np.max(lon_range)))
        )
    mmap = xr.where(mask, def_map, np.nan)
    return mmap


def extract_subregion(da: xr.DataArray,
                      lat_range: tuple[float, float],
                      lon_range: tuple[float, float]) -> xr.DataArray:
    """
    Extracts a subregion of an xr.DataArray based on given latitude and longitude ranges.
    Args:
        da (xr.DataArray): input xr.DataArray that needs to be subset.
        lat_range (Tuple[float, float]): latitude range of the desired subregion.
        lon_range (Tuple[float, float]): longitude range of the desired subregion.
    Returns:
        xr.DataArray: returns the subregion as xr.DataArray
    """
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range
    lat_sub = da.lat.where((da.lat >= lat_min) &
                           (da.lat <= lat_max), drop=True)
    lon_sub = da.longitude.where(
        (da.lon >= lon_min) & (da.lon <= lon_max), drop=True)
    da_sub = da.sel(lat=lat_sub, lon=lon_sub)
    return da_sub


def get_ts_in_range(ds,
                    lon_range=[-180, 180],
                    lat_range=[-90, 90],
                    dateline=False):
    loc_map = get_locations_in_range(def_map=ds,
                                     lon_range=lon_range,
                                     lat_range=lat_range,
                                     dateline=dateline)
    dims = gut.get_dims(ds)
    if 'points' in dims:
        return loc_map.dropna(dim='points')
    elif 'lat' in dims:
        lat_sub = loc_map.dropna(dim='lat').lat
        lon_sub = loc_map.dropna(dim='lon').lon
        ds_sub = ds.sel(lat=lat_sub, lon=lon_sub)

        return ds_sub
    else:
        raise ValueError(
            f'Dimensions have to contain points or lat-lon but are {dims}')


def gdistance(pt1, pt2, radius=RADIUS_EARTH):
    """Distance between two points pt1 and pt2 given as tuple (lon1,lat1) and (lon2,lat2)

    Args:
        pt1 (tuple): tuple of floats.
        pt2 (tuple): tuple of floats.
        radius (float, optional): Radius. Defaults to RADIUS_EARTH.

    Returns:
        float: distance between pt1 and pt2.
    """
    lon1, lat1 = pt1[1], pt1[0]
    lon2, lat2 = pt2[1], pt2[0]
    return haversine(lon1=lon1, lat1=lat1, lon2=lon2, lat2=lat2)


@np.vectorize
def haversine(lon1, lat1, lon2, lat2, radius=1):
    """Computes spatial distance between two points, given as (lon1,lat1) and
    (lon2,lat2).

    Args:
        lon1 (float): longitude of point 1.
        lat1 (float): latitude of point1.
        lon2 (float): longitude of point 2
        lat2 (float): latitude of point 2.
        radius (float, optional): radius. Defaults to 1.

    Returns:
        float: distance between points.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * \
        np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))


def degree2distance_equator(grid_step, radius=RADIUS_EARTH):
    """Get distance between grid_step in km"""
    distance = haversine(0, 0, grid_step, 0, radius=radius)
    return distance


def neighbor_distance(lon, lat, radius=RADIUS_EARTH):
    """Distance between next-nearest neighbor points on a sphere.
    Args:
    -----
    lon: np.ndarray
        longitudes of the grid points
    lat: np.ndarray
        latitude values of the grid points

    Return:
    -------
    Array of next-nearest neighbor points
    """
    distances = []
    for i in range(len(lon)):
        d = haversine(lon[i], lat[i], lon, lat, radius)
        neighbor_d = np.sort(d)
        distances.append(neighbor_d[1:2])

    return np.array(distances)


def far_locations(locations: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    A function that takes a list of longitude-latitude pairs and returns
    only those locations that have at least two other locations in the list that
    are closer than in total 2.5 degrees.

    Args:
    - locations: list of tuples, where each tuple contains a longitude-latitude pair

    Returns:
    - a list of tuples, containing only the longitude-latitude pairs of locations that meet the criteria.
    """

    min_dist = degree2distance_equator(2.5)

    def is_close(loc1: Tuple[float, float], loc2: Tuple[float, float]) -> bool:
        """Check whether two locations are closer than 2.5 degrees."""
        return haversine(loc1[0], loc1[1], loc2[0], loc2[0], radius=RADIUS_EARTH) <= min_dist

    # Filter out locations that don't have at least two locations within 2.5 degrees
    far_locs = []
    for loc1 in locations:
        close_locs = [loc2 for loc2 in locations if is_close(loc1, loc2)]
        if len(close_locs) < 3:  # Add loc1 to far_locs if it has less than two close locations
            far_locs.append(loc1)

    return far_locs


def find_nearest_lat_lon(lon, lat, lon_arr, lat_arr):
    """Find nearest lon-lat position for array of lon arr and lat arr

    Args:
        lon ([float]): lon searched
        lat ([float]): lat searched
        lon_arr ([type]): longitude array
        lat_arr ([type]): latitude array

    Returns:
        lon, lat, idx:
    """
    if len(lon_arr) != len(lat_arr):
        raise ValueError("Lon-array and lat array not of same size!")
    dist_arr = haversine(lon, lat, np.array(lon_arr),
                         np.array(lat_arr), radius=1)
    idx = np.argmin(dist_arr)  # point index, not considering mask!
    return float(lon_arr[idx]), float(lat_arr[idx]), idx


def cut_lon_lat(lon, lat, lon_range, lat_range):
    min_lon = np.min(lon_range)
    max_lon = np.max(lon_range)
    min_lat = np.min(lat_range)
    max_lat = np.max(lat_range)
    if min_lon > -180 or max_lon < 180 or min_lat > -90 or max_lat < 90:
        print("WARNING: Cut Map!")
        idx_lon = np.where((lon > min_lon) & (lon < max_lon))
        idx_lat = np.where((lat > min_lat) & (lat < max_lat))[0]
        intersect_lat_lon = np.intersect1d(idx_lon, idx_lat)
        lon = lon[intersect_lat_lon]
        lat = lat[intersect_lat_lon]

    return lon, lat


def get_nn_bw(nn_points=1, grid_step=1):
    dist_eq = degree2distance_equator(grid_step,
                                      radius=RADIUS_EARTH)
    bandwidth = nn_points * dist_eq/RADIUS_EARTH
    return bandwidth


def spherical_kde(link_points, coord_rad, bw_opt=None):
    """
    Inspired from https://science.nu/amne/in-depth-kernel-density-estimation/
    Because the coordinate system here lies on a spherical surface rather than a flat plane, we will use the haversine distance metric,
       which will correctly represent distances on a curved surface.

    Parameters
    ----------
    link_points: np.array (num_links, 2)
        List of latidude and longitudes.
    coord_rad : array
        Array of all locations provided as [lon, lat]
    bw_opt: float
        bandwidth of the kde, used Scott rule here

    Returns
    -------
    Link density estimation

    """
    assert link_points.shape[1] == 2
    # Do KDE fit by using haversine metric that accounts for spherical coordinates
    num_links = len(link_points)
    if num_links <= 2:
        scott_factor = 0.2
        # Scott's rule of thumb (compare Boers et al. 2019)
        if bw_opt is None:
            bw_opt = scott_factor * num_links**(-1./(2+4))

        kde = KernelDensity(metric='haversine', kernel='gaussian',
                            algorithm='ball_tree', bandwidth=bw_opt)
        kde.fit(link_points)
        Z = np.exp(kde.score_samples(coord_rad))
    else:
        # Use scipy version because it automatically selects the bandwidth
        kde = st.gaussian_kde(link_points.T, bw_method=bw_opt)
        Z = np.exp(kde.evaluate(coord_rad.T))
    return Z


def get_kde_map(ds, data, coord_rad=None, nn_points=None, bandwidth='scott'):
    if nn_points is not None:
        bandwidth = get_nn_bw(nn_points=nn_points, grid_step=ds.grid_step)

    if coord_rad is None:
        coord_deg, coord_rad, map_idx = ds.get_coordinates_flatten()

    link_points = ds.get_idx_point_lst(np.where(data > 0)[0])

    links_rad = coord_rad[link_points]
    Z_kde = spherical_kde(link_points=links_rad,
                          coord_rad=coord_rad,
                          bw_opt=bandwidth)

    return ds.get_map(Z_kde)


def get_mask_for_nan_array(arr, var_name=None):

    if isinstance(arr, xr.Dataset):
        if var_name is None:
            var_names = gut.get_varnames_ds(arr)
            var_name = var_names[0]
            if len(var_names) > 1:
                gut.myprint(f'WARNING: More than one variable in dataset! Select first variable {var_name}!',
                            color='yellow')
        arr = arr[var_name]

    if gut.contains_nan(arr):
        time_dim = tu.get_time_dim(arr)
        mask = xr.where(arr.mean(dim=time_dim) == np.nan, 1, 0)
    else:
        mask = xr.ones_like(arr[0])
    return mask


def compute_correlation(data_array, ts,
                        correlation_type='spearman',
                        lag_arr=None, prev_lags=True,
                        multi_lag=True):
    """
    Compute the Pearson or Spearman correlation between a time series t_p and all time series in an xarray DataArray.

    Parameters
    ----------
    data_array: xarray.DataArray
        A DataArray with dimensions (lon, lat, time).
    ts: array-like
        The time series to compute the correlation with.
    correlation_type: str
        The type of correlation to be computed. Can be either 'pearson' or 'spearman'.

    Returns
    -------
    xarray.DataArray
        A DataArray with dimensions (lon, lat) containing the correlation values between t_p and the time series in data_array.
    """
    data_array, ts = tu.equalize_time_points(ts1=data_array, ts2=ts)
    # check whether the datarray is of dimension (lon, lat, time)
    if len(data_array.dims) != 3:
        raise ValueError(
            f"The data_array must have dimensions (lon, lat, time). But has dimensions {data_array.dims}.")

    lag_arr = [0] if lag_arr is None else lag_arr
    if multi_lag:
        lag_arr = gut.make_arr_negative(arr=lag_arr) if prev_lags else gut.add_compliment(
            lag_arr)  # Only use past lags to target time series
    corr_array = xr.DataArray(np.zeros(
        (data_array.lon.size, data_array.lat.size, len(lag_arr))),
        dims=("lon", "lat", "lag"),
        name='corr')
    p_array = xr.DataArray(np.zeros(
        (data_array.lon.size, data_array.lat.size, len(lag_arr))),
        dims=("lon", "lat", "lag"),
        name='p')
    gut.myprint(
        f"Computing {correlation_type} correlation for {len(data_array.lon)} locations ...")
    for i, lon in enumerate(data_array.lon):
        for j, lat in enumerate(data_array.lat):
            for l, lag in enumerate(lag_arr):
                time_series_loc = data_array.sel(
                    lon=lon, lat=lat).values.flatten()
                ts_1_lag, ts_2_lag = tu.get_lagged_ts(ts1=ts,   # the lags are with respect to the target time series
                                                      ts2=time_series_loc,
                                                      lag=lag)
                if correlation_type == 'pearson':
                    corr, p = st.pearsonr(ts_1_lag, ts_2_lag)
                elif correlation_type == 'spearman':
                    corr, p = st.spearmanr(ts_1_lag, ts_2_lag)
                else:
                    raise ValueError(
                        f"Invalid correlation_type: {correlation_type}. Must be either 'pearson' or 'spearman'.")
                corr_array[i, j, l] = corr
                p_array[i, j, l] = p
    corr_array.coords["lon"] = data_array.lon
    corr_array.coords["lat"] = data_array.lat
    corr_array.coords["lag"] = lag_arr
    da_corr = corr_array.to_dataset()
    p_array_corr = sut.correct_p_values(pvals=p_array.data,
                                        method='fdr_bh')
    p_array_corr = xr.DataArray(
        p_array_corr,
        dims=("lon", "lat", "lag"),
        name='p')
    da_corr['p'] = p_array
    da_corr = da_corr.transpose().compute()
    return da_corr


def get_quantile_map(ds, val_map):
    return (ds < val_map).mean(dim='time')


def get_val_array(da, val, var=None):
    """Gets the value of an xarray dataarry

    Args:
        da (xr.DataArray): Dataarray

    Returns:
        xr.DataArray: dataarray that contains the maximum and its coordinates
    """
    val_data = da.where(da == val, drop=True).squeeze()
    if var is not None:
        val_data = val_data[var]
    return val_data


def get_q_array(da, dim='time', var=None, q=0.5):
    """Gets the maximum of an xarray dataarry

    Args:
        da (xr.DataArray): Dataarray

    Returns:
        xr.DataArray: dataarray that contains the maximum and its coordinates
    """
    if dim is None:
        q_data = da.where(da == da.quantile(q=q), drop=True).squeeze()
    else:
        if var is not None:
            q_data = []
            for tp in da.time:
                da_tmp = da.sel(time=tp)
                q_data.append(da_tmp.where(da_tmp == da_tmp.quantile(q=q), drop=True).squeeze()[
                    var].data)  # To only get the data along the remaining dimension
            q_data = xr.DataArray(
                data=q_data,
                dims=['time'],
                coords=dict(
                    time=da.time
                ),
                name=var
            )
        else:
            q_data = da.quantile(
                q=q,
                dim=dim,
                skipna=True)

    return q_data


def get_max_array(da, dim=None, var=None):
    """Gets the maximum of an xarray dataarry

    Args:
        da (xr.DataArray): Dataarray

    Returns:
        xr.DataArray: dataarray that contains the maximum and its coordinates
    """
    if dim is None:
        max_data = da.where(da == da.max(), drop=True).squeeze()
    else:
        if var is not None:
            max_data = []
            for tp in da.time:
                da_tmp = da.sel(time=tp)
                max_data.append(da_tmp.where(da_tmp == da_tmp.max(), drop=True).squeeze()[
                                var].data)  # To only get the data along the remaining dimension
            max_data = xr.DataArray(
                data=max_data,
                dims=['time'],
                coords=dict(
                    time=da.time
                ),
                name=var
            )
        else:
            max_data = da.where(da == da.max(
                dim=dim, skipna=True), drop=True).squeeze()

    return max_data


def get_min_array(da):
    """Gets the maximum of an xarray dataarry

    Args:
        da (xr.DataArray): Dataarray

    Returns:
        xr.DataArray: dataarray that contains the minimum and its coordinates
    """
    max_data = da.where(da == da.min(), drop=True).squeeze()
    return max_data


def get_vals_above_th(da, th):
    """Gets the locations and time points of a xr.Dataarray of multiple dimensions.

    Args:
        da (xr.DataArray): Input dataarry
        th (float): threshold for above values

    Returns:
        dict: dict of dimensions and values.60
    """
    dims = list(da.dims)
    coords = da.coords
    idx_dims = np.where(da.data > th)

    res_dict = {}
    for idx, dim in enumerate(dims):
        # choose the particular indices
        res_dict[dim] = coords[dim][idx_dims[idx]]

    return res_dict


def lon2_360(lon1):
    lon1 = np.array(lon1)
    lon1[lon1 < 0] = lon1[lon1 < 0] + 360
    return lon1


def lon2_180(lon3):
    lon1 = (lon3 + 180) % 360 - 180
    return lon1


def get_lat_lon_for_value(da: xr.DataArray, value: float):
    """
    Get the latitudes and longitudes for a specific value in an xarray DataArray.

    Args:
    da (xr.DataArray): The xarray DataArray containing data with 'lat' and 'lon' coordinates.
    specific_value (float): The value for which to find the corresponding latitude and longitude.

    Returns:
    list of tuples: A list of (latitude, longitude) tuples where the value is located.
    """
    # Find the indices where the data matches the specific value
    matching_indices = np.where(da.values == value)

    # Get the corresponding lat/lon values
    matching_lats = np.array(da.coords['lat'][matching_indices[0]])
    matching_lons = np.array(da.coords['lon'][matching_indices[1]])

    # Combine the lat and lon values into a list of tuples
    locations = list(zip(matching_lons, matching_lats))

    return locations


def sum_eres(data):
    """Gives sum of EREs per point over the whole dataset provided.

    Args:
        data (xr.DataArray): Dataarray of climnet dataset

    Returns:
        float: average number of expteded EREs per point.
    """
    num_eres = data.sum(dim=['points', 'time'])
    return num_eres


def average_num_eres(data):
    """Gives the average number of EREs per point over the whole dataset provided.

    Args:
        data (xr.DataArray): Dataarray of climnet dataset

    Returns:
        float: average number of expteded EREs per point.
    """
    N = len(data.points)
    num_eres = sum_eres(data=data)
    return num_eres/N


valid_dims = ['time', 'lat', 'lon', 'plevel', 'lev', 'plev',
              'level', 'hour', 'dayofyear',
              'dimx_lon', 'dimy_lon', 'dimz_lon', 'x',
              'Year', 'year', 'month', 'day', 'points']


def check_var_dims(ds, validate_dims=False, rm_var_dims=False):
    if isinstance(ds, xr.Dataset):
        # Remove useless dimensions in ds for all variables
        dims = gut.get_dims(ds=ds)
        if 'expver' in dims:
            # occurs for ERA5 data sometimes
            # see https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation
            gut.myprint(f'Select expver=1 (ERA5) from ds!')
            ds = ds.sel(expver=1)  # .combine_first(ds.sel(expver=5))
            # ds.load()
            gut.myprint(f'Combined expver 1 and 5 for ds!')
            dims = gut.get_dims(ds=ds)

        if validate_dims:
            for dim in dims:
                if dim not in valid_dims:
                    ds = ds.drop_dims(dim)
                    gut.myprint(f'Remove dimension {dim}!')

        # Remove useless variables
        if rm_var_dims:
            vars = gut.get_vars(ds=ds)
            for var in vars:
                this_dims = gut.get_dims(ds[var])
                if (
                    (gut.compare_lists(this_dims, ['lat', 'lon', 'time'])) or
                    (gut.compare_lists(this_dims, ['lat', 'lon'])) or
                    (gut.compare_lists(this_dims, ['lat', 'lon', 'time', 'plevel'])) or
                    (gut.compare_lists(this_dims, ['lat', 'lon', 'time', 'lev'])) or
                    (gut.compare_lists(this_dims, ['lat', 'lon', 'lev'])) or
                    (gut.compare_lists(this_dims, ['lat', 'lon', 'plevel'])) or
                    (gut.compare_lists(this_dims, ['time', 'dimx_lon'])) or
                    (gut.compare_lists(this_dims, ['time', 'dimy_lon'])) or
                    (gut.compare_lists(this_dims, ['time', 'dimz_lon'])) or
                    (gut.compare_lists(this_dims, ['time', 'x']))
                ):
                    continue
                else:
                    ds = ds.drop(var)
                    gut.myprint(
                        f'Remove variable {var} with dims: {this_dims}!')

    return ds


def remove_single_dim(ds):
    if isinstance(ds, xr.Dataset) or isinstance(ds, xr.DataArray):
        dims = dict(ds.sizes)  # new in xarray 2023.06.
        for dim, num in dims.items():
            if num < 2:
                # Time and pressure level dimension is the only one that is allowed to be kept
                if dim not in ['time', 'lev', 'plevel', 'points']:
                    ds = ds.mean(dim=dim)  # Removes the single variable axis
                    gut.myprint(f'Remove single value dimension {dim}!')
                else:
                    gut.myprint(
                        f'WARNING: dimension {dim} contains only 1 value!')
    return ds


def da_lon2_180(da):
    return da.assign_coords(lon=(((da.lon + 180) % 360) - 180))


def da_lon2_360(da):
    da = da.assign_coords(lon=(((da.lon) % 360)))

    return da


def transpose_3D_data(da, dims=['lat', 'lon', 'time']):
    da_dims = list(da.dims)
    if not gut.compare_lists(da_dims, dims):
        gut.myprint(f'Transpose data from {da_dims} to {dims} not possible!')
        raise ValueError('Dimensions are not the same!')
    else:
        da = da.transpose(dims[0], dims[1], dims[2]).compute()
    return da


def transpose_2D_data(da, dims=['lat', 'lon']):
    da_dims = list(da.dims)
    if not gut.compare_lists(da_dims, dims):
        gut.myprint(f'Transpose data from {da_dims} to {dims} not possible!')
        raise ValueError('Dimensions are not the same!')
    else:
        da = da.transpose(dims[0], dims[1]).compute()
    return da


def check_dim_coords(ds):
    """Checks whether all dimensions have coordinates.
    If not, add these to coordinates.

    Args:
        ds (xr.Dataset): xarray dataset
    """
    dims = set(gut.get_dims(ds))
    coords = set(ds.coords)
    dims_without_coords = dims - coords
    if len(dims_without_coords) > 0:
        for dim in dims_without_coords:
            if dim in ds.dims:
                ds = ds.assign_coords({dim: np.arange(ds.sizes[dim])})
                gut.myprint(f'Added coordinate for dimension {dim}!')
    return ds


def check_dimensions(ds, datetime_ts=True,
                     sort=True,  # necessary when transforming from 0-360 to -180-180
                     lon360=False,
                     lon_2_180=True,
                     keep_time=False,
                     freq='D',
                     transpose_dims=False,
                     hours_to_zero=False,
                     set_netcdf_encoding=False,
                     check_vars=False,
                     validate_dims=True,
                     verbose=False):
    """
    Checks whether the dimensions are the correct ones for xarray!
    """
    reload(tu)
    gut.myprint('Check dimensions of dataset!', verbose=verbose)
    ds = rename_dims(ds=ds, verbose=verbose)
    ds = remove_single_dim(ds=ds)
    if check_vars:
        ds = check_var_dims(ds=ds, validate_dims=validate_dims)
    dims = list(ds.dims)
    numdims = len(dims)
    ds = check_dim_coords(ds=ds)
    gut.myprint(
        f'Checked labelling according to netcdf conventions!', verbose=verbose)
    dims = gut.get_dims(ds)
    if transpose_dims:
        gut.myprint(f'Transpose dimensions {dims}', verbose=verbose)
        dims4 = ['lev', 'time', 'lat', 'lon']
        dims3 = ['time', 'lat', 'lon']
        dims2 = ['lat', 'lon']
        if numdims == 4 and gut.are_arrays_equal(dims, dims4):
            # Actually change location in memory if necessary!
            ds = ds.transpose(*dims4).compute()
            gut.myprint('object transposed to lev-time-lat-lon!',
                        verbose=verbose)
        elif numdims == 3 and gut.are_arrays_equal(dims, dims3):
            # Actually change location in memory if necessary!
            ds = ds.transpose(*dims3).compute()
            gut.myprint('3d object transposed to time-lat-lon!',
                        verbose=verbose)
        elif numdims == 2 and gut.are_arrays_equal(dims, dims2):
            if 'lon' in dims:
                ds = ds.transpose(*dims2).compute()
            gut.myprint('2d oject transposed to lat-lon!', verbose=verbose)
        elif numdims == 1:
            gut.myprint('1d oject only!', verbose=verbose)

    if numdims >= 2 and 'lon' in dims:
        # If lon from 0 to 360 shift to -180 to 180
        if lon360:
            gut.myprint('Longitudes 0 - 360!', verbose=verbose)
            if max(ds.lon) < 180:
                ds = da_lon2_360(da=ds)
        else:
            if max(ds.lon) > 180 and lon_2_180:
                gut.myprint("Shift longitude -180 - 180!", verbose=verbose)
                ds = da_lon2_180(da=ds)

        if sort:
            ds = ds.sortby('lon')
            ds = ds.sortby('lat')
            gut.myprint(
                'Sorted longitudes and latitudes in ascending order!',
                verbose=verbose)

    if 'time' in dims:
        # create a time index deepending if a datetime time series is provided or not
        if datetime_ts:
            if gut.is_datetime360(time=ds.time.data[0]) or keep_time:
                ds = ds
            else:
                # Sets to equal hours. If days sets hours to 0
                if set_netcdf_encoding:
                    ds = tu.get_netcdf_encoding(ds=ds,
                                                calendar='gregorian',
                                                verbose=verbose,
                                                hours_to_zero=hours_to_zero,
                                                )
        else:
            time_ds = ds.time
            num_steps = len(time_ds)
            calendar = '365_day'
            # Default start year set to 1900: https://docs.xarray.dev/en/stable/user-guide/time-series.html
            # dates = np.array(tu.get_dates_for_time_steps(start='0001-01-01',
            #                                              num_steps=num_steps,
            #                                              freq=freq),
            #                  dtype="datetime64[D]")
            # times = xr.DataArray(
            #     data=np.arange(num_steps),
            #     dims=['time'],
            #     coords={'time': dates}
            #     )
            # units = 'days since 0001-01-01 00:00'
            # times = times.convert_calendar(calendar='365_day', use_cftime=True)
            if freq == 'M':
                freq = '1MS'  # to start with month
            cfdates = xr.cftime_range(start='0001-01-01',
                                      periods=num_steps,
                                      freq=freq,
                                      calendar=calendar,
                                      )
            ds = ds.assign_coords(
                time=cfdates)

            ds.time.attrs.pop('calendar', None)
            # ds.time.attrs.update({'calendar': '365_day'})
            ds.time.encoding['calendar'] = calendar

    return ds


def rename_dims(ds, verbose=True):
    rename_dict = {
        'longitude': 'lon',
        'latitude': 'lat',
        't': 'time',
        'valid_time': 'time',
        'month': 'time',
        'time_counter': 'time',
        'AR_key': 'time',
        'plevel': 'lev',
        'dimx_lon': 'x',
        'dimy_lon': 'y',
        'dimz_lon': 'z',
        'x': 'lon',
        'y': 'lat',
        'Lon': 'lon',
        'Lat': 'lat',
        'plev': 'lev',
    }
    lon_lat_names = list(rename_dict.keys())

    dims = list(ds.dims)
    for idx, lon_lat in enumerate(lon_lat_names):
        if lon_lat in dims:
            gut.myprint(
                f'Rename:{lon_lat} : {rename_dict[lon_lat]}', verbose=verbose)
            coords_lon_lat = ds.coords[lon_lat]
            if rename_dict[lon_lat] in coords_lon_lat.coords:
                gut.myprint(
                    f'WARNING: {rename_dict[lon_lat]} already in coordinates!', verbose=verbose)
                ds = ds.drop(rename_dict[lon_lat])
            ds = ds.rename({lon_lat: rename_dict[lon_lat]})
            dims = list(ds.dims)
            gut.myprint(dims, verbose=verbose)
    return ds


def long_dimension_names(ds):
    dims = gut.get_dims(ds)
    xdims = ['lon', 'x']
    ydims = ['lat', 'y']
    for dim in dims:
        if dim in xdims:
            ds = ds.rename({dim: 'longitude'})
        elif dim in ydims:
            ds = ds.rename({dim: 'latitude'})
    return ds


def get_lon_range(ds, decimals=3):
    lon = ds.lon
    lon_range = [float(lon.min()), float(lon.max())]
    return np.around(lon_range, decimals=decimals)


def get_lat_range(ds, decimals=3):
    lat = ds.lat
    lat_range = [float(lat.min()), float(lat.max())]
    return np.around(lat_range, decimals=decimals)


def get_lon_lat_range(ds, decimals=3):
    lon = get_lon_range(ds, decimals=decimals)
    lat = get_lat_range(ds, decimals=decimals)
    return lon, lat


def check_full_globe_coverage(data_array: xr.DataArray,
                              max_deviation=10) -> bool:
    """
    Check whether the longitude range in an xarray DataArray covers the whole globe (360°)
    or only part of it.

    Parameters:
    - data_array (xr.DataArray): An xarray DataArray containing a longitude dimension.

    Returns:
    - str: A message indicating whether the longitude range covers the whole globe or only parts.
    """
    if 'lon' not in data_array.dims and 'longitude' not in data_array.dims:
        raise ValueError(
            "The DataArray must have a 'lon' or 'longitude' dimension.")

    # Identify the longitude dimension
    lon_dim = 'lon' if 'lon' in data_array.dims else 'longitude'
    lon_values = np.array(data_array[lon_dim].values)
    # Normalize longitudes to the range [0, 360)
    normalized_lons_360 = np.sort(lon2_360(lon_values))
    normalized_lons_180 = np.sort(lon2_180(lon_values))
    # Check the range
    lon_range_360 = normalized_lons_360[-1] - normalized_lons_360[0]
    lon_range_180 = normalized_lons_180[-1] - normalized_lons_180[0]
    if lon_range_360 >= 360-max_deviation and lon_range_180 >= 360-max_deviation:
        return True

    else:
        return False


def check_if_crosses_dateline(data_array: xr.DataArray) -> bool:
    """
    Check whether the longitude values in an xarray DataArray cross the dateline.

    Parameters:
    - data_array (xr.DataArray): An xarray DataArray containing a longitude dimension.

    Returns:
    - bool: True if the longitude values cross the dateline, False otherwise.
    """
    if 'lon' not in data_array.dims and 'longitude' not in data_array.dims:
        raise ValueError(
            "The DataArray must have a 'lon' or 'longitude' dimension.")

    # Identify the longitude dimension
    lon_dim = 'lon' if 'lon' in data_array.dims else 'longitude'
    lon_values = data_array[lon_dim].values

    # Normalize longitudes to the range [0, 360)
    normalized_lons = lon2_180(lon_values)

    # Check for dateline crossing
    diff = np.diff(normalized_lons)
    if np.any(np.abs(diff) > 180):  # A gap larger than 180° indicates crossing the dateline
        return True

    return False


def get_grid_range(da):
    assert_has_grid_dims(da=da)
    xmin = float(np.min(da.lon))
    xmax = float(np.max(da.lon))
    ymin = float(np.min(da.lat))
    ymax = float(np.max(da.lat))

    return xmin, xmax, ymin, ymax


def get_grid_step(ds, verbose=True):
    """
    Calculates the grid step (distance between grid points) in longitude and latitude direction
    for an xarray dataarray or dataset object.

    Parameters:
    -----------
    ds: xarray dataarray or dataset object
          An xarray dataarray or dataset object with 'lon' and 'lat' dimensions.

    Returns:
    --------
    grid_step: tuple of floats
                Tuple containing the grid step in longitude and latitude direction.
    """

    if 'lon' not in ds.dims or 'lat' not in ds.dims:
        raise ValueError(
            'Input dataarray or dataset does not have "lon" and "lat" dimensions.')

    lon = np.array(ds.lon)
    lat = np.array(ds.lat)
    grid_step_lon = np.round(np.abs(lon[1] - lon[0]), 2)
    grid_step_lat = np.round(np.abs(lat[1] - lat[0]), 2)

    if grid_step_lat == grid_step_lon:
        grid_step = grid_step_lat
    else:
        gut.myprint(
            f'Different lon {grid_step_lon} and lat {grid_step_lat} step!',
            verbose=verbose)
        grid_step = grid_step_lat
    return grid_step, grid_step_lon, grid_step_lat


def merge_datasets(datasets, multiple='max'):
    """
    Merges multiple xarray datasets into a single dataset, checking for consistency of lon and lat dimensions.

    Args:
    datasets (list): A list of xarray datasets.

    Returns:
    merged_dataset (xarray.Dataset): A merged xarray dataset with sorted time points.

    Raises:
    ValueError: If the datasets have inconsistent lon or lat dimensions.
    """

    # Check for consistency of lon and lat dimensions
    ref_ds = datasets[0]
    for ds in datasets[1:]:
        if not np.array_equal(ds['lon'], ref_ds['lon']) or not np.array_equal(ds['lat'], ref_ds['lat']):
            raise ValueError("Inconsistent lon or lat dimensions")

    # Check consistency of variable names
    variable_names = [set(ds.variables.keys()) for ds in datasets]
    if not all(x == variable_names[0] for x in variable_names):
        raise ValueError('Variable names are not consistent.')

    # Merge the datasets
    gut.myprint(f'Merge {len(datasets)} datasets to 1 dataset')
    merged_dataset = tu.merge_time_arrays(
        time_arrays=datasets, multiple=multiple)

    return merged_dataset


def get_data_coordinate(da, coord,
                        lon_name='lon',
                        lat_name='lat',
                        in3d=False, method='nearest'):

    if in3d and len(coord) != 3:
        raise ValueError(
            'Coordinate must have 3 elements for 3D dataarray!')

    if lon_name not in da.dims or lat_name not in da.dims:
        raise ValueError(
            f'Dataarray must have {lon_name} and {lat_name} dimensions!')

    if in3d:
        x_0, y_0, lev_0 = coord
        if method == 'nearest':
            da_coord = da.sel(lon=x_0, lat=y_0, lev=lev_0,
                              method=method)
        else:
            da_coord = da.interp(lon=x_0, lat=y_0, lev=lev_0,
                                 method=method,
                                 kwargs={"fill_value": "extrapolate"}
                                 )
    else:
        x_0, y_0 = coord
        if method == 'nearest':
            da_coord = da.sel({lon_name: x_0,
                               lat_name: y_0},
                              method=method)
        else:
            da_coord = da.interp({lon_name: x_0,
                                  lat_name: y_0},
                                 method=method)

    return da_coord


def calculate_percentile(arr1, arr2,):
    """
    Calculate the percentile of values in arr1 with respect to arr2 along the time dimension.

    Parameters:
        arr1 (xarray.DataArray): Array of dimension (lon, lat).
        arr2 (xarray.DataArray): Array of dimension (time, lon, lat).

    Returns:
        xarray.DataArray: An array of dimension (lon, lat) representing the percentile values.
    """
    # compute the percentile values of arr1 with respect to arr2
    # arr1 = transpose_2D_data(da=arr1)
    # arr2 = transpose_3D_data(da=arr2)

    percentile_values = np.zeros(arr1.shape)

    for i, lat in enumerate(arr1.lat.values):
        for j, lon in enumerate(arr1.lon.values):
            this_val = st.percentileofscore(
                arr2.sel(lon=lon, lat=lat, method='nearest'), arr1.sel(lon=lon, lat=lat, method='nearest'))
            percentile_values[i, j] = this_val

    return xr.DataArray(percentile_values, coords=arr1.coords, dims=arr1.dims)


def extract_defined_time_series(xarr, mask):
    """
    Extracts the defined time series from an xarray object.

    Parameters:
        xarr (xarray.DataArray): An xarray object of dimension (lat, lon, time).

    Returns:
        xarray.DataArray: An xarray object containing the defined time series with the same time coordinates.
    """
    # Convert the xarray object to a numpy array
    mask = transpose_2D_data(da=mask, dims=['lat', 'lon'])
    ids = np.where(mask.data.flatten())[0]
    xarr = xarr.where(mask)
    xarr = transpose_3D_data(da=xarr, dims=['lat', 'lon', 'time'])
    arr = xarr.values

    # Find the locations where all time points are np.nan
    undefined_locs = np.all(np.isnan(arr), axis=-1)

    # Extract the defined time series by excluding the undefined locations
    defined_time_series = arr[~undefined_locs]

    # Create a new xarray object with the defined time series and the same time coordinates
    defined_xarr = xr.DataArray(defined_time_series, coords={'time': xarr.time,
                                                             'ids': ids},
                                dims=['ids', 'time'])

    return defined_xarr


def find_location_groups(locations, grid_step=3, min_num_locations=5, verbose=True):
    """
    Find groups of locations where each location shares at least one neighbor within a specified distance.

    Parameters:
        locations (list): A list of locations, where each location is a tuple of float values (lon, lat).
        grid_step (float): The maximum distance allowed between neighboring locations within a group.

    Returns:
        list: A list of arrays, where each array represents a group of locations.
    """
    groups = []

    def find_group(location):
        for group in groups:
            for loc in group:
                if is_neighbor(location, loc):
                    return group
        return None

    def is_neighbor(loc1, loc2):
        return np.abs(loc1[0] - loc2[0]) <= grid_step \
            and np.abs(loc1[1] - loc2[1]) <= grid_step

    for location in locations:
        group = find_group(location)
        if group is not None:
            group.append(location)
        else:
            groups.append([location])

    merged_groups = []
    for group in groups:
        merged = False
        for merged_group in merged_groups:
            for loc in group:
                for merged_loc in merged_group:
                    if is_neighbor(loc, merged_loc):
                        merged_group.extend(group)
                        merged = True
                        break
                if merged:
                    break
            if merged:
                break
        if not merged:
            merged_groups.append(group)

    return_arr = []
    tot_sign_locs = 0
    for group in merged_groups:
        if len(group) >= min_num_locations:
            return_arr.append(np.array(group))
            tot_sign_locs += len(group)
    gut.myprint(f'Found {len(return_arr)} groups with {tot_sign_locs} locations',
                verbose=verbose)

    return return_arr


def stack_lat_lon(da, new_dim='z', reindex=False):
    dims = gut.get_dims(da)
    if len(dims) == 3:
        if 'time' in dims:
            dims.remove('time')
    else:
        raise ValueError('Dataarray not of dimension 3!')
    da_flat = da.stack(z=dims)
    if reindex:
        da_flat = gut.assign_new_coords(da=da_flat, dim='z',
                                        coords=np.arange(len(da_flat.z)))
    if new_dim != 'z':
        da_flat = da_flat.rename({'z': new_dim})

    return da_flat


def unstack_lat_lon(da, dim='z'):

    da_unstacked = da.unstack(dim)

    return da_unstacked


def get_defined_lon_lat_range(dataarray):
    """
    Returns the range of latitudes and longitudes within which values are defined (not NaN).

    Parameters:
    - dataarray (xr.DataArray): An xarray DataArray with coordinates 'lat' and 'lon'.

    Returns:
    - lat_range (tuple): Min and max latitude where data is defined.
    - lon_range (tuple): Min and max longitude where data is defined.
    """
    # Mask to find locations with non-NaN values
    non_nan_data = dataarray.where(~np.isnan(dataarray), drop=True)

    # Get min and max of latitude and longitude where data is not NaN
    lat_range = np.array([non_nan_data.lat.min().item(),
                         non_nan_data.lat.max().item()])
    lon_range = np.array([non_nan_data.lon.min().item(),
                         non_nan_data.lon.max().item()])

    return lon_range, lat_range


def max_power_of_2(n):
    """Finds the largest power of 2 that divides n."""
    power = 0
    while n % 2 == 0 and n > 1:
        n //= 2
        power += 1
    return power


def trim_range(arr, target_length, mode):
    """Trims an array to the target length based on the given mode."""
    current_length = len(arr)
    trim_amount = current_length - target_length
    if trim_amount <= 0:
        return arr
    else:
        if mode == "start":  # Cut from beginning (e.g., west or south)
            return arr[trim_amount:]
        elif mode == "end":  # Cut from end (e.g., east or north)
            return arr[:target_length]
        elif mode == "center":  # Cut evenly from both sides
            start_trim = trim_amount // 2
            end_trim = trim_amount - start_trim
            return arr[start_trim: -end_trim]
        else:
            raise ValueError("Mode must be 'start', 'end', or 'center'")


def find_optimal_subrange(ds, lat_name="lat", lon_name="lon",
                          min_power=2,
                          lat_trim="center", lon_trim="center"):
    """
    Finds the largest subrange where the number of latitude and longitude values
    is divisible by 2 at least `min_power` times.

    Parameters:
        ds (xarray.Dataset): Input dataset.
        lat_name (str): Name of the latitude dimension.
        lon_name (str): Name of the longitude dimension.
        min_power (int): Minimum number of times both lat and lon counts should be divisible by 2.

    Returns:
        xarray.Dataset: Optimized subset of the dataset.
    """
    lat_vals = ds[lat_name].values
    lon_vals = ds[lon_name].values

    # Compute original lengths
    num_lat = len(lat_vals)
    num_lon = len(lon_vals)

    # Compute initial power of 2 divisibility
    best_lat_pow = max_power_of_2(num_lat)
    best_lon_pow = max_power_of_2(num_lon)

    print(
        f"Original lat count: {num_lat}, divisible by 2: {best_lat_pow} times")
    print(
        f"Original lon count: {num_lon}, divisible by 2: {best_lon_pow} times")

    # Ensure both dimensions are divisible by 2 at least `min_power` times
    num_iters = 0
    while num_lat > 1 and num_lon > 1:
        if best_lat_pow >= min_power and best_lon_pow >= min_power:
            break  # Stop when both meet the required divisibility

        num_lat -= 1 if best_lat_pow < min_power else 0
        num_lon -= 1 if best_lon_pow < min_power else 0
        best_lat_pow = max_power_of_2(num_lat)
        best_lon_pow = max_power_of_2(num_lon)
        num_iters += 1

    print(
        f"Optimized lat count: {num_lat}, divisible by 2: {best_lat_pow} times")
    print(
        f"Optimized lon count: {num_lon}, divisible by 2: {best_lon_pow} times")

    # Trim latitude and longitude ranges based on the user-defined method
    if num_iters > 0:
        gut.myprint(f"Trimmed {num_iters} times!")
        lat_vals_optimal = trim_range(lat_vals, num_lat, mode=lat_trim)
        lon_vals_optimal = trim_range(lon_vals, num_lon, mode=lon_trim)
        # Select optimized range from dataset
        ds_optimal = ds.sel({lat_name: lat_vals_optimal,
                            lon_name: lon_vals_optimal})
    else:
        ds_optimal = ds
    return ds_optimal


def rapsd(field: np.ndarray,
                           fft_method=np.fft,
                           return_freq: bool = True,
                           d: float = 1.0,
                           normalize: bool = False
                           ) -> np.ndarray:
    """

    Adapted from https://github.com/pySTEPS/pysteps/blob/57ece4335acffb111d4de7665fb678b875d844ac/pysteps/utils/spectral.py#L100

    Compute radially averaged power spectral density (RAPSD) from the given
    2D input field.

    Args:
        field: A 2d array of shape (m, n) containing the input field.
        fft_method: A module or object implementing the same methods as numpy.fft and
            scipy.fftpack. If set to None, field is assumed to represent the
            shifted discrete Fourier transform of the input field, where the
            origin is at the center of the array
            (see numpy.fft.fftshift or scipy.fftpack.fftshift).
        return_freq: Whether to also return the Fourier frequencies.
        d: Sample spacing (inverse of the sampling rate). Defaults to 1.
            Applicable if return_freq is 'True'.
        normalize: If True, normalize the power spectrum so that it sums to one.

    Returns:
    out: One-dimensional array containing the RAPSD. The length of the array is
         int(l/2) (if l is even) or int(l/2)+1 (if l is odd), where l=max(m,n).
    freq: One-dimensional array containing the Fourier frequencies.
    """

    if len(field.shape) != 2:
        raise ValueError(
            f"{len(field.shape)} dimensions are found, but the number "
            "of dimensions should be 2"
        )

    if np.sum(np.isnan(field)) > 0:
        raise ValueError("input field should not contain nans")

    m, n = field.shape

    yc, xc = compute_centred_coord_array(m, n)
    r_grid = np.sqrt(xc * xc + yc * yc).round()
    l = max(field.shape[0], field.shape[1])

    if l % 2 == 1:
        r_range = np.arange(0, int(l / 2) + 1)
    else:
        r_range = np.arange(0, int(l / 2))

    if fft_method is not None:
        psd = fft_method.fftshift(fft_method.fft2(field))
        psd = np.abs(psd) ** 2 / psd.size
    else:
        psd = field

    result = []
    for r in r_range:
        mask = r_grid == r
        psd_vals = psd[mask]
        result.append(np.mean(psd_vals))

    result = np.array(result)

    if normalize:
        result /= np.sum(result)

    if return_freq:
        freq = np.fft.fftfreq(l, d=d)
        freq = freq[r_range]
        return result, freq
    else:
        return result


def compute_centred_coord_array(M: np.ndarray, N: np.ndarray) -> np.ndarray:
    """
    Compute a 2D coordinate array, where the origin is at the center.

    Args:
        M: The height of the array.
        N: The width of the array.

    Returns:
        The coordinate array.
    """

    if M % 2 == 1:
        s1 = np.s_[-int(M / 2): int(M / 2) + 1]
    else:
        s1 = np.s_[-int(M / 2): int(M / 2)]

    if N % 2 == 1:
        s2 = np.s_[-int(N / 2): int(N / 2) + 1]
    else:
        s2 = np.s_[-int(N / 2): int(N / 2)]

    YC, XC = np.ogrid[s1, s2]

    return YC, XC


def mean_rapsd(data: xr.DataArray, normalize: bool = False):
    """
    Averages the RAPSD in time over a DataArray.

    Args:
        data: The dataset with shape [time, lat, lon]
        normalize: Normalize the spectra

    Returns:
        Average RAPSD, Fourier frequencies
    """
    from tqdm import tqdm
    assert (len(data.lat) == len(data.lon)
            ), "Number of latitude coordinates must equal the number of lon."

    mean_psd = np.zeros(len(data.lat)//2)

    for i in tqdm(range(len(data))):
        data_slice = data[i].values
        psd, freq = rapsd(data_slice, fft_method=np.fft,
                          normalize=normalize, return_freq=True)
        mean_psd += psd

    mean_psd /= len(data.time)

    return mean_psd, freq
