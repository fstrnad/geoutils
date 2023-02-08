import cftime
import geoutils.utils.statistic_utils as sut
from tqdm import tqdm
import geoutils.utils.general_utils as gut
from sklearn.neighbors import KernelDensity
import scipy.stats as st
import geoutils.utils.time_utils as tu
import numpy as np
import xarray as xr
import scipy.interpolate as interp
from importlib import reload

RADIUS_EARTH = 6371  # radius of earth in km


def flatten_array(dataarray, mask=None, time=True, check=False):
    """Flatten and remove NaNs.
    """

    if mask is not None:
        idx_land = np.where(mask.data.flatten() == 1)[0]
    else:
        idx_land = None
    if time is False:
        buff = dataarray.data.flatten()
        buff[np.isnan(buff)] = 0.0  # set missing data to climatology
        data = buff[idx_land] if idx_land is not None else buff[:]
    else:
        data = []
        for idx, t in enumerate(dataarray.time):
            buff = dataarray.sel(time=t.data).data.flatten()
            buff[np.isnan(buff)] = 0.0  # set missing data to climatology
            data_tmp = buff[idx_land] if idx_land is not None else buff[:]
            data.append(data_tmp)

    # check
    if check is True:
        num_nonzeros = np.count_nonzero(data[-1])
        num_landpoints = sum(~np.isnan(mask.data.flatten()))
        gut.myprint(
            f"The number of non-zero datapoints {num_nonzeros} "
            + f"should approx. be {num_landpoints}."
        )

    return np.array(data)


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
            'std_lon': std_lon,
            'std_lat': std_lat}


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
            print(f'Compute RM Year {year}')
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
            dateline=False, ):
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
            lon=lon_range,
            lat=slice(np.min(lat_range), np.max(lat_range))
        )
    else:
        ds_cut = da.sel(
            lon=lon_range,
            lat=slice(np.max(lat_range), np.min(lat_range))
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
        if np.abs(np.max(lon_range) - np.min(lon_range)) > 180:
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


@ np.vectorize
def haversine(lon1, lat1, lon2, lat2, radius=1):
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
        Array of all links provided as [lon, lat]
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


def get_kde_map(ds, data, nn_points=None, bandwidth='scott'):
    if nn_points is not None:
        bandwidth = get_nn_bw(nn_points=nn_points, grid_step=ds.grid_step)

    coord_deg, coord_rad, map_idx = ds.get_coordinates_flatten()

    link_points = np.where(data > 0)[0]
    links_rad = coord_rad[link_points]
    Z_kde = spherical_kde(link_points=links_rad,
                          coord_rad=coord_rad,
                          bw_opt=bandwidth)

    return ds.get_map(Z_kde)


def get_LR_map(ds, var, method='standardize', sids=None, deg=1):
    reload(sut)
    if sids is None:
        sids = ds.indices_flat

    pids = ds.get_points_for_idx(sids)

    arr = np.zeros((len(sids), ds.mask.shape[0]))
    if method == 'standardize':
        y_data = sut.standardize(ds.ds[var])
    elif method == 'normalize':
        y_data = sut.normalize(ds.ds[var])
    elif method == 'rank':
        y_data = sut.rank_data(ds.ds[var])
    else:
        print('No Normalization on time series performed!')
        y_data = ds.ds[var]
    for idx, pid in enumerate(tqdm(pids)):
        ts = y_data.sel(points=pid)
        poly_coeff = np.polyfit(
            x=ts, y=y_data, full=False, deg=deg)
        arr[idx, :], _ = poly_coeff

    da_LR = xr.DataArray(
        data=arr,
        dims=['sids', 'points'],
        coords=dict(
            sids=sids,
            points=ds.ds.points,
            lon=ds.ds.lon,
            lat=ds.ds.lat,
        ),
        name='LR'
    )

    return da_LR


def get_corr_map(ds, var, sids=None, method='spearman', p_value_test='twosided'):
    """Generate a map of correlations to all points in a dataarray for a set of ids.

    Args:
        ds (climnet.BaseDataset): BaseDataset class
        var (string): Variable name on which to apply the correlations
        sids (int, optional): List of int ids of (defined) data points. Defaults to None.
        method (str, optional): Correlation method. Defaults to 'spearman'.
        p_value_test (str, optional): selected p-value test. Defaults to 'twosided'.

    Returns:
        xr.DataSet: Dataset that as as variables corr, pvalue and dimension is len(sids).
    """
    if sids is None:
        sids = ds.indices_flat

    pids = ds.get_points_for_idx(sids)
    if method == 'pearson':
        corr, p = sut.calc_pearson(data=ds.ds[var].data)
    elif method == 'spearman':
        corr, p = sut.calc_spearman(data=ds.ds[var].data, test=p_value_test)

    arr = np.zeros((len(sids), ds.mask.shape[0]))
    p_arr = np.zeros((len(sids), ds.mask.shape[0]))
    for idx, pid in enumerate(tqdm(pids)):
        arr[idx, :] = corr[pid, :]
        p_arr[idx, :] = p[pid, :]

    da_corr = xr.DataArray(
        data=arr,
        dims=['sids', 'points'],
        coords=dict(
            sids=sids,
            points=ds.ds.points,
            lon=ds.ds.lon,
            lat=ds.ds.lat,
        ),
        name='corr'
    )
    da_p = xr.DataArray(
        data=p_arr,
        dims=['sids', 'points'],
        coords=dict(
            sids=sids,
            points=ds.ds.points,
            lon=ds.ds.lon,
            lat=ds.ds.lat,
        ),
        name='p'
    )
    da_corr = da_corr.to_dataset()
    da_corr['p'] = da_p

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


def remove_useless_variables(ds):

    vars = gut.get_vars(ds=ds)
    for var in vars:
        this_dims = gut.get_dims(ds[var])
        if(
            (gut.compare_lists(this_dims, ['lat', 'lon', 'time'])) or
            (gut.compare_lists(this_dims, ['lat', 'lon'])) or
            (gut.compare_lists(this_dims, ['lat', 'lon', 'time', 'plevel'])) or
            (gut.compare_lists(this_dims, ['lat', 'lon', 'plevel']))
        ):
            continue
        else:
            ds = ds.drop(var)
            gut.myprint(f'Remove variable {var} with dims: {this_dims}!')

    dims = gut.get_dims(ds=ds)
    for dim in dims:
        if dim not in ['time', 'lat', 'lon', 'plevel']:
            ds = ds.drop_dims(dim)
            gut.myprint(f'Remove dim {dim}!')

    return ds


def remove_single_dim(ds):
    dims = dict(ds.dims)
    for dim, num in dims.items():
        if num < 2:
            ds = ds.mean(dim=dim)  # Removes the single variable axis
            gut.myprint(f'Remove single value dimension {dim}!')
    return ds


def da_lon2_180(da):
    return da.assign_coords(lon=(((da.lon + 180) % 360) - 180))


def da_lon2_360(da):
    da = da.assign_coords(lon=(((da.lon) % 360)))

    return da


def check_dimensions(ds, ts_days=True, sort=True, lon360=False, keep_time=False,
                     freq='D'):
    """
    Checks whether the dimensions are the correct ones for xarray!
    """
    reload(tu)

    lon_lat_names = ['longitude', 'latitude', 't', 'month', 'time_counter']
    xr_lon_lat_names = ['lon', 'lat', 'time', 'time', 'time']
    dims = list(ds.dims)
    dim3 = len(dims) > 2
    for idx, lon_lat in enumerate(lon_lat_names):
        if lon_lat in dims:
            gut.myprint(dims)
            gut.myprint(f'Rename:{lon_lat} : {xr_lon_lat_names[idx]} ')
            ds = ds.rename({lon_lat: xr_lon_lat_names[idx]})
            dims = list(ds.dims)
            gut.myprint(dims)
    ds = remove_single_dim(ds=ds)
    ds = remove_useless_variables(ds=ds)

    if dim3:
        clim_dims = ['time', 'lat', 'lon']
    else:
        clim_dims = ['lat', 'lon']
    for dim in clim_dims:
        if dim not in dims:
            raise ValueError(
                f"The dimension {dims} not consistent with required dims {clim_dims}!")

    if dim3:
        # Actually change location in memory if necessary!
        ds = ds.transpose("lat", "lon", "time").compute()
        gut.myprint('3d object transposed to lat-lon-time!')
    else:
        ds = ds.transpose('lat', 'lon').compute()
        gut.myprint('2d oject transposed to lat-lon!')

    # If lon from 0 to 360 shift to -180 to 180
    if lon360:
        gut.myprint('Longitudes 0 - 360!')
        if max(ds.lon) < 180:
            ds = da_lon2_360(da=ds)
    else:
        if max(ds.lon) > 180:
            gut.myprint("Shift longitude -180 - 180!")
            ds = da_lon2_180(da=ds)

    if sort:
        ds = ds.sortby('lon')
        ds = ds.sortby('lat')
        gut.myprint(
            'Sorted longitudes and latitudes in ascending order, respectively')

    if 'time' in dims:
        if ts_days:
            if gut.is_datetime360(time=ds.time.data[0]) or keep_time:
                ds = ds
            else:
                reload(tu)
                ds = tu.get_netcdf_encoding(ds=ds,
                                            calendar='gregorian',
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
