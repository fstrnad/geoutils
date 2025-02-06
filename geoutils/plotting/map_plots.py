"""Basic plotting functions for maps"""
# import matplotlib.cm as cm
import copy
import geoutils.utils.spatial_utils as sput
import matplotlib.patches as mpatches
import geoutils.utils.general_utils as gut
import numpy as np
import xarray as xr
import cartopy
import cartopy as ctp
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from importlib import reload

import geoutils.plotting.plotting_utils as put
import geoutils.plotting.plot_settings as pst
reload(put)
reload(pst)
reload(gut)


def estimate_distance(minimum_value, maximum_value, min_dist_val=1,
                      multiple=2):
    if maximum_value - minimum_value > 20:
        multiple = 5
    # Calculate the range between minimum and maximum value
    value_range = maximum_value - minimum_value

    # Calculate the maximum step size to be a multiple of 6
    step_size = max(value_range / 6, min_dist_val)

    # Check if the step size is smaller than min_dist_val
    if step_size < min_dist_val:
        return min_dist_val

    # Check if the step size is already a multiple of 5
    if step_size % 5 == 0:
        return step_size

    # Find the closest multiple of multiple to the step size
    closest_multiple = ((step_size // multiple) + 1) * multiple

    return closest_multiple


def get_grid_steps(grid_step, min_value=-90, max_value=90):
    """
    Generates a sorted NumPy array of grid steps within the specified range,
    with the specified grid step, including the 0 value.

    Args:
        grid_step (int): The step size between consecutive grid points.
        min_value (int, optional): The minimum value of the grid steps range (default: -90).
        max_value (int, optional): The maximum value of the grid steps range (default: 90).

    Returns:
        numpy.ndarray: A sorted NumPy array of grid steps.

    Examples:
        >>> grid_steps = get_grid_steps(15)
        >>> print(grid_steps)
        [-90 -75 -60 -45 -30 -15   0  15  30  45  60  75  90]

        >>> grid_steps = get_grid_steps(10, -50, 50)
        >>> print(grid_steps)
        [-50 -40 -30 -20 -10   0  10  20  30  40  50]
    """

    steps = [0]  # Include 0 value

    current_step = grid_step
    while current_step <= max_value:
        steps.append(current_step)
        current_step += grid_step

    current_step = -grid_step
    while current_step >= min_value:
        steps.append(current_step)
        current_step -= grid_step

    steps.sort()
    return np.array(steps)


def get_grid_dist(ext_dict, min_dist_val_lon=1, min_dist_val_lat=1):
    min_lon = ext_dict['min_lon']
    max_lon = ext_dict['max_lon']
    min_lat = ext_dict['min_lat']
    max_lat = ext_dict['max_lat']

    gs_lon = estimate_distance(min_lon, max_lon, min_dist_val=min_dist_val_lon)
    gs_lat = estimate_distance(min_lat, max_lat, min_dist_val=min_dist_val_lat)

    # print(f'Grid steps for longitude: {gs_lon}, latitude: {gs_lat}')
    return gs_lon, gs_lat


def set_grid(ax, alpha=0.5,
             proj='PlateCarree',
             ext_dict={},
             **kwargs):
    """Generates a grid for a spatial map.

    Args:
        ax (mpl.axes): axis object
        alpha (float, optional): transparancy of the grid. Defaults to 0.5.

    Returns:
        _type_: _description_
    """

    # Set grid steps for longitude and latitude
    if ext_dict is not None:
        gs_lon = kwargs.pop('gs_lon', None)
        gs_lat = kwargs.pop('gs_lat', None)
        if gs_lon is None:
            gs_lon, gs_lat = get_grid_dist(ext_dict=ext_dict)
        else:
            gs_lat = 20 if gs_lat is None else gs_lat

        if proj != 'PlateCarree':
            gs_lon = 60
            gs_lat = 30
    else:
        gs_lon = 30
        gs_lat = 20

    # Generate the grid
    gl = ax.gridlines(
        draw_labels=True,
        xlocs=get_grid_steps(gs_lon, min_value=-180, max_value=180),
        ylocs=get_grid_steps(gs_lat, min_value=-90, max_value=90),
        crs=ccrs.PlateCarree(),
        x_inline=False,
        y_inline=False,
        alpha=alpha,
    )
    # gl = ax.gridlines(draw_labels=True, dms=True,
    #                   x_inline=False, y_inline=False, )
    # gl.rotate_labels = False
    gl.left_labels = True
    gl.right_labels = False
    # gl.right_labels=True
    gl.bottom_labels = False
    gl.ylabel_style = {'rotation': -0,
                       'color': 'black', 'size': pst.MEDIUM_SIZE}
    gl.xlabel_style = {'rotation': -0,
                       'color': 'black', 'size': pst.MEDIUM_SIZE}
    gl.top_labels = True

    return ax, kwargs


def set_extent(da, ax,
               lat_range=None,
               lon_range=None,
               dateline=False,
               set_global=False,
               **kwargs):
    if not isinstance(ax, ctp.mpl.geoaxes.GeoAxesSubplot) and not isinstance(ax, ctp.mpl.geoaxes.GeoAxes):
        raise ValueError(
            f'Axis is not of type Geoaxis, but of type {type(ax)}!')
    verbose = kwargs.get('verbose', False)
    min_ext_lon, max_ext_lon, min_ext_lat, max_ext_lat = ax.get_extent()
    if dateline:
        if ax.projection.proj4_params['lon_0'] != 180:
            raise ValueError(
                f'Axis is not set to dateline, but central_longitude not to 180!')
        projection = ccrs.PlateCarree(central_longitude=0)
        if lon_range is not None:
            lon_range = sput.lon2_360(lon_range)
        min_ext_lon += 180
        max_ext_lon += 180
        gut.myprint(
            f'Dateline: Set extend {min_ext_lon} - {max_ext_lon}!',
            verbose=verbose)
    else:
        projection = ccrs.PlateCarree(central_longitude=0)
    if not isinstance(da, xr.DataArray):
        if da is not None:
            if [min_ext_lon, max_ext_lon] == [-180, 180] and [min_ext_lat, max_ext_lat] == [-90, 90]:
                # expect list of tuple of type (lon, lat)
                min_ext_lon = np.min(da[:, 0])
                min_ext_lat = np.min(da[:, 1])
                max_ext_lon = np.max(da[:, 0])
                max_ext_lat = np.max(da[:, 1])

    else:
        if da is None and [min_ext_lon, max_ext_lon] == [-180, 180] and [min_ext_lat, max_ext_lat] == [-90, 90]:
            if lat_range is None and lon_range is None:
                set_global = True
        else:
            # expect DataArray with lon and lat coordinates
            # If condition is true, the extent is not set yet and is defined by data coordinates
            if [min_ext_lat, max_ext_lat] == [-90, 90]:
                min_ext_lat = float(
                    np.min(da.coords["lat"])) if da is not None else min_ext_lat
                max_ext_lat = float(
                    np.max(da.coords["lat"])) if da is not None else max_ext_lat
            if [min_ext_lon, max_ext_lon] == [-180, 180] or dateline:
                min_ext_lon = float(
                    np.min(da.coords["lon"])) if da is not None else min_ext_lon
                max_ext_lon = float(
                    np.max(da.coords["lon"])) if da is not None else max_ext_lon
                min_ext_lat = float(
                    np.min(da.coords["lat"])) if da is not None else min_ext_lat
                max_ext_lat = float(
                    np.max(da.coords["lat"])) if da is not None else max_ext_lat

    if lat_range is not None or lon_range is not None:
        lat_range = lat_range if lat_range is not None else [
            min_ext_lat, max_ext_lat]
        lon_range = lon_range if lon_range is not None else [
            min_ext_lon, max_ext_lon]
        min_ext_lon = np.min(lon_range)
        max_ext_lon = np.max(lon_range)
        min_ext_lat = np.min(lat_range)
        max_ext_lat = np.max(lat_range)
    if not set_global:
        if (abs(min_ext_lon) > 179 and
            abs(max_ext_lon) > 179 and
            abs(min_ext_lat) > 89 and
                abs(max_ext_lat) > 89):
            set_global = True
            # gut.myprint('WARNING! Set global map!')
    final_extent = [min_ext_lon, max_ext_lon, min_ext_lat, max_ext_lat]
    if set_global:
        ax.set_global()
    else:
        ax.set_extent(final_extent, crs=projection)

    ext_dict = dict(
        ax=ax,
        min_lon=min_ext_lon,
        max_lon=max_ext_lon,
        min_lat=min_ext_lat,
        max_lat=max_ext_lat
    )

    return ext_dict


def utm_from_lon(lon):
    """
    utm_from_lon - UTM zone for a longitude

    Not right for some polar regions (Norway, Svalbard, Antartica)

    :param float lon: longitude
    :return: UTM zone number
    :rtype: int
    """
    from math import floor
    return floor((lon + 180) / 6) + 1


def scale_bar(ax, proj='PlateCarree',
              length=1000, location=(0.5, 0.05),
              linewidth=3,
              units='km',
              m_per_unit=1000):
    """

    http://stackoverflow.com/a/35705477/1072212
    ax is the axes to draw the scalebar on.
    proj is the projection the axes are in
    location is center of the scalebar in axis coordinates ie. 0.5 is the middle of the plot
    length is the length of the scalebar in km.
    linewidth is the thickness of the scalebar.
    units is the name of the unit
    m_per_unit is the number of meters in a unit
    """
    from matplotlib import patheffects

    proj = get_projection(projection=proj)

    # find lat/lon center to find best UTM zone
    x0, x1, y0, y1 = ax.get_extent(proj.as_geodetic())
    # Projection in metres
    utm = ccrs.UTM(utm_from_lon((x0+x1)/2))
    # Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(utm)
    # Turn the specified scalebar location into coordinates in metres
    sbcx, sbcy = x0 + (x1 - x0) * location[0], y0 + (y1 - y0) * location[1]
    # Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbcx - length * m_per_unit/2, sbcx + length * m_per_unit/2]
    # buffer for scalebar
    buffer = [patheffects.withStroke(linewidth=5, foreground="w")]
    # Plot the scalebar with buffer
    ax.plot(bar_xs, [sbcy, sbcy], transform=utm, color='k',
            linewidth=linewidth, path_effects=buffer)
    # buffer for text
    buffer = [patheffects.withStroke(linewidth=3, foreground="w")]
    # Plot the scalebar label
    t0 = ax.text(sbcx, sbcy, str(length) + ' ' + units, transform=utm,
                 horizontalalignment='center', verticalalignment='bottom',
                 path_effects=buffer, zorder=2)
    left = x0+(x1-x0)*0.05
    # # Plot the N arrow
    # t1 = ax.text(left, sbcy, u'\u25B2\nN', transform=utm,
    #              horizontalalignment='center', verticalalignment='bottom',
    #              path_effects=buffer, zorder=2)
    # Plot the scalebar without buffer, in case covered by text buffer
    ax.plot(bar_xs, [sbcy, sbcy], transform=utm, color='k',
            linewidth=linewidth, zorder=3)


def create_map(
    da=None,
    ax=None,
    fig=None,
    projection="PlateCarree",
    central_longitude=None,
    alpha=1,
    plot_grid=True,  # Check if this is already set from before!
    lat_range=None,
    lon_range=None,
    dateline=False,
    verbose=False,
    **kwargs,
):
    projection = 'PlateCarree' if lon_range is not None else projection
    central_latitude = kwargs.pop("central_latitude", 0)

    proj = get_projection(projection=projection,
                          central_longitude=central_longitude,
                          central_latitude=central_latitude,
                          dateline=dateline)
    figsize = kwargs.pop("figsize", (9, 6))
    # create figure
    if ax is None:
        # better working in cartopy and matplotlib=3.8.2
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=proj)
        unset_grid = kwargs.pop('unset_grid', False)
        plot_grid = True if not unset_grid else False
    else:
        ax_central_longitude = ax.projection.proj4_params['lon_0']
        if ax_central_longitude == 180:
            dateline = True

            gut.myprint('Dateline set to true!', verbose=verbose)

        if central_longitude is not None:
            if central_longitude != ax_central_longitude:
                gut.myprint(
                    f'WARNING! Central longitude set to {central_longitude} but has no effect since axis argument is passed is {ax_central_longitude}!')
        # Because this is already set from before!

    set_map = kwargs.pop('set_map', True)
    set_global = kwargs.pop('set_global', False)

    ext_dict = set_extent(
        da=da, ax=ax,
        lat_range=lat_range,
        lon_range=lon_range,
        dateline=dateline,
        set_global=set_global,
        **kwargs)

    if set_map:
        # axes properties
        # coast_color = kwargs.pop("coast_color", "k")
        ax.coastlines(alpha=alpha,
                      #   color=coast_color
                      )
        set_borders = kwargs.pop("set_borders", False)
        if set_borders:
            ax.add_feature(ctp.feature.BORDERS,
                           linestyle="--",
                           #    color="grey",
                           alpha=alpha)
        land_ocean = kwargs.pop("land_ocean", False)
        if land_ocean:
            # ax.add_feature(ctp.feature.OCEAN, alpha=.4, zorder=-1)
            ax.add_feature(ctp.feature.LAND, alpha=.3, zorder=-1, color='grey')
        if projection == "PlateCarree":
            rem_frame = kwargs.pop("rem_frame", False)
            if rem_frame:
                # Remove the frame around the map
                ax.outline_patch.set_visible(False)

    if plot_grid:
        ax, kwargs = set_grid(ax, alpha=alpha,
                              proj=projection,
                              ext_dict=ext_dict,
                              **kwargs)

    return dict(ax=ax, fig=fig, kwargs=kwargs)


def get_projection(projection, central_longitude=None, central_latitude=None,
                   dateline=False):
    central_longitude = 0 if central_longitude is None else central_longitude
    central_latitude = 0 if central_latitude is None else central_latitude
    if dateline:
        central_longitude = 180

    if not isinstance(central_longitude, float) and not isinstance(central_longitude, int):
        raise ValueError(
            f'central_longitude is not of type{type(central_longitude)}!'
        )
    if projection == "Mollweide":
        proj = ccrs.Mollweide(central_longitude=central_longitude)
    elif projection == "EqualEarth":
        proj = ccrs.EqualEarth(central_longitude=central_longitude)
    elif projection == "Robinson":
        proj = ccrs.Robinson(central_longitude=central_longitude)
    elif projection == "PlateCarree":
        proj = ccrs.PlateCarree(central_longitude=central_longitude)
    elif projection == "Nearside":
        proj = ccrs.NearsidePerspective(
            central_longitude=central_longitude, central_latitude=central_latitude)
    elif projection == "LambertConformal":
        proj = ccrs.LambertConformal(central_longitude=central_longitude)
    else:
        raise ValueError(f"This projection {projection} is not available yet!")

    return proj


def plot_map(dmap: xr.DataArray,
             ds: object = None,
             fig: plt.Figure = None,
             ax: plt.Axes = None,
             plot_type: str = "colormesh",
             central_longitude: int = None,
             vmin: float = None,
             vmax: float = None,
             cmap: str = 'viridis',
             projection: str = None,
             label: str = None,
             title: str = None,
             significance_mask: xr.DataArray = None,
             lat_range: tuple[float, float] = None,
             lon_range: tuple[float, float] = None,
             dateline: bool = False,
             **kwargs):
    """
    This function plots a map of a given xr.DataArray.
    Parameters:
      dmap (xr.DataArray): DataArray that needs to be plotted.
      fig (plt.Figure): Matplotlib figure object.
      ax (plt.Axes): Matplotlib axes object.
      ds (object): dataset object.
      plot_type (str): Type of plot, contourf is default.
      central_longitude (int): Central longitude of the map.
      vmin (int): Minimum value for the colorbar.
      vmax (int): Maximum value for the colorbar.
      cmap (str): Colormap used for the plot.
      bar (bool): Add colorbar or not.
      projection (str): Map projection to be used.
      label (str): Label for the colorbar.
      title (str): Title for the plot.
      significance_mask (xr.DataArray): Significance mask to apply on the plot.
      lat_range (Tuple[float, float]): Latitude range for the plot.
      lon_range (Tuple[float, float]): Longitude range for the plot.
      **kwargs: Additional keyword arguments.
    """

    reload(put)
    reload(sput)
    plt.rcParams["pcolor.shading"] = "nearest"  # For pcolormesh
    if label is None and isinstance(dmap, xr.DataArray):
        unset_label = kwargs.pop('unset_label', False)
        label = dmap.name if not unset_label else None
    if isinstance(dmap, xr.DataArray):
        data_dims = gut.get_dims(dmap)
        point_dims = 'points' in data_dims
    else:
        point_dims = False
    hatch_type = kwargs.pop('hatch_type', '..')
    inverse_mask = kwargs.pop('inverse_mask', False)
    set_map = kwargs.pop('set_map', True)
    figsize = kwargs.pop("figsize", (9, 6))
    alpha = kwargs.pop("alpha", 1.0)
    sig_plot_type = kwargs.pop('sig_plot_type', 'hatch')
    if plot_type != 'points' and not point_dims:
        dmap = sput.check_dimensions(dmap,
                                     transpose_dims=True,
                                     verbose=False)
        if not sput.check_full_globe_coverage(dmap):
            dateline = sput.check_if_crosses_dateline(dmap)
            trafo_lon = True if dateline else False
            dmap = sput.check_dimensions(dmap,
                                         transpose_dims=True,
                                         verbose=False,
                                         lon360=trafo_lon)

    put.check_plot_type(plot_type)
    projection = put.check_projection(ax=ax, projection=projection)

    if not isinstance(dmap, xr.DataArray) and plot_type != 'points':
        raise ValueError(
            f'data needs to be xarray object for plot_type = {plot_type}!')

    map_dict = create_map(
        da=dmap,
        ax=ax,
        projection=projection,
        central_longitude=central_longitude,
        set_map=set_map,
        figsize=figsize,
        lat_range=lat_range,
        lon_range=lon_range,
        dateline=dateline,
        **kwargs
    )
    ax = map_dict['ax']
    fig = map_dict['fig']
    kwargs = map_dict['kwargs']
    projection = ccrs.PlateCarree()  # nicht: central_longitude=central_longitude!

    # interpolate grid of points to regular grid
    grid_step = kwargs.pop('grid_step', 1)  # by default 1 grid

    if isinstance(dmap, xr.DataArray):
        if 'points' in list(dmap.dims):
            if plot_type != 'points':
                if ds is not None:
                    grid_step = ds.grid_step
                dmap_dict = sput.interp2gaus(
                    dataarray=dmap, grid_step=grid_step)
                dmap = dmap_dict['intpol']
        x, y = dmap.coords["lon"], dmap.coords["lat"]
        z = dmap
    else:
        # Expect a list of tuples of lon, lat
        z = kwargs.pop('z', None)
        if np.shape(dmap)[1] != 2:
            raise ValueError(
                'For plot_type points please provide a list of tuples!')
        x = dmap[:, 0]
        y = dmap[:, 1]
        if z is not None:
            if len(z) != len(x):
                raise ValueError(
                    f'Length of input z and lon-lat tuples not the same!')

    if plot_type == "scatter":
        x = dmap.coords["lon"]
        y = dmap.coords["lat"]
        z = dmap.data
    elif plot_type == "points":
        bar = False
        if not isinstance(dmap, xr.DataArray):
            x = dmap[:, 0]
            y = dmap[:, 1]
            if z is not None:
                plot_type = 'scatter'  # use scatter function of plot2D
        else:
            flat_idx_lst = gut.flatten_array(dataarray=dmap,
                                             mask=ds.mask,
                                             time=False,
                                             check=False)
            flat_idx = np.where(np.abs(flat_idx_lst) >
                                1e-5)[0]  # get all points >0
            gut.myprint('WARNING! Plot all points that are > 0!')
            x = []
            y = []

            for idx in flat_idx:
                map_idx = ds.get_map_index(idx)
                x.append(map_idx["lon"])
                y.append(map_idx["lat"])
    elif plot_type == 'colormesh':
        plot_type += '_map'

    # not to run into conflicts with significance mask
    pad = kwargs.pop('pad', -2.5)  # for maps
    im = plot_array(x=x, y=y, z=z,
                    fig=fig, ax=ax, plot_type=plot_type, projection=projection,
                    vmin=vmin, vmax=vmax, cmap=cmap, label=label, title=title,
                    alpha=alpha, pad=pad,
                    **kwargs)

    # areas which are dotted are mask
    if significance_mask is not None:
        if plot_type == 'points':
            gut.myprint(
                f'WARNING! plot_type {plot_type} cannot be used togehter with significance mask so far!')
        else:
            if isinstance(significance_mask, bool) and ds is None:
                raise ValueError(
                    'Significance mask set to true requires a dataset!')
            if significance_mask is False:
                raise ValueError(
                    'Significance mask has to be None, not False!')
            if ds is not None or inverse_mask is True:
                if ds is not None:
                    this_mask = ds.mask
                else:
                    this_mask = significance_mask
                significance_mask = xr.where(
                    this_mask == 1, False, True)  # Turn around the mask
                hatch_type = '///'
            if sig_plot_type == 'hatch':
                significance_mask = xr.where(significance_mask == 1, 1, np.nan)
            elif sig_plot_type == 'contour':
                significance_mask = xr.where(
                    significance_mask, 1, 0)
            if 'points' in list(significance_mask.dims):
                mask_dict = sput.interp2gaus(
                    dataarray=significance_mask, grid_step=grid_step)
                mask = mask_dict['intpol']
            else:
                mask = significance_mask

            if mask.shape != z.shape:
                raise ValueError(
                    'Significance mask not of same dimension as data input dimension!')

            if sig_plot_type == 'hatch':
                plot_array(x=x, y=y, z=mask, ax=im['ax'],
                           plot_type=sig_plot_type, alpha=0.0,
                           projection=projection,
                           hatch_type=hatch_type,
                           **kwargs)
            elif sig_plot_type == 'contour':
                color = kwargs.pop('color', 'black')
                levels = kwargs.pop('levels', 1)
                plot_array(x=x, y=y, z=mask, ax=im['ax'],
                           plot_type='contour',
                           levels=1,
                           vmin=0, vmax=1,
                           color='black',
                           projection=projection,
                           lw=2,
                           **kwargs)
    return im

# Plotting of 2D data (not necessary a map)


def plot_array(
    z=None,  # Not required for points
    x=None,
    y=None,
    fig=None,
    ax=None,
    plot_type="colormesh",
    projection=None,
    vmin=None,
    vmax=None,
    cmap='viridis',
    label=None,
    title=None,
    significance_mask=None,
    set_axis=True,
    **kwargs,
):
    reload(put)
    reload(sput)
    if x is None and y is None and z is None:
        raise ValueError(
            'Please provide at least z')
    if x is None and y is None:
        if isinstance(z, xr.DataArray):
            dims = gut.get_dims(z)
            # for plotting dimensions are transposed
            x = z.coords[dims[1]]
            y = z.coords[dims[0]]
        else:
            x = np.arange(0, len(z[0]))
            y = np.arange(0, len(z))
    # plotting
    color = kwargs.pop("color", None)
    cmap = None if color is not None else cmap
    alpha = kwargs.pop("alpha", 1.0)
    lw = kwargs.pop("lw", 1)
    size = kwargs.pop("size", 1)
    marker = kwargs.pop("marker", "o")
    fillstyle = kwargs.pop("fillstyle", "full")
    # such that setting to 1 will always be in the background
    zorder = kwargs.pop("zorder", 1)
    sci_x = kwargs.pop('sci_x', None)
    sci = kwargs.pop("sci", None)
    if ax is None:
        figsize = kwargs.pop("figsize", (7, 5))
        fig, ax = plt.subplots(figsize=(figsize))
        set_axis = True
    if projection is None and set_axis:
        # This is the Sci for the x axis
        ax, kwargs = put.prepare_axis(ax, sci=sci_x,
                                      **kwargs)
    if projection is not None:
        if isinstance(projection, str):
            projection = get_projection(projection='PlateCarree')

    set_norm = kwargs.pop('norm', None)
    levels = kwargs.pop("levels", None)
    if plot_type == 'contourf' and levels is None:
        levels = 8  # Default for contourf

    # Set 0.95 Quantile as definition for vmin, vmax to exclude outliers
    if vmin is not None:
        if not isinstance(vmin, float) and not isinstance(vmin, int):
            raise ValueError(
                f'vmin has to be of type float or int but is of type {type(vmin)}!')
    if vmax is not None:
        if not isinstance(vmax, float) and not isinstance(vmax, int):
            raise ValueError(
                f'Vmax has to be of type float or int but is of type {type(vmax)}!')

    if z is not None:
        vmin = np.nanquantile(z, q=0.1) if vmin is None else vmin
        vmax = np.nanquantile(z, q=0.9) if vmax is None else vmax
        if vmin == vmax and plot_type != 'hatch':
            raise ValueError(f'Vmin and vmax are equal: {vmin}!')
    else:
        vmin = vmax = None

    if set_norm == 'log':
        base = kwargs.pop('base', 2)
        # vmin and vmax are now the exponentials
        levels = np.logspace(vmin, vmax, levels + 1, base=base)
    elif levels is not None:
        levels = np.linspace(vmin, vmax, levels + 1, endpoint=True)
    round_dec = kwargs.pop("round_dec", None)
    if z is not None:
        if plot_type != 'hatch':
            expo = gut.get_exponent10(np.abs(vmax-vmin)) if float(
                vmin) != 0. else gut.get_exponent10(np.abs(vmax-vmin))
            if sci is None:
                sci = expo if np.abs(expo) > 1 else None
            if sci is not None:
                if sci < 0:
                    round_dec = abs(sci) + 1
                elif sci > 0:
                    round_dec = -1*(sci-2)
            else:
                if round_dec is None:
                    round_dec = 1 if vmax > 1 else 2  # because the range is between 0 and 10
            if levels is not None:
                this_levels = np.around(
                    levels, round_dec) if round_dec is not None else levels
                i = 1
                while gut.has_duplicate(this_levels):
                    this_levels = np.around(
                        levels, round_dec+i)
                    i += 1
                levels = this_levels
            if levels is not None and plot_type != 'points' and plot_type != 'contour' and cmap is not None:
                # norm = mpl.colors.LogNorm(levels=levels)
                centercolor = kwargs.pop('centercolor', None)
                leftcolor = kwargs.pop('leftcolor', None)
                cmap, norm = put.create_cmap(cmap, levels,
                                             centercolor=centercolor,
                                             leftcolor=leftcolor,
                                             **kwargs)
            else:
                if isinstance(cmap, str):
                    cmap, norm = put.create_cmap(cmap)
                # elif not isinstance(cmap, mpl.colors.Colormap) or not isinstance(cmap, mpl.colors.LinearSegmentedColormap) or not isinstance(color, str):
                #     raise ValueError(
                #         f'cmap has to be of type str or mpl.colors.Colormap but is of type {type(cmap)}!')
                norm = None
    else:
        sci = round_dec = norm = levels = None

    extend, kwargs = put.set_cb_boundaries(data=z, im=None,
                                           vmin=vmin, vmax=vmax, **kwargs)
    if plot_type == "scatter":
        im = ax.scatter(
            x=x,
            y=y,
            c=z,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            transform=projection,
            alpha=alpha,
            marker=mpl.markers.MarkerStyle(marker=marker, fillstyle=fillstyle),
            s=size,
        )
    elif plot_type == 'points' or z is None:
        im = ax.plot(
            x,
            y,
            c=color,
            linewidth=0,
            linestyle='None',
            markersize=size,
            marker=marker,
            fillstyle=fillstyle,
            transform=projection,
            alpha=alpha,
            zorder=zorder,
        )
    elif plot_type == "colormesh":
        im = ax.pcolor(
            x,
            y,
            z,
            cmap=cmap,
            # vmin=vmin,  # Is directly passed by the norm! Otherwise warning will arise!
            # vmax=vmax,
            shading="auto",
            norm=norm,
            zorder=zorder,
        )
    elif plot_type == "colormesh_map":
        if levels is not None:
            im = ax.pcolor(
                x,
                y,
                z,
                cmap=cmap,
                # vmin=vmin,   # Is directly passed by the norm! Otherwise warning will arise!
                # vmax=vmax,
                transform=projection,
                shading="auto",
                norm=norm,
                zorder=zorder,
            )
        else:
            im = ax.pcolor(
                x,
                y,
                z,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                transform=projection,
                shading="auto",
                zorder=zorder,
            )
    elif plot_type == "contourf" or plot_type == "discrete":
        """
        see https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.contourf.html
        """
        if plot_type == "discrete":
            vmin = np.nanmin(z)
            vmax = np.nanmax(z)
            normticks = put.discrete_norm_ticks(
                vmin, vmax, num_ticks=levels, shift_ticks=True)
            ticks = normticks[:-1] + 0.5
            levels = np.array(normticks[:], dtype=float)
        if cmap is None and color is None:
            raise ValueError(
                f'Please provide a cmap or color for plot_type {plot_type}!')
        if cmap is not None and color is not None:
            raise ValueError(
                f'Please provide either a cmap or color for plot_type {plot_type}!')
        im = ax.contourf(
            x,
            y,
            z,
            levels=levels,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            colors=color,  # color all levels with the same color, see documentation
            transform=projection,
            alpha=alpha,
            extend=extend,
            zorder=zorder,
        )
        if vmin is None:
            vmin = im.get_clim()[0]
        if vmax is None:
            vmax = im.get_clim()[1]
        # im = ScalarMappable(cmap=im.get_cmap())
    elif plot_type == "contour":
        cmap = cmap if color is None else None
        ls = kwargs.pop('ls', 'solid')
        clabel = kwargs.pop('clabel', False)

        if color == 'solid_dashed':
            colors = kwargs.pop('color_contour', ['black', 'black'])
            styles = ['solid', 'dashed']
            level_arr = [levels[levels > 0], levels[levels < 0]]
        else:
            colors = [color]
            styles = [ls]
            level_arr = [levels]

        for color, ls, this_levels in zip(colors, styles, level_arr):
            im = ax.contour(
                x,
                y,
                z,
                levels=this_levels,
                cmap=cmap,
                transform=projection,
                colors=color,  # color all levels with the same color, see documentation
                linewidths=lw,  # maybe linewidth=
                alpha=alpha,
                linestyles=ls,
                zorder=zorder,
            )
            if clabel:
                clabel_dict = dict(
                    fontsize=kwargs.pop('clabel_fsize', pst.MINI_SIZE),
                    fmt=kwargs.pop('clabel_fmt', None),
                    colors=color,
                    inline=True,
                    inline_spacing=8,
                    use_clabeltext=True,
                    rightside_up=True,
                )
                if clabel_dict['fmt'] is None:
                    del clabel_dict['fmt']
                ax.clabel(im, **clabel_dict)

    if plot_type == 'hatch' or significance_mask is not None:
        if significance_mask is not None:
            z = xr.where(significance_mask == 1, 1, np.nan)
        from matplotlib.colors import ListedColormap
        hatch_type = kwargs.pop('hatch_type', '///')
        if projection is None:
            im_tmp = ax.contourf(
                x,
                y,
                z,
                # vmin=vmin,
                # vmax=vmax,
                cmap=ListedColormap(['none']),
                # norm=norm,
                hatches=[hatch_type],
                alpha=0.0,
                zorder=10,
            )
        else:
            im_tmp = ax.pcolor(
                x,
                y,
                z,
                # tranform=None causes no plotting of hatches...
                transform=projection,
                cmap=ListedColormap(['none']),
                hatch=hatch_type,
                alpha=0.0,
                shading='nearest',
                zorder=10,
            )
        if plot_type == 'hatch':
            im = im_tmp
    # else:
    #     raise ValueError(f"Plot type {plot_type} does not exist!")
    if put.check_geoaxis(ax):
        y_title = kwargs.pop('y_title', 1.2)
    else:
        y_title = kwargs.pop('y_title', 1.05)
    kwargs = put.set_title(title=title, ax=ax,
                           y_title=y_title, **kwargs)

    if label is not None:
        tick_step = kwargs.pop('tick_step', 2)
        unset_sci = kwargs.pop('unset_sci', False)
        shift_ticks = kwargs.pop('shift_ticks', None)
        sci = sci if not unset_sci else None
        orientation = kwargs.pop('orientation', 'horizontal')
        cbar = put.make_colorbar(ax=ax, im=im,
                                 norm=set_norm,
                                 ticks=levels,
                                 label=label,
                                 set_cax=True,
                                 sci=sci,
                                 tick_step=tick_step if plot_type != 'discrete' else 1,
                                 orientation=orientation,
                                 shift_ticks=shift_ticks,
                                 **kwargs)
    if plot_type != "points":
        return {"ax": ax, "fig": fig, "projection": projection, "im": im,
                'ticks': levels, 'extend': extend}
    else:
        return {"ax": ax, "fig": fig, "projection": projection, "im": im}


def plot_edges(
    ds,
    edges,
    weights=None,
    central_longitude=None,
    fig=None,
    ax=None,
    projection="EqualEarth",
    plot_points=False,
    vmin=None,
    vmax=None,
    **kwargs,
):

    plot_grid = kwargs.pop("plot_grid", False)
    set_map = kwargs.pop("set_map", False)
    if ax is None:
        map_dict = create_map(
            da=ds.ds,
            ax=ax,
            projection=projection,
            central_longitude=central_longitude,
            plot_grid=plot_grid,
            set_map=set_map,
            **kwargs
        )
        ax = map_dict['ax']
        fig = map_dict['fig']
        kwargs = map_dict['kwargs']
    else:
        fig = ax.get_figure()
    counter = 0
    lw = kwargs.pop("lw", 1)
    alpha = kwargs.pop("alpha", 1)
    color = kwargs.pop("color", "gray")

    if vmin is None:
        vmin = np.min(weights)
    if vmax is None:
        vmax = np.max(weights)

    if weights is not None:
        cmap = plt.get_cmap(color)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    for i, (u, v) in enumerate(edges):
        counter += 1
        map_idx_u = ds.get_map_index(u)
        map_idx_v = ds.get_map_index(v)
        lon_u = map_idx_u["lon"]
        lat_u = map_idx_u["lat"]
        lon_v = map_idx_v["lon"]
        lat_v = map_idx_v["lat"]
        if plot_points is True:
            ax.scatter(
                [lon_u, lon_v], [lat_u, lat_v], c="k", transform=ccrs.PlateCarree(), s=4
            )
        if weights is not None:
            c = cmap(norm(weights[i]))
        else:
            c = color

        ax.plot(
            [lon_u, lon_v],
            [lat_u, lat_v],
            c=c,
            linewidth=lw,
            alpha=alpha,
            transform=ccrs.Geodetic(),
            zorder=10,
        )  # zorder = -1 to always set at the background

    gut.myprint(f"number of edges: {counter}")
    return {"ax": ax, "fig": fig, "projection": projection}


def plot_wind_field(
    u,
    v,
    ax=None,
    x_vals=None,
    y_vals=None,
    key_loc=(0.95, -.06),
    stream=False,
    transform=True,
    **kwargs,
):
    if ax is None:
        figsize = kwargs.pop("figsize", (7, 5))
        fig, ax = plt.subplots(figsize=(figsize))

    if x_vals is None:
        x_vals = u.coords["lon"]
    if y_vals is None:
        y_vals = u.coords["lat"]
    steps = kwargs.pop('steps', 1)
    x_steps = kwargs.pop('x_steps', 1)
    y_steps = kwargs.pop('y_steps', 1)

    if steps != 1:
        y_steps = x_steps = steps

    u_dat = u.data[::y_steps, ::x_steps]
    v_dat = v.data[::y_steps, ::x_steps]

    lw = kwargs.pop("lw", 4)
    scale = kwargs.pop("scale", None)
    # headwidth = kwargs.pop('headwidth', 5)
    # width = kwargs.pop('width', 0.006)
    headwidth = kwargs.pop('headwidth', 4.5)
    width = kwargs.pop('width', 0.003)
    headaxislength = kwargs.pop('headaxislength', 3)
    headlength = kwargs.pop('headlength', 4)
    zorder = kwargs.pop('zorder', 10)
    if stream:
        magnitude = (u ** 2 + v ** 2) ** 0.5
        im_stream = ax.streamplot(
            x_vals[::x_steps],
            y_vals[::y_steps],
            u_dat,
            v_dat,
            transform=ccrs.PlateCarree(),
            linewidth=lw,
            color=magnitude,
        )
    else:
        pivot = kwargs.pop("pivot", "middle")
        if transform:
            Q = ax.quiver(
                x_vals[::x_steps],
                y_vals[::y_steps],
                u_dat,
                v_dat,
                pivot=pivot,
                transform=ccrs.PlateCarree(),
                scale=scale,
                headwidth=headwidth,
                width=width,
                headaxislength=headaxislength,
                headlength=headlength,
                linewidth=lw,
                zorder=zorder,
            )
        else:
            Q = ax.quiver(
                x_vals[::x_steps],
                y_vals[::y_steps],
                u_dat,
                v_dat,
                headwidth=headwidth,
                pivot=pivot,
                scale=scale,
                width=width,
                headaxislength=headaxislength,
                headlength=headlength,
                linewidth=lw,
                zorder=zorder,
            )
        key_length = kwargs.pop('key_length', 1)
        wind_unit = kwargs.pop('wind_unit', r"ms$^{-1}$")
        if key_loc:
            ax.quiverkey(
                Q,
                key_loc[0],
                key_loc[1],
                key_length,
                f"{key_length} {wind_unit}",
                labelpos="W",
                coordinates="axes",
                zorder=pst.MAX_ZORDER,
            )
    return {'ax': ax}


def plot_trajectory(lon_lat_pairs,
                    vals=None,
                    ax=None,
                    fig=None,
                    vmin=None, vmax=None,
                    smooth_traj=False,
                    label=None,
                    lon_range=None,
                    lat_range=None,
                    **kwargs):
    # plotting parameters
    lw = kwargs.pop("lw", .5)
    alpha = kwargs.pop("alpha", 1)
    color = kwargs.pop("color", None)
    cmap = kwargs.pop("cmap", None)

    if isinstance(lon_lat_pairs, list):
        lon_lat_pairs = np.array(lon_lat_pairs)

    lons = lon_lat_pairs[:, 0]
    lats = lon_lat_pairs[:, 1]

    if ax is None:
        central_longitude = kwargs.pop('central_longitude', 0)
        projection = kwargs.pop('projection', 'PlateCarree')
        map_dict = create_map(da=None,
                              lat_range=lat_range,
                              lon_range=lon_range,
                              central_longitude=central_longitude,
                              projection=projection,
                              **kwargs)
        ax = map_dict['ax']
        title = kwargs.pop('title', None)
        y_title = kwargs.pop('y_title', 1.18)
        kwargs = put.set_title(title=title, ax=ax,
                               y_title=y_title,
                               **kwargs)

    if cmap is not None:
        cmap = plt.get_cmap(cmap)
        if vals is not None:
            vmin = np.nanquantile(vals, q=0.05) if vmin is None else vmin
            vmax = np.nanquantile(vals, q=0.95) if vmax is None else vmax
        else:
            # plot as a line with color, where color denotes the sequence of the trajectory
            vals = np.arange(len(lon_lat_pairs)-1)
            vmin = 0
            vmax = len(lon_lat_pairs)-1
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
        for i, lon_lat in enumerate(lon_lat_pairs[:-1]):
            lon_lat_pair = np.array([lon_lat, lon_lat_pairs[i+1]])
            traj_plot = dict(lon=lon_lat_pair[:, 0], lat=lon_lat_pair[:, 1])
            c = cmap.to_rgba(vals[i])

            lon_traj = traj_plot['lon']
            if np.abs(np.max(lon_traj) - np.min(lon_traj)) > 300:
                lon_traj = sput.lon2_360(lon_traj)

            X = lon_traj
            Y = traj_plot['lat']

            ax.plot(
                X, Y,
                transform=ccrs.PlateCarree(central_longitude=0),
                c=c, linewidth=lw, alpha=alpha)

        label = 'Trajectory' if label is None else label
        cbar = put.make_colorbar(ax,
                                 im=cmap,
                                 norm=True,
                                 label=label,
                                 set_cax=True,
                                 **kwargs)

    else:
        X, Y = lons, lats
        if smooth_traj:
            from scipy.interpolate import splprep, splev
            tck, _ = splprep([X, Y], s=0, per=False)
            X, Y = splev(np.linspace(0, 1, 1000), tck, der=0)

        ax.plot(X,
                Y,
                transform=ccrs.PlateCarree(central_longitude=0),
                c=color, linewidth=lw, alpha=alpha)

    return dict(
        ax=ax,
        fig=fig,
    )


def create_multi_plot(nrows, ncols, projection=None,
                      lon_range=None, lat_range=None,
                      plot_grid=False, **kwargs):
    reload(put)
    figsize = kwargs.pop('figsize', None)
    if figsize is None:
        figsize = (5*ncols, 5*nrows)

    end_idx = kwargs.pop('end_idx', None)
    end_idx = int(nrows*ncols) if end_idx is None else end_idx
    full_length_row = kwargs.pop('full_length_row', False)
    map_axis = kwargs.pop('map_axis', [])
    if len(map_axis) == 0:
        map_axis = np.arange(nrows*ncols)
    if nrows*ncols/end_idx >= nrows:
        nrows = nrows - 1 if nrows > 1 else nrows

    ratios_w = kwargs.pop('ratios_w', np.ones(ncols))
    ratios_h = kwargs.pop('ratios_h', np.ones(nrows))
    gs_rows = nrows
    gs_cols = ncols
    if len(ratios_w) != ncols:
        raise ValueError(
            f'Length of ratios_w {len(ratios_w)} does not match number of columns {ncols}!')
    if len(ratios_h) != nrows:
        raise ValueError(
            f'Length of ratios_h {len(ratios_h)} does not match number of rows {nrows}!')
    fig = plt.figure(figsize=(figsize[0], figsize[1]))

    hspace = kwargs.pop('hspace', 0.)
    wspace = kwargs.pop('wspace', 0.)

    gs = fig.add_gridspec(gs_rows, gs_cols,
                          height_ratios=ratios_h,
                          width_ratios=ratios_w,
                          hspace=hspace, wspace=wspace)
    proj_arr = kwargs.pop('proj_arr', None)
    if proj_arr is None:
        proj_arr = kwargs.pop('projection_arr', None)

    diff_projs = False

    if projection is not None or proj_arr is not None:
        central_longitude = kwargs.pop("central_longitude", None)
        central_latitude = kwargs.pop("central_latitude", None)
        dateline = kwargs.pop('dateline', False)
        dateline_arr = kwargs.pop('dateline_arr', None)
        if proj_arr is None:
            proj_arr = gut.replicate_object(projection, nrows*ncols)
        else:
            diff_projs = True
            if len(proj_arr) > end_idx:
                raise ValueError(
                    f'Length of projection array {len(proj_arr)} does not match number of pannels {end_idx}!')
    else:
        proj_arr = gut.replicate_object(None, nrows*ncols)
    axs = []

    set_map = kwargs.pop('set_map', False)  # is set later
    run_idx = 0
    for i in range(nrows):
        for j in range(ncols):
            if gut.has_non_none_objects(proj_arr):
                if dateline_arr is not None:
                    dateline = dateline_arr[run_idx]
                projection = proj_arr[run_idx]
                if projection is not None:
                    proj = get_projection(projection=projection,
                                          central_longitude=central_longitude,
                                          central_latitude=central_latitude,
                                          dateline=dateline)

                    if run_idx == end_idx-1 and full_length_row:
                        axs.append(fig.add_subplot(gs[i, j:], projection=proj,
                                                   ))
                    else:
                        axs.append(fig.add_subplot(gs[i, j], projection=proj))
                    if run_idx in map_axis:
                        map_dict = create_map(
                            ax=axs[run_idx],
                            projection=projection,
                            central_longitude=central_longitude,
                            plot_grid=plot_grid,
                            lon_range=lon_range,
                            lat_range=lat_range,
                            dateline=dateline,
                            set_map=set_map,
                            **kwargs
                        )
                        kwargs = map_dict['kwargs']
                else:
                    # Make a row at full length until the end of the columns
                    if run_idx == end_idx-1 and full_length_row:
                        axs.append(fig.add_subplot(gs[i, j:]))
                    else:
                        axs.append(fig.add_subplot(gs[i, j]))
            else:
                axs.append(fig.add_subplot(gs[i, j]))

            run_idx += 1
            if run_idx >= end_idx:
                break
    # fig.tight_layout()
    if nrows > 1 or ncols > 1:
        enumerate_subplots = kwargs.pop('enumerate_subplots', True)
        if enumerate_subplots:
            pos_x = kwargs.pop('pos_x', -0.1)
            pos_y = kwargs.pop('pos_y', 1.1)
            if diff_projs and isinstance(pos_x, (int, float)):
                pos_x_proj = kwargs.pop('pos_x_proj', pos_x)
                pos_x = gut.replicate_object(pos_x, end_idx)
                indices_proj = gut.get_not_None_indices(proj_arr)
                pos_x[indices_proj] = pos_x_proj
            put.enumerate_subplots(axs, pos_x=pos_x, pos_y=pos_y,
                                   )
    else:
        axs = axs[0]

    y_title_pos = 1.1 - 0.05*(nrows-1)
    title = kwargs.pop('title', None)
    y_title = kwargs.pop('y_title', y_title_pos)
    if title is not None:
        put.set_title(title=title, ax=None, fig=fig,
                      y_title=y_title)

    return {"ax": axs, "fig": fig, "projection": projection}


def plt_text_map(ax, lon_pos, lat_pos, text, color="k"):
    ax.text(
        lon_pos,
        lat_pos,
        text,
        horizontalalignment="center",
        transform=ccrs.Geodetic(),
        color=color,
    )

    return ax


def plot_rectangle(ax, lon_range, lat_range, text=None, **kwargs):
    """Plots a rectangle on a cartopy map

    Args:
        ax (geoaxis): Axis of cartopy object
        lon_range (list): list of min and max longitude
        lat_range (list): list of min and max lat

    Returns:
        geoaxis: axis with rectangle plotted
    """
    from shapely.geometry.polygon import LinearRing

    shortest = kwargs.pop("shortest", True)
    if (max(lon_range) - min(lon_range) <= 180) or shortest is False:
        cl = 0
        lons = [max(lon_range), min(lon_range), min(lon_range), max(lon_range)]
    else:
        cl = 180
        lons = [
            max(lon_range) - 180,
            180 + min(lon_range),
            180 + min(lon_range),
            max(lon_range) - 180,
        ]
    lats = [min(lat_range), min(lat_range), max(lat_range), max(lat_range)]

    ring = LinearRing(list(zip(lons, lats)))
    lw = kwargs.pop("lw", 1)
    ls = kwargs.pop("ls", "-")
    color = kwargs.pop("color", "k")
    fill = kwargs.pop("fill", False)
    facecolor = color if fill else "none"
    zorder = kwargs.pop('zorder', 11)
    ax.add_geometries(
        [ring],
        ccrs.PlateCarree(central_longitude=cl),
        facecolor=facecolor,
        edgecolor=color,
        linewidth=lw,
        zorder=zorder,
        linestyle=ls,
    )

    if text is not None:
        put.plt_text(ax=ax,
                     text=text,
                     color=color,
                     xpos=np.mean(lon_range),
                     # always plot above(!) the rectangle
                     ypos=np.max(lat_range)+2,
                     zorder=zorder+1,
                     )

    return ax


def plot_ring(ax, xpos=0, ypos=0, **kwargs):
    """Plots a rectangle on a cartopy map

    Args:
        ax (geoaxis): Axis of cartopy object
        lon_range (list): list of min and max longitude
        lat_range (list): list of min and max lat

    Returns:
        geoaxis: axis with rectangle plotted
    """
    lw = kwargs.pop("lw", 1)
    color = kwargs.pop("color", "k")
    fill = kwargs.pop("fill", False)
    facecolor = color if fill else "none"
    width = kwargs.pop('width', 2)  # Which makes a unit circle of radius 1
    height = kwargs.pop('height', 2)

    ring = mpatches.Ellipse(
        xy=(xpos, ypos),
        facecolor=facecolor,
        edgecolor=color,
        linewidth=lw,
        width=width,
        height=height,
        zorder=10,)

    ax.add_patch(ring)

    return ax


def plot_horizontal_line_at_latitude(ax, latitude, **kwargs):
    """
    Plot a horizontal line at a specified latitude on a Cartopy map.

    Parameters:
        latitude (float): The latitude coordinate (in degrees) where the horizontal line will be plotted.

    Returns:
        None
    """
    # Plot a horizontal line at the specified latitude
    lats = np.ones(100) * latitude
    lons = np.linspace(-180, 180, 100)
    ax.plot(lons, lats, zorder=20,
            transform=ccrs.PlateCarree(), **kwargs)

    return ax
