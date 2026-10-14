"""Base map creation: extent, gridlines, coastlines, scale bar."""
# import matplotlib.cm as cm
import geoutils.utils.spatial_utils as sput
import geoutils.utils.general_utils as gut
import numpy as np
import xarray as xr
import cartopy as ctp
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import geoutils.plotting.style as pst
from geoutils.plotting.maps.projections import get_projection

# keyword options understood by create_map / set_extent / set_grid
BASEMAP_OPTIONS = frozenset({
    'central_latitude', 'coast_color', 'figsize', 'land_ocean', 'lw_borders',
    'plot_borders', 'rem_frame', 'set_global', 'set_map', 'unset_grid',
    'verbose', 'gs_lat', 'gs_lon',
})




def central_longitude_of(projection):
    """Central longitude of a cartopy CRS.

    PlateCarree reports it as the prime meridian ``pm`` instead of ``lon_0``
    from cartopy 0.26 on.
    """
    params = projection.proj4_params
    return params.get('lon_0', params.get('pm', 0))


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
        linewidth=0.4,
        linestyle=":",
        color="0.35",
    )
    # gl = ax.gridlines(draw_labels=True, dms=True,
    #                   x_inline=False, y_inline=False, )
    # gl.rotate_labels = False
    gl.left_labels = True
    gl.right_labels = False
    # gl.right_labels=True
    gl.bottom_labels = True
    gl.ylabel_style = {'rotation': -0,
                       'color': pst.INK, 'size': pst.TICK_SIZE}
    gl.xlabel_style = {'rotation': -0,
                       'color': pst.INK, 'size': pst.TICK_SIZE}
    gl.top_labels = False

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
        if central_longitude_of(ax.projection) != 180:
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


@pst.styled
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
    figsize = kwargs.pop("figsize", (6, 4))
    # create figure
    if ax is None:
        # better working in cartopy and matplotlib=3.8.2
        fig = plt.figure(figsize=figsize, layout="constrained")
        ax = fig.add_subplot(projection=proj)
        unset_grid = kwargs.pop('unset_grid', False)
        plot_grid = True if not unset_grid else False
    else:
        ax_central_longitude = central_longitude_of(ax.projection)
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
        coast_color = kwargs.pop("coast_color", None)
        plot_borders = kwargs.pop("plot_borders", False)
        lw_borders = kwargs.pop("lw_borders", None)
        # fine dark-grey coastlines; borders one step lighter and thinner
        ax.coastlines(alpha=alpha,
                      color=coast_color or "#333333",
                      lw=0.6 if lw_borders is None else lw_borders
                      )
        if plot_borders:
            ax.add_feature(ctp.feature.BORDERS,
                           linestyle="-",
                           color=coast_color or "#666666",
                           lw=0.4 if lw_borders is None else lw_borders,
                           alpha=alpha)
        land_ocean = kwargs.pop("land_ocean", False)
        if land_ocean:
            # ax.add_feature(ctp.feature.OCEAN, alpha=.4, zorder=-1)
            ax.add_feature(ctp.feature.LAND, alpha=.3, zorder=-1, color='grey')
        if projection == "PlateCarree":
            rem_frame = kwargs.pop("rem_frame", False)
            if rem_frame:
                # Remove the frame around the map
                ax.spines['geo'].set_visible(False)

    if plot_grid:
        ax, kwargs = set_grid(ax,
                              proj=projection,
                              ext_dict=ext_dict,
                              **kwargs)

    return dict(ax=ax, fig=fig, kwargs=kwargs)
