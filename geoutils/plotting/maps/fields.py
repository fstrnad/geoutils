"""Plotting of gridded fields and scattered points on maps and axes."""
# import matplotlib.cm as cm
from dataclasses import dataclass
import geoutils.utils.spatial_utils as sput
import geoutils.utils.general_utils as gut
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import geoutils.plotting.axes as pax
import geoutils.plotting.colors as pcol
import geoutils.plotting.style as pst
from geoutils.plotting.result import PlotResult
from geoutils.plotting.maps.projections import get_projection
from geoutils.plotting.maps.basemap import create_map




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
             mask: xr.DataArray = None,
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
      mask (xr.DataArray): Mask to apply on the plot. 1 for masked data
      lat_range (Tuple[float, float]): Latitude range for the plot.
      lon_range (Tuple[float, float]): Longitude range for the plot.
      **kwargs: Additional keyword arguments.
    """

    plt.rcParams["pcolor.shading"] = "nearest"  # For pcolormesh
    if label is None and isinstance(dmap, xr.DataArray):
        unset_label = kwargs.pop('unset_label', True)
        label = dmap.name if not unset_label else None
    if isinstance(dmap, xr.Dataset):
        dmap = dmap[list(dmap.data_vars)[0]]  # take the first variable
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
        if mask is not None:
            mask = sput.check_dimensions(mask,
                                         transpose_dims=True,
                                         verbose=False)
            dmap = xr.where(mask, dmap, np.nan)
            significance_mask = xr.where(mask, np.nan, 1)  # flip the mask

    check_plot_type(plot_type)
    projection = check_projection(ax=ax, projection=projection)

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
    projection = ccrs.PlateCarree()  # note: do NOT pass central_longitude here!

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


@dataclass
class _DrawContext:
    """Mutable state shared between plot_array and its per-type draw helpers."""
    x: object
    y: object
    z: object
    cmap: object
    norm: object
    vmin: object
    vmax: object
    levels: object
    extend: object
    projection: object
    color: object
    alpha: float
    lw: float
    size: float
    marker: str
    fillstyle: str
    zorder: int
    kwargs: dict


def _draw_scatter(ax, c):
    return ax.scatter(
        x=c.x, y=c.y, c=c.z,
        vmin=c.vmin, vmax=c.vmax, cmap=c.cmap,
        transform=c.projection, alpha=c.alpha,
        marker=mpl.markers.MarkerStyle(marker=c.marker, fillstyle=c.fillstyle),
        s=c.size,
    )


def _draw_points(ax, c):
    return ax.plot(
        c.x, c.y, c=c.color, linewidth=0, linestyle='None',
        markersize=c.size, marker=c.marker, fillstyle=c.fillstyle,
        transform=c.projection, alpha=c.alpha, zorder=c.zorder,
    )


def _draw_colormesh(ax, c):
    return ax.pcolor(
        c.x, c.y, c.z, cmap=c.cmap, shading="auto", norm=c.norm, zorder=c.zorder,
    )


def _draw_colormesh_map(ax, c):
    if c.levels is not None:
        return ax.pcolor(
            c.x, c.y, c.z, cmap=c.cmap, transform=c.projection,
            shading="auto", norm=c.norm, zorder=c.zorder,
        )
    return ax.pcolor(
        c.x, c.y, c.z, cmap=c.cmap, vmin=c.vmin, vmax=c.vmax,
        transform=c.projection, shading="auto", zorder=c.zorder,
    )


def _draw_contourf(ax, c):
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contourf.html
    if c.cmap is None and c.color is None:
        raise ValueError('Please provide a cmap or color for contourf/discrete!')
    if c.cmap is not None and c.color is not None:
        raise ValueError(
            'Please provide either a cmap or color for contourf/discrete!')
    im = ax.contourf(
        c.x, c.y, c.z, levels=c.levels, norm=c.norm, vmin=c.vmin, vmax=c.vmax,
        cmap=c.cmap, colors=c.color, transform=c.projection, alpha=c.alpha,
        extend=c.extend, zorder=c.zorder,
    )
    if c.vmin is None:
        c.vmin = im.get_clim()[0]
    if c.vmax is None:
        c.vmax = im.get_clim()[1]
    return im


def _draw_discrete(ax, c):
    c.vmin = np.nanmin(c.z)
    c.vmax = np.nanmax(c.z)
    normticks = pcol.discrete_norm_ticks(
        c.vmin, c.vmax, num_ticks=c.levels, shift_ticks=True)
    c.levels = np.array(normticks[:], dtype=float)
    return _draw_contourf(ax, c)


def _draw_contour(ax, c):
    cmap = c.cmap if c.color is None else None
    ls = c.kwargs.pop('ls', 'solid')
    clabel = c.kwargs.pop('clabel', False)
    if c.color == 'solid_dashed':
        colors = c.kwargs.pop('color_contour', ['black', 'black'])
        styles = ['solid', 'dashed']
        level_arr = [c.levels[c.levels > 0], c.levels[c.levels < 0]]
    else:
        colors = [c.color]
        styles = [ls]
        level_arr = [c.levels]
    im = None
    for color, ls, this_levels in zip(colors, styles, level_arr):
        im = ax.contour(
            c.x, c.y, c.z, levels=this_levels, cmap=cmap,
            transform=c.projection, colors=color, linewidths=c.lw,
            alpha=c.alpha, linestyles=ls, zorder=c.zorder,
        )
        if clabel:
            clabel_dict = dict(
                fontsize=c.kwargs.pop('clabel_fsize', pst.MINI_SIZE),
                fmt=c.kwargs.pop('clabel_fmt', None),
                colors=color, inline=True, inline_spacing=8,
                use_clabeltext=True, rightside_up=True,
            )
            if clabel_dict['fmt'] is None:
                del clabel_dict['fmt']
            ax.clabel(im, **clabel_dict)
    return im


# scatter/points are resolved separately because `z is None` also routes to
# points; the rest dispatch straight off plot_type.


_FIELD_DRAWERS = {
    "colormesh": _draw_colormesh,
    "colormesh_map": _draw_colormesh_map,
    "contourf": _draw_contourf,
    "discrete": _draw_discrete,
    "contour": _draw_contour,
}


def _resolve_drawer(plot_type, z):
    """Pick the draw helper, preserving plot_array's original precedence."""
    if plot_type == "scatter":
        return _draw_scatter
    if plot_type == "points" or z is None:
        return _draw_points
    return _FIELD_DRAWERS.get(plot_type)


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
    color_bounds='quantile',
    **kwargs,
):
    """Draw a 2D array (or scattered points) onto an axis.

    Parameters
    ----------
    color_bounds : {'quantile', 'minmax', 'absmax', 'absquantile'}
        How vmin/vmax are derived when not passed explicitly. NOTE: the default
        ``'quantile'`` clips the colour scale to the 10th-90th percentile of the
        data, so extreme values saturate unless you pass vmin/vmax or
        ``color_bounds='minmax'``.
    """
    if x is None and y is None and z is None:
        raise ValueError(
            'Please provide at least z')
    if z is not None and np.isnan(z).all():
        raise ValueError('z is all-NaN, nothing to plot!')
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
    centercolor = kwargs.pop('centercolor', None)
    leftcolor = kwargs.pop('leftcolor', None)
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
        ax, kwargs = pax.prepare_axis(ax, sci=sci_x,
                                      **kwargs)
    if projection is not None:
        if isinstance(projection, str):
            projection = get_projection(projection='PlateCarree')

    set_norm = kwargs.pop('norm', None)
    levels = kwargs.pop("levels", None)
    norm = None  # reassigned below when applicable; defined for all types
    if plot_type == 'contourf' and levels is None:
        levels = 8  # Default for contourf

    scale = pcol.resolve_color_scale(
        z, vmin=vmin, vmax=vmax, levels=levels, color_bounds=color_bounds,
        extend=kwargs.pop('extend', None),
        sci=sci, round_dec=kwargs.pop("round_dec", None),
        log_base=kwargs.pop('base', 2) if set_norm == 'log' else None,
        is_hatch=plot_type == 'hatch')
    vmin, vmax, levels, extend, sci = (
        scale.vmin, scale.vmax, scale.levels, scale.extend, scale.sci)
    kwargs['extend'] = extend
    if z is not None and plot_type != 'hatch':
        if levels is not None and plot_type not in ('points', 'contour') and cmap is not None:
            cmap, norm = pcol.create_cmap(cmap, levels,
                                          centercolor=centercolor,
                                          leftcolor=leftcolor,
                                          **kwargs)
        elif isinstance(cmap, str):
            cmap, _ = pcol.create_cmap(cmap)
    draw = _resolve_drawer(plot_type, z)
    if draw is not None:
        ctx = _DrawContext(
            x=x, y=y, z=z, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
            levels=levels, extend=extend, projection=projection, color=color,
            alpha=alpha, lw=lw, size=size, marker=marker, fillstyle=fillstyle,
            zorder=zorder, kwargs=kwargs,
        )
        im = draw(ax, ctx)
        levels, vmin, vmax = ctx.levels, ctx.vmin, ctx.vmax
    else:
        im = None

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
    if pax.check_geoaxis(ax):
        # set distance of plot to the title
        y_title = kwargs.pop('y_title', 1.15)
    else:
        y_title = kwargs.pop('y_title', 1.05)
    kwargs = pax.set_title(title=title, ax=ax,
                           y_title=y_title, **kwargs)

    if label is not None:
        tick_step = kwargs.pop('tick_step', 2)
        unset_sci = kwargs.pop('unset_sci', False)
        shift_ticks = kwargs.pop('shift_ticks', None)
        sci = sci if not unset_sci else None
        orientation = kwargs.pop('orientation', 'horizontal')
        cbar = pcol.make_colorbar(ax=ax, im=im,
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
        return PlotResult(ax=ax, fig=fig, projection=projection, im=im,
                          ticks=levels, extend=extend)
    else:
        return PlotResult(ax=ax, fig=fig, projection=projection, im=im)


def check_plot_type(plot_type):
    avail_types = ['contour', 'contourf', 'scatter',
                   'points', 'colormesh', 'discrete']
    if plot_type not in avail_types:
        raise ValueError(f'ERROR plot_type {plot_type} not available!')
    return True


def check_projection(ax, projection):
    if ax is not None and projection is not None:
        raise ValueError(
            f'Axis already given, projection {projection} will have no effect.')
    else:
        # Set default to PlateCarree
        projection = 'PlateCarree' if projection is None else projection
    return projection
