"""Basic plotting functions for maps"""
# import matplotlib.cm as cm
import geoutils.utils.spatial_utils as sput
import matplotlib.patches as mpatches
import geoutils.utils.general_utils as gut
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy as ctp
from importlib import reload

import geoutils.plotting.plotting_utils as put
import geoutils.plotting.plot_settings as pst
reload(put)
reload(pst)


def set_grid(ax, alpha=0.5, **kwargs):
    gs_lon = kwargs.pop('gs_lon', 60)
    gs_lat = kwargs.pop('gs_lat', 30)
    gl = ax.gridlines(
        draw_labels=True,
        xlocs=np.arange(-180, 181, gs_lon),
        ylocs=range(-90, 90, gs_lat),
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
    return ax


def set_extent(da, ax, **kwargs):
    if not isinstance(ax, ctp.mpl.geoaxes.GeoAxesSubplot) and not isinstance(ax, ctp.mpl.geoaxes.GeoAxes):
        raise ValueError(
            f'Axis is not of type Geoaxis, but of type {type(ax)}!')

    projection = ccrs.PlateCarree()

    min_ext_lon = float(np.min(da.coords["lon"]))
    max_ext_lon = float(np.max(da.coords["lon"]))
    min_ext_lat = float(np.min(da.coords["lat"]))
    max_ext_lat = float(np.max(da.coords["lat"]))

    set_global = kwargs.pop('set_global', False)
    if not set_global:
        if abs(min_ext_lon) > 179 or abs(max_ext_lon) > 179:
            set_global = True
        if abs(min_ext_lat) > 89 or abs(max_ext_lat) > 89:
            set_global = True

    if set_global:
        ax.set_global()
    else:
        # min_ext_lon = math.floor(np.min(lon_range))
        # max_ext_lon = math.floor(np.max(lon_range))
        # min_ext_lat = math.floor(np.min(lat_range))
        # max_ext_lat = math.ceil(np.max(lat_range))

        ax.set_extent(
            [min_ext_lon, max_ext_lon, min_ext_lat, max_ext_lat], crs=projection
        )
        # print(
        #     [min_ext_lon,
        #      max_ext_lon,
        #      min_ext_lat,
        #      max_ext_lat]
        # )

    return ax


def create_map(
    da=None,
    ax=None,
    fig=None,
    projection="EqualEarth",
    central_longitude=None,
    alpha=1,
    plt_grid=False,
    **kwargs,
):
    proj = get_projection(projection, central_longitude)

    # create figure
    if ax is None:
        figsize = kwargs.pop("figsize", (9, 6))
        fig, ax = plt.subplots(figsize=(figsize))
        ax = plt.axes(projection=proj)
    else:
        ax_central_longitude = ax.projection.proj4_params['lon_0']
        if central_longitude is not None:
            if central_longitude != ax_central_longitude:
                gut.myprint(
                    f'WARNING! Central longitude set to {central_longitude} but has no effect since axis argument is passed is {ax_central_longitude}!')

    set_map = kwargs.pop('set_map', True)
    ax = set_extent(
        da=da, ax=ax, **kwargs)

    if set_map:
        # axes properties
        # coast_color = kwargs.pop("coast_color", "k")
        ax.coastlines(alpha=alpha,
                      #   color=coast_color
                      )
        ax.add_feature(ctp.feature.BORDERS,
                       linestyle=":",
                       #    color="grey",
                       alpha=alpha)
        land_ocean = kwargs.pop("land_ocean", False)
        if land_ocean:
            ax.add_feature(ctp.feature.OCEAN, alpha=alpha, zorder=-1)
            ax.add_feature(ctp.feature.LAND, alpha=alpha, zorder=-1)
    if plt_grid is True:
        ax = set_grid(ax, alpha=alpha, **kwargs)

    return ax, fig


def get_projection(projection, central_longitude=None):
    central_longitude = 0 if central_longitude is None else central_longitude
    if not isinstance(central_longitude, float) and not isinstance(central_longitude, int):
        raise ValueError(
            f'central_longitude is not of type int or float, but of type {type(central_longitude)}!'
        )
    if projection == "Mollweide":
        proj = ccrs.Mollweide(central_longitude=central_longitude)
    elif projection == "EqualEarth":
        proj = ccrs.EqualEarth(central_longitude=central_longitude)
    elif projection == "Robinson":
        proj = ccrs.Robinson(central_longitude=central_longitude)
    elif projection == "PlateCarree":
        proj = ccrs.PlateCarree(central_longitude=central_longitude)
    else:
        raise ValueError(f"This projection {projection} is not available yet!")

    return proj


def plot_map(
    dmap,
    fig=None,
    ax=None,
    ds=None,
    plot_type="contourf",
    central_longitude=None,
    vmin=None,
    vmax=None,
    cmap='coolwarm',
    bar=True,
    projection=None,
    label=None,
    title=None,
    significance_mask=None,
    **kwargs,
):
    """Plotting functions to plot an xarray dataarray as a map

    Args:
        dmap (xr.dataarray): dataarray to be plotted.
        fig (mpl, optional): mpl object of the figure. Defaults to None.
        ax (mpl.ax, optional): axis object of mpl. Defaults to None.
        ds (climnet.dataset object, optional): climnet dataset object. Defaults to None.
        plot_type (str, optional): Which plot interpolation. Defaults to "contourf".
        central_longitude (int, optional): Central longitude. Defaults to 0.
        vmin (float, optional): color bar minimum. Defaults to None.
        vmax (float, optional): color bar maximum. Defaults to None.
        cmap (str, optional): Color map. Defaults to 'coolwarm'.
        bar (bool, optional): Plot as well color maps. Defaults to True.
        projection (str, optional): Which map projection to use. Defaults to "PlateCarree".
        label (str, optional): label of the colorbar. Defaults to None.
        title (str, optional): title of the plot. Defaults to None.
        significance_mask (xr.dataarray, optional): dataarray with the mask as 1. Defaults to None.
        ds_mask (bool, optional): use the mask of the dataset as mask. Defaults to False.

    Raises:
        ValueError:

    Returns:
        dict: dictionary of ax, fig, colorbar object, ticks.
    """
    reload(put)
    reload(sput)
    plt.rcParams["pcolor.shading"] = "nearest"  # For pcolormesh

    plt_grid = kwargs.pop("plt_grid", True)
    set_global = kwargs.pop('set_global', False)
    hatch_type = kwargs.pop('hatch_type', '..')
    set_map = kwargs.pop('set_map', True)
    figsize = kwargs.pop("figsize", (9, 6))
    if ax is not None and projection is not None:
        raise ValueError(
            f'Axis already given, projection {projection} will have no effect. Please do not pass projection argument!')
    else:
        if projection is None:
            projection = 'PlateCarree'  # Set default to PlateCarree
    ax, fig = create_map(
        da=dmap,
        ax=ax,
        projection=projection,
        central_longitude=central_longitude,
        plt_grid=plt_grid,
        set_global=set_global,
        set_map=set_map,
        figsize=figsize,
        **kwargs
    )

    projection = ccrs.PlateCarree()  # nicht: central_longitude=central_longitude!

    if bar == "discrete":
        kwargs_pl = dict()  # kwargs plot function
        kwargs_cb = dict()  # kwargs colorbar
        normticks = np.arange(0, dmap.max(skipna=True) + 2, 1)
        kwargs_pl["norm"] = mpl.colors.BoundaryNorm(normticks, cmap.N)
        kwargs_cb["ticks"] = normticks + 0.5

    # interpolate grid of points to regular grid
    grid_step = kwargs.pop('grid_step', 1)  # by default 1 grid

    if 'points' in list(dmap.dims):
        if plot_type != 'points':
            if ds is not None:
                grid_step = ds.grid_step
            dmap_dict = sput.interp2gaus(dataarray=dmap, grid_step=grid_step)
            dmap = dmap_dict['intpol']
    x, y = dmap.coords["lon"], dmap.coords["lat"]
    z = dmap

    if plot_type == "scatter":
        x = dmap.coords["lon"]
        y = dmap.coords["lat"]
        z = dmap.data
    elif plot_type == "points":
        bar = False
        if ds is None:
            raise ValueError(
                f'For plot_type == points a dataset object is needed!')
        flat_idx_lst = sput.flatten_array(dataarray=dmap,
                                          mask=ds.mask,
                                          time=False,
                                          check=False)
        flat_idx = np.where(np.abs(flat_idx_lst) >
                            1e-5)[0]  # get all points >0
        print('WARNING! Plot all points that are > 0!')
        x = []
        y = []

        for idx in flat_idx:
            map_idx = ds.get_map_index(idx)
            x.append(map_idx["lon"])
            y.append(map_idx["lat"])
    elif plot_type == 'colormesh':
        plot_type += '_map'

    # not to run into conflicts with significance mask
    alpha = kwargs.pop("alpha", 1.0)
    im = plot_2D(x=x, y=y, z=z,
                 fig=fig, ax=ax, plot_type=plot_type, projection=projection,
                 vmin=vmin, vmax=vmax, cmap=cmap, label=label, title=title,
                 alpha=alpha,
                 **kwargs)

    # areas which are dotted are mask
    if significance_mask is not None:
        if isinstance(significance_mask, bool) and ds is None:
            raise ValueError(
                'Significance mask set to true requires a dataset!')
        if significance_mask is False:
            raise ValueError('Significance mask has to be None, not False!')
        if ds is not None:
            significance_mask = xr.where(
                ds.mask, False, True)  # Turn around the mask
            hatch_type = '///'
            gut.myprint(f'WARNING! So far the dataset mask is only plotted as significnance mask!',
                        verbose=False)
        significance_mask = xr.where(significance_mask == 1, 1, np.nan)
        if 'points' in list(significance_mask.dims):
            mask_dict = sput.interp2gaus(
                dataarray=significance_mask, grid_step=grid_step)
            mask = mask_dict['intpol']
        else:
            mask = significance_mask

        if mask.shape != z.shape:
            raise ValueError(
                'Significance mask not of same dimension as data input dimension!')

        plot_2D(x=x, y=y, z=mask, ax=im['ax'],
                plot_type='hatch', alpha=0.0,
                projection=projection,
                hatch_type=hatch_type,
                **kwargs)

    return im


def plot_2D(
    x,
    y,
    z,
    fig=None,
    ax=None,
    plot_type="contourf",
    projection=None,
    vmin=None,
    vmax=None,
    cmap='coolwarm',
    label=None,
    title=None,
    significance_mask=None,
    **kwargs,
):
    reload(put)
    reload(sput)
    if ax is None:
        figsize = kwargs.pop("figsize", (7, 5))
        fig, ax = plt.subplots(figsize=(figsize))

    sci_x = kwargs.pop('sci_x', None)
    sci = kwargs.pop("sci", None)
    if projection is None:
        # This is the Sci for the x axis
        ax, kwargs = put.prepare_axis(ax, sci=sci_x, **kwargs)

    # plotting
    color = kwargs.pop("color", None)
    alpha = kwargs.pop("alpha", 1.0)
    lw = kwargs.pop("lw", 1)
    size = kwargs.pop("size", 1)
    marker = kwargs.pop("marker", "o")
    fillstyle = kwargs.pop("fillstyle", "full")

    # set colormap
    if cmap is not None:
        cmap = plt.get_cmap(cmap)

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

    vmin = np.nanquantile(z, q=0.1) if vmin is None else vmin
    vmax = np.nanquantile(z, q=0.9) if vmax is None else vmax

    if set_norm == 'log':
        base = kwargs.pop('base', 2)
        # vmin and vmax are now the exponentials
        levels = np.logspace(vmin, vmax, levels + 1, base=base)
    elif levels is not None:
        levels = np.linspace(vmin, vmax, levels + 1, endpoint=True)
    round_dec = kwargs.pop("round_dec", None)
    expo = gut.get_exponent10(vmin) if vmin != 0 else gut.get_exponent10(vmax)
    if round_dec is None:
        round_dec = np.abs(expo)+1
    if round_dec < np.abs(expo):
        raise ValueError(
            f'Vmin {vmin} smaller than decimal to round {round_dec}!')
    if sci is None:
        sci = expo if np.abs(expo) > 1 else None
    if levels is not None:
        levels = np.around(
            levels, round_dec) if round_dec is not None else levels

    if levels is not None and plot_type != 'points' and plot_type != 'contour' and cmap is not None:
        # norm = mpl.colors.LogNorm(levels=levels)
        norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        # vmin = vmax = None
    else:
        norm = None
    ticks = None
    extend = put.set_cb_boundaries(data=z, im=None,
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
    elif plot_type == "colormesh":
        im = ax.pcolor(
            x,
            y,
            z,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
            norm=norm,
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
            )
    elif plot_type == "contourf":
        """
        see https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.contourf.html
        """
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
        )
        if vmin is None:
            vmin = im.get_clim()[0]
        if vmax is None:
            vmax = im.get_clim()[1]
        # im = ScalarMappable(cmap=im.get_cmap())
        # print(vmin, vmax)
    elif plot_type == "contour":
        cmap = cmap if color is None else None
        im = ax.contour(
            x,
            y,
            z,
            levels=levels,
            cmap=cmap,
            transform=projection,
            colors=color,  # color all levels with the same color, see documentation
            linewidths=lw,  # maybe linewidth=
            alpha=alpha,
        )
    elif plot_type == "discrete":
        vmin = np.nanmin(z)
        vmax = np.nanmax(z)
        cmap, norm = put.discrete_cmap(
            vmin, vmax, colormap=cmap, num_ticks=levels, shift_ticks=True
        )
        normticks = put.discrete_norm_ticks(
            vmin, vmax, num_ticks=levels, shift_ticks=True)
        ticks = normticks[:-1] + 0.5

        im = ax.pcolor(
            x,
            y,
            z,
            # vmin=vmin,
            # vmax=vmax,
            cmap=cmap,
            transform=projection,
            norm=norm,
        )
    elif plot_type == "points":
        # print(x)
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
        )

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
                zorder=15,
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
                zorder=15,
            )
        if plot_type == 'hatch':
            im = im_tmp
    # else:
    #     raise ValueError(f"Plot type {plot_type} does not exist!")

    kwargs = put.set_title(title=title, ax=ax, **kwargs)

    if label is not None:
        cbar = put.make_colorbar(ax, im=im,
                                 norm=set_norm,
                                 ticks=levels,
                                 label=label,
                                 set_cax=True,
                                 sci=sci,
                                 **kwargs)
        if plot_type == "discrete":
            cbar.set_ticks(ticks)
            cbar.ax.set_xticklabels(ticks, rotation=45)
            cbar.set_ticklabels(normticks)

    return {"ax": ax, "fig": fig, "projection": projection, "im": im,
            'ticks': levels, 'extend': extend}


def plot_edges(
    ds,
    edges,
    weights=None,
    central_longitude=0,
    fig=None,
    ax=None,
    projection="EqualEarth",
    plot_points=False,
    vmin=None,
    vmax=None,
    **kwargs,
):

    plt_grid = kwargs.pop("plt_grid", False)
    set_map = kwargs.pop("set_map", False)
    ax, fig = create_map(
        da=ds.ds,
        ax=ax,
        projection=projection,
        central_longitude=central_longitude,
        plt_grid=plt_grid,
        set_map=set_map,
        **kwargs
    )

    counter = 0
    lw = kwargs.pop("lw", 1)
    alpha = kwargs.pop("alpha", 1)
    color = kwargs.pop("color", "k")

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

    print(f"number of edges: {counter}")
    return {"ax": ax, "fig": fig, "projection": projection}


def plot_wind_field(
    u,
    v,
    ax=None,
    x_vals=None,
    y_vals=None,
    key_loc=(0.95, -0.06),
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

    lw = kwargs.pop("lw", 1)
    scale = kwargs.pop("scale", None)
    headwidth = kwargs.pop('headwidth', 10)
    width = kwargs.pop('width', 0.007)
    headaxislength = kwargs.pop('headaxislength', 3)
    headlength = kwargs.pop('headlength', 6)
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
                # width=width,
                headaxislength=headaxislength,
                headlength=headlength,
                linewidth=lw,
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
            )
        key_length = kwargs.pop('key_length', 1)
        wind_unit = kwargs.pop('wind_unit', r"$\frac{m}{s}$")
        if key_loc:
            ax.quiverkey(
                Q,
                key_loc[0],
                key_loc[1],
                key_length,
                f"{key_length} {wind_unit}",
                labelpos="W",
                coordinates="axes",
            )
    return {'ax': ax}


def create_multi_plot(nrows, ncols, ds=None, projection=None,
                      plt_grid=True, **kwargs):
    figsize = kwargs.pop('figsize', None)
    if figsize is None:
        figsize = (6*ncols, 4*nrows)

    ratios_w = np.ones(ncols)
    ratios_h = np.ones(nrows)
    gs_rows = nrows
    gs_cols = ncols
    fig = plt.figure(figsize=(figsize[0], figsize[1]))

    hspace = kwargs.pop('hspace', 0.1)
    wspace = kwargs.pop('wspace', 0.1)
    space = kwargs.pop('space', 0)
    if space != 0:
        hspace = wspace = space
    gs = fig.add_gridspec(gs_rows, gs_cols,
                          height_ratios=ratios_h,
                          width_ratios=ratios_w,
                          hspace=hspace, wspace=wspace)
    if projection is not None:
        central_longitude = kwargs.pop("central_longitude", 0)
        proj = get_projection(projection=projection,
                              central_longitude=central_longitude)
    else:
        proj = None
    axs = []

    run_idx = 1
    end_idx = kwargs.pop('end_idx', None)
    end_idx = int(nrows*ncols) if end_idx is None else end_idx

    for i in range(nrows):
        for j in range(ncols):
            axs.append(fig.add_subplot(gs[i, j], projection=proj))

            if ds is not None:
                ax, _ = create_map(
                    da=ds,
                    ax=axs[run_idx-1],
                    projection=projection,
                    central_longitude=central_longitude,
                    plt_grid=plt_grid,
                    **kwargs
                )
            run_idx += 1
            if run_idx > end_idx:
                break
    fig.tight_layout()
    if nrows > 1 or ncols > 1:
        put.enumerate_subplots(axs, pos_x=-0.1, pos_y=1.07)
    else:
        axs = axs[0]

    title = kwargs.pop('title', None)
    y_title = kwargs.pop('y_title', 1.)
    if title is not None:
        put.set_title(title=title, ax=None, fig=fig,
                      y_title=y_title)

    return {"ax": axs, "fig": fig, "projection": projection}


def create_multi_map_plot_gs(nrows, ncols,
                             projection="EqualEarth",
                             orientation='horizontal',
                             **kwargs):
    """OUTDATED!!!
    """
    figsize = kwargs.pop('figsize', None)
    if figsize is None:
        figsize = (6*ncols, 4*nrows)
    if orientation == 'vertical':
        ratios_w = np.append(np.ones(ncols), 0.05)
        ratios_h = np.ones(nrows)
        gs_rows = nrows
        gs_cols = ncols + 1
    elif orientation == 'horizontal':
        ratios_w = np.ones(ncols)
        ratios_h = np.append(np.ones(nrows), 0.05)
        gs_rows = nrows + 1
        gs_cols = ncols
    else:
        ratios_w = np.ones(ncols)
        ratios_h = np.ones(nrows)
        gs_rows = nrows
        gs_cols = ncols
    fig = plt.figure(figsize=(figsize[0], figsize[1]))

    hspace = kwargs.pop('hspace', 0)
    wspace = kwargs.pop('wspace', 0)
    space = kwargs.pop('space', 0)
    if space != 0:
        hspace = wspace = space
    gs = fig.add_gridspec(gs_rows, gs_cols,
                          height_ratios=ratios_h,
                          width_ratios=ratios_w,
                          hspace=hspace, wspace=wspace)
    central_longitude = kwargs.pop("central_longitude", 0)
    proj = get_projection(projection=projection,
                          central_longitude=central_longitude)
    axs = []
    cbar_size = 1
    for i in range(nrows):
        for j in range(ncols):
            axs.append(fig.add_subplot(gs[i, j], projection=proj))

    axs = np.array(axs)
    put.enumerate_subplots(axs, pos_x=-0.1, pos_y=1.07)
    if orientation == 'vertical':
        axs = np.append(axs, fig.add_subplot(gs[:, ncols]))
    elif orientation == 'horizontal':
        axs = np.append(axs, fig.add_subplot(gs[nrows, :]))

    return {"ax": axs.flatten(), "fig": fig, "gs": gs, "projection": projection}


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


def plot_corr_matrix(
    mat_corr,
    pick_x=None,
    pick_y=None,
    label_x=None,
    label_y=None,
    ax=None,
    vmin=-1,
    vmax=1,
    color="BrBG",
    bar_title="correlation",
):
    """Plot correlation matrix.

    Args:
        mat_corr ([type]): [description]
        pick_x ([type], optional): [description]. Defaults to None.
        pick_y ([type], optional): [description]. Defaults to None.
        label_x ([type], optional): [description]. Defaults to None.
        label_y ([type], optional): [description]. Defaults to None.
        ax ([type], optional): [description]. Defaults to None.
        vmin (int, optional): [description]. Defaults to -1.
        vmax (int, optional): [description]. Defaults to 1.
        color (str, optional): [description]. Defaults to 'BrBG'.
        bar_title (str, optional): [description]. Defaults to 'correlation'.

    Returns:
        im (plt.imshow): [description]
    """
    if ax is None:
        fig, ax = plt.subplots()

    if pick_y is not None and pick_x is not None:
        corr = mat_corr[pick_x, :].copy()
        corr = corr[:, pick_y]
    elif pick_x is not None:
        corr = mat_corr[pick_x, :]
    elif pick_y is not None:
        corr = mat_corr[:, pick_y]
    else:
        corr = mat_corr

    cmap = plt.get_cmap(color)
    im = ax.imshow(corr, vmin=vmin, vmax=vmax, aspect="auto", cmap=cmap)

    cbar = plt.colorbar(
        im, extend="both", orientation="horizontal", label=bar_title, shrink=1.0, ax=ax
    )

    if label_x is not None:
        ax.set_xticks(np.arange(0, len(label_x)))
        ax.set_xticklabels(label_x)
    if label_y is not None:
        ax.set_yticks(np.arange(0, len(label_y)))
        ax.set_yticklabels(label_y)

    return im


def plot_rectangle(ax, lon_range, lat_range, **kwargs):
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
