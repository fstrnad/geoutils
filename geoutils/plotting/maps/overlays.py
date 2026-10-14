"""Annotations and overlays drawn on existing map axes."""
# import matplotlib.cm as cm
import geoutils.utils.spatial_utils as sput
import matplotlib.patches as mpatches
import geoutils.utils.general_utils as gut
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import geoutils.plotting.axes as pax
import geoutils.plotting.colors as pcol
import geoutils.plotting.style as pst
from geoutils.plotting.maps.basemap import create_map




@pst.styled
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


@pst.styled
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
        figsize = kwargs.pop("figsize", None) or pst.default_figsize()
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")

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


@pst.styled
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
        kwargs = pax.set_title(title=title, ax=ax,
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
        cbar = pcol.make_colorbar(ax,
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


@pst.styled
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
        pax.plot_text(ax=ax,
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
