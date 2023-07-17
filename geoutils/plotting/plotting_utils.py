from matplotlib.offsetbox import AnchoredText
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import copy
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import string
import cartopy.crs as ccrs
import geoutils.plotting.plot_settings as pst
from importlib import reload


def set_title(title, ax=None, fig=None, **kwargs):
    y_title = kwargs.pop('y_title', 1.)
    vertical_title = kwargs.pop('vertical_title', None)
    title_color = kwargs.pop('title_color', 'black')
    # fw = kwargs.pop('title_fontweight', "normal")
    fw = kwargs.pop('title_fontweight', "bold")
    fsize = kwargs.pop('title_fsize', pst.MEDIUM_SIZE)
    if title is not None:
        if ax is not None:
            ax.set_title(title, color=title_color,
                         fontweight=fw, size=fsize,
                         y=y_title)
        elif fig is not None:
            fig.suptitle(title,
                         color=title_color,
                         y=y_title,
                         fontweight=fw,
                         fontsize=pst.BIGGER_SIZE)
    if vertical_title is not None:
        x_title_offset = kwargs.pop('x_title_offset', -0.22)
        ax.text(x=x_title_offset, y=.5, s=vertical_title,
                transform=ax.transAxes,
                color=title_color, rotation='vertical',
                verticalalignment="center",
                fontweight=fw, size=fsize)
    return kwargs


def prepare_axis(ax, log=False, **kwargs):
    """Prepares an axis for any type x to y plot

    Args:
        ax (ax-object): axis object from matplotlib

    Returns:
        ax: ax matplotlib object
    """
    reload(pst)
    set_grid = kwargs.pop('set_grid', False)
    plot_type = kwargs.pop('plot_type', 'xy')
    xlim = kwargs.pop("xlim", None)
    ylim = kwargs.pop("ylim", None)
    xticks = kwargs.pop('xticks', None)
    xticklabels = kwargs.pop('xticklabels', None)
    ylabel = kwargs.pop("ylabel", None)
    xlabel = kwargs.pop("xlabel", None)
    xpos = kwargs.pop("xlabel_pos", None)

    if plot_type != 'polar':
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        ax.grid(set_grid)
        ax.tick_params(
            direction="out", length=pst.SMALL_SIZE / 2, width=1, colors="k", grid_alpha=0.5
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if xpos is not None:
            if xpos == "right":
                ax.xaxis.set_label_coords(1.0, -0.2)
            else:
                raise ValueError(f"{xpos} does not exist.")

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        rot = kwargs.pop("rot", 0)
        ax.tick_params(axis="x", labelrotation=rot)
        if xticks is not None:
            ax.set_xticks(xticks)
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels)

    else:
        ax.margins(y=0)
        # x_pos = np.deg2rad(np.arange(0, 360, 360/len(xticks)))
        x_pos = np.deg2rad(np.linspace(0, 360, len(xticks)))
        ax.set_xticks(x_pos)
        ax.set_xticklabels(xticklabels)
        ax.set_rlabel_position(60)  # get radial labels away from plotted line
        # ax.set_rticks([0., 0.1, 0.2, .3, 0.4])  # Less radial ticks
        ax.set_rlim(ylim)
        # rotate the axis arbitrarily, just replace pi with the angle you want.
        # ax.set_theta_offset(np.pi)
    title = kwargs.pop("title", None)
    set_title(title=title, ax=ax, **kwargs)

    ylog = kwargs.pop("ylog", False)
    if ylog is True:
        ax.set_yscale("log")

    xlog = kwargs.pop("xlog", False)
    if xlog is True:
        ax.set_xscale("log")

    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")

        x_major = mpl.ticker.LogLocator(base=10.0, numticks=10)
        ax.xaxis.set_major_locator(x_major)
        x_minor = mpl.ticker.LogLocator(
            base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10
        )
        ax.xaxis.set_minor_locator(x_minor)
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    unset_ticks = kwargs.pop("unset_ticks", False)
    if unset_ticks:
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

    yticks = kwargs.pop('yticks', True)
    if not yticks:
        ax.yaxis.set_ticklabels([])

    sci = kwargs.pop("sci", None)
    if sci is not None:
        ax.ticklabel_format(style="sci", axis="x", scilimits=(sci, sci))

    flip_x = kwargs.pop('flip_x', False)
    invert_x = kwargs.pop('invert_x', False)

    if flip_x or invert_x:
        ax.invert_xaxis()

    flip_y = kwargs.pop('flip_y', False)
    invert_y = kwargs.pop('invert_y', False)
    if flip_y or invert_y:
        ax.invert_yaxis()

    return ax, kwargs


def create_cmap(cmap, levels, **kwargs):
    n_colors = len(levels)
                # set colormap
    if cmap is not None:
        cmap = plt.get_cmap(cmap, n_colors)
    colors = np.array([mpl.colors.rgb2hex(cmap(i)) for i in range(n_colors)])
                # Set center of colormap to specific color
    centercolor = kwargs.pop('centercolor', '#FFFFFF')
    if centercolor is not None:
        idx = [len(colors) // 2 - 1, len(colors) // 2]
        colors[idx] = centercolor
    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.BoundaryNorm(
                    levels, ncolors=cmap.N, clip=True)
    if levels is None:
        cmap = plt.get_cmap(cmap)
        norm = None

    return cmap, norm



def make_colorbar_discrete(ax, im, fig=None, vmin=None, vmax=None, **kwargs):
    if vmin is None:
        vmin = im.get_clim()[0]
    if vmax is None:
        vmax = im.get_clim()[1]
    m = ScalarMappable(cmap=im.get_cmap())
    m.set_array(im.get_array())
    m.set_clim(im.get_clim())
    step = im.levels[1] - im.levels[0]
    num_levels = len(im.levels)

    noextend = "extend" in kwargs.keys() and kwargs["extend"] == "neither"
    # set the colorbar boundaries
    cliplower = im.zmin < vmin if not noextend else False
    clipupper = im.zmax > vmax if not noextend else False
    if "extend" in kwargs.keys() and kwargs["extend"] == "both":
        cliplower = True
        clipupper = True

    ticks = kwargs.pop('ticks', None)
    if ticks is None:
        boundaries = np.linspace(
            vmin - (cliplower and noextend) * step,
            vmax + (clipupper and noextend) * (step + 1),
            num_levels,
            endpoint=True,
        )
    else:
        boundaries = ticks
    boundaries_map = copy.deepcopy(boundaries)
    if cliplower:
        boundaries_map = np.insert(
            boundaries_map, 0, vmin - abs(vmin / 100), axis=0)
    if clipupper:
        boundaries_map = np.insert(
            boundaries_map, len(boundaries_map), vmax + abs(vmax / 100), axis=0
        )

    kwargs["boundaries"] = boundaries_map

    # if the z-values are outside the colorbar range, add extend marker(s)
    # This behavior can be disabled by providing extend='neither' to the function call
    if not ("extend" in kwargs.keys()) or kwargs["extend"] in ["min", "max"]:
        extend_min = cliplower or (
            "extend" in kwargs.keys() and kwargs["extend"] == "min"
        )
        extend_max = clipupper or (
            "extend" in kwargs.keys() and kwargs["extend"] == "max"
        )
        if extend_min and extend_max:
            kwargs["extend"] = "both"
        elif extend_min:
            kwargs["extend"] = "min"
        elif extend_max:
            kwargs["extend"] = "max"

    cbar = make_colorbar(ax=ax, im=m, fig=fig, vmin=vmin, vmax=vmax,

                         **kwargs)

    return cbar


def cbarfmt(x, pos):
    a, b = f'{x:.2e}'.split('e')
    b = int(b)
    return rf'${a} \times 10^{{{b}}}$'


def set_cb_boundaries(data, im=None, vmin=None, vmax=None, **kwargs):
    noextend = "extend" in kwargs.keys() and kwargs["extend"] == "neither"
    extend = kwargs.pop('extend', 'neither')
    if extend == 'neither':
        if data is not None:
            if vmin is None:
                vmin = im.norm.vmin
            if vmax is None:
                vmax = im.norm.vmax
            data_min = np.nanmin(data)
            cliplower = data_min < vmin if not noextend else False
            data_max = np.nanmax(data)
            clipupper = data_max > vmax if not noextend else False

            if not ("extend" in kwargs.keys()) or kwargs["extend"] in ["min", "max"]:
                extend_min = cliplower or (
                    "extend" in kwargs.keys() and kwargs["extend"] == "min"
                )
                extend_max = clipupper or (
                    "extend" in kwargs.keys() and kwargs["extend"] == "max"
                )
                if extend_min and extend_max:
                    extend = "both"
                elif extend_min:
                    extend = "min"
                elif extend_max:
                    extend = "max"

    kwargs["extend"] = extend

    return extend


def add_colorbar(im, fig, label,
                 x_pos=0, y_pos=0, width=0.8, height=0.05,
                 **kwargs):
    extend = kwargs.pop('extend', None)
    if extend is None:
        extend = im['extend']

    ax_cbar = fig.add_axes([x_pos, y_pos, width, height])
    orientation = kwargs.pop('orientation', 'horizontal')
    cbar = make_colorbar(ax=ax_cbar,
                         im=im['im'],
                         fig=fig,
                         orientation=orientation,
                         label=label,
                         set_cax=False,
                         ticks=im['ticks'],
                         extend=extend,  # gives a warning for contourf...
                         **kwargs,
                         )
    return cbar


def make_colorbar(ax, im, fig=None, **kwargs):

    ticks = kwargs.pop('ticks', None)
    sci = kwargs.pop("sci", None)

    tick_step = int(kwargs.pop("tick_step", 2))
    if ticks is not None:
        ticks = ticks[::tick_step]
    else:
        ticks = None
    # if sci is not None:
    #     if sci < 0:
    #         round_dec = abs(sci) + 1
    #     else:
    #         round_dec = 0
    round_dec = kwargs.pop("round_dec", None)
    if round_dec is not None:
        ticks = np.around(ticks, round_dec)
    if fig is None:
        fig = ax.get_figure()

    orientation = kwargs.pop("orientation", "horizontal")
    set_cax = kwargs.pop('set_cax', False)
    pad = kwargs.pop('pad', "35%")  # defines distance to lower y-range
    if set_cax:
        if orientation == "vertical":
            loc = "right"
            pad = "3%"
        elif orientation == "horizontal":
            loc = "bottom"
            pad = pad
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(loc, "5%", pad=pad, axes_class=plt.Axes)
    else:
        cax = ax

    label = kwargs.pop('label', None)

    if sci is not None:
        fmt = mpl.ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((sci, sci))
        cbar = fig.colorbar(
            im,
            cax=cax,
            orientation=orientation,
            label=label,
            format=fmt,
            ticks=ticks,
            **kwargs,
        )
    else:
        norm = kwargs.pop('norm', None)
        if norm == 'log':
            print(norm)
            fmt = mpl.ticker.FuncFormatter(cbarfmt)
        else:
            fmt = None

        cbar = fig.colorbar(
            im,
            cax=cax,
            orientation=orientation,
            label=label,
            ticks=ticks,
            format=fmt,
            **kwargs
        )

    return cbar


def discrete_cmap(vmin, vmax, colormap=None, num_ticks=None, shift_ticks=False):
    import matplotlib as mpl

    # colormap=pt.Spectral_11.mpl_colormap
    if colormap is None:
        # import palettable.colorbrewer.diverging as pt
        import palettable.colorbrewer.qualitative as pt

        colormap = pt.Paired_12.mpl_colormap
    cmap = plt.get_cmap(colormap)

    normticks = discrete_norm_ticks(
        vmin, vmax, num_ticks=num_ticks, shift_ticks=shift_ticks
    )

    norm = mpl.colors.BoundaryNorm(normticks, cmap.N)
    return cmap, norm


def discrete_norm_ticks(vmin, vmax, shift_ticks=False, num_ticks=None):
    if vmin is None or vmax is None:
        return None
    if num_ticks is None:
        num_ticks = 10

    if shift_ticks is True:
        # +1.1 to account for start and end
        normticks = np.arange(vmin, vmax + 1.1, dtype=int)
    else:
        normticks = np.linspace(vmin, vmax, num_ticks + 1)

    return normticks


def get_arr_colorbar(data, cmap='spectral'):
    vmin = np.nanquantile(data, q=0.05)
    vmax = np.nanquantile(data, q=0.95)
    cmap = plt.get_cmap(cmap)
    # Normalize the values in the array to fall within the range [0, 1]
    norm = (array - vmin) / (vmax - vmin)

    # Map the normalized values to colors in the colormap
    colors = cmap(norm)

    return colors


def plot_line(ax, x_range, y_range, **kwargs):
    lw = kwargs.pop("lw", 1)
    color = kwargs.pop("color", "k")
    zorder = kwargs.pop('zorder', 1)

    ax.plot(x_range, y_range,
            linewidth=lw,
            color=color,
            zorder=zorder
            )

    return ax


def plot_vline(ax, x, **kwargs):
    lw = kwargs.pop("lw", 1)
    ls = kwargs.pop("ls", '--')
    color = kwargs.pop("color", "k")
    zorder = kwargs.pop('zorder', 10)
    label = kwargs.pop('label', None)
    trafo_axis = kwargs.pop('trafo_axis', False)
    if trafo_axis:
        ax.plot([x, x], [-90, 90],
                color=color,
                linestyle=ls,
                linewidth=lw,
                transform=ccrs.PlateCarree(),
                label=label,)
    else:
        ax.axvline(x=x, color=color,
                   zorder=zorder,
                   lw=lw,
                   linestyle=ls,
                   label=label,
                   )

    return {'ax': ax, 'x': x}


def plot_arrow(ax, x1, y1, x2, y2, **kwargs):
    lw = kwargs.pop("lw", 1)
    width = kwargs.pop("width", 0.02)
    ls = kwargs.pop("ls", '-')
    color = kwargs.pop("color", "k")
    zorder = kwargs.pop('zorder', 10)
    label = kwargs.pop('label', "")
    fill_color = kwargs.pop('fill_color', True)
    trafo_axis = kwargs.pop('trafo_axis', False)
    arrowprops = dict(color=color,
                      lw=lw,
                      ls=ls,
                      label=label,
                      width=width,
                      ec=color,
                      fc=color if fill_color else 'none'
                      )
    if label is not None:
        ax.annotate(label,
                    xytext=(x1, y1),
                    xy=(x2, y2),
                    xycoords='data',
                    arrowprops=dict(arrowstyle='->',
                                    color=color,
                                    lw=lw,
                                    linestyle=ls),
                    zorder=zorder,)
    else:
        ax.arrow(x1, y1, x2-x1, y2-y1,
                 #  xycoords='data',
                 zorder=zorder,
                 **arrowprops,
                 transform=ccrs.PlateCarree()._as_mpl_transform(
                     ax) if trafo_axis else ax.transData,
                 )


def plot_hline(ax, y, **kwargs):
    lw = kwargs.pop("lw", 1)
    ls = kwargs.pop("ls", '--')
    color = kwargs.pop("color", "k")
    zorder = kwargs.pop('zorder', 10)
    label = kwargs.pop('label', None)
    ax.axhline(y=y, color=color,
               zorder=zorder,
               lw=lw,
               linestyle=ls,
               label=label)

    return {'im': ax}


def enumerate_subplots(axs, pos_x=-0.12, pos_y=1.06, fontsize=20):
    """Adds letters to subplots of a figure.

    Args:
        axs (list): List of plt.axes.
        pos_x (float, optional): x position of label. Defaults to 0.02.
        pos_y (float, optional): y position of label. Defaults to 0.85.
        fontsize (int, optional): Defaults to 18.

    Returns:
        axs (list): List of plt.axes.
    """
    axs = np.array(axs)
    if type(pos_x) == float:
        pos_x = [pos_x] * len(axs.flatten())
    if type(pos_y) == float:
        pos_y = [pos_y] * len(axs.flatten())

    for n, ax in enumerate(axs.flatten()):
        plt_text(
            ax=ax,
            xpos=pos_x[n],
            ypos=pos_y[n],
            text=f"{string.ascii_lowercase[n]}." if n < 26 else f"{string.ascii_lowercase[n-26]}{string.ascii_lowercase[n-26]}.",
            size=fontsize,
            weight="bold",
            transform=True,
            box=False
        )
        # ax.text(
        #     pos_x[n],
        #     pos_y[n],
        #     f"{string.ascii_lowercase[n]}.",
        #     transform=ax.transAxes,
        #     size=fontsize,
        #     weight="bold",
        # )

    return axs


def plt_text(ax, text, xpos=0, ypos=0,
             geoaxis=False, **kwargs):

    color = kwargs.pop("color", "k")
    zorder = kwargs.pop('zorder', 10)
    rot = kwargs.pop("rot", 0)
    fsize = kwargs.pop('fsize', pst.MEDIUM_SIZE)
    weight = kwargs.pop('weight', "bold")
    trafo_axis = kwargs.pop('transform', False)
    box = kwargs.pop('box', False)
    plot_box = dict(facecolor='white', edgecolor='black',
                    pad=10.0) if box else None
    if geoaxis:
        lon_pos = xpos
        lat_pos = ypos
        ax.text(
            lon_pos,
            lat_pos,
            text,
            horizontalalignment="center",
            color=color,
            zorder=zorder,
            rotation=rot,
            fontsize=fsize,
            weight=weight,
            transform=ccrs.Geodetic(),
            bbox=plot_box
        )
    else:
        # Add the text s to the Axes at location x, y in data coordinates.
        ax.text(
            x=xpos,
            y=ypos,
            s=text,
            color=color,
            zorder=zorder,
            rotation=rot,
            fontsize=fsize,
            weight=weight,
            transform=ax.transAxes if trafo_axis else ax.transData,
            bbox=plot_box
        )
    return ax


def text_box(ax, text, pos="upper right", fsize=pst.MEDIUM_SIZE, **kwargs):
    fw = kwargs.pop('fw', "normal")
    fsize = kwargs.pop('fsize', pst.MEDIUM_SIZE)
    anchored_text = AnchoredText(text, loc=pos,
                                 prop=dict(size=fsize,
                                           fontweight=fw))
    ax.add_artist(anchored_text)

    return ax


def check_plot_type(plot_type):
    avail_types = ['contour', 'contourf', 'scatter',
                   'points', 'colormesh', 'discrete']
    if plot_type not in avail_types:
        raise ValueError(f'ERROR plot_type {plot_type} not available!')
    return True
