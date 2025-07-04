from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.offsetbox import AnchoredText
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import copy
import numpy as np
import string
import cartopy.crs as ccrs
import cartopy as ctp
import cartopy.mpl as cmpl
import geoutils.plotting.plot_settings as pst
from importlib import reload
reload(pst)


def set_legend(ax,
               fig=None,
               label_arr=[],
               legend_items=None,
               **kwargs):
    if fig is None:
        fig = ax
    loc = kwargs.pop("loc", "upper right")
    box_loc_default = (0.96, 1)
    box_loc = kwargs.pop('box_loc', None)
    if box_loc is None:
        box_loc = (-0.05, -0.15) if loc == 'under' else box_loc_default
    bbox_to_anchor = box_loc if loc == "outside" or loc == 'under' else None
    ncol_legend = kwargs.pop('ncol_legend', 1)
    ncol_legend = 2 if ncol_legend == 1 and loc == 'under' else ncol_legend
    loc = 'upper left' if loc == 'outside' or loc == 'under' else loc  # loc is set above
    fsize = kwargs.pop("fontsize", pst.MEDIUM_SIZE)
    order = kwargs.pop("order", None)
    if len(label_arr) == 0:
        legend_items, label_arr = ax.get_legend_handles_labels()
    if len(label_arr) > 0:
        if order is not None:
            legend_items = [legend_items[idx] for idx in order],
            label_arr = [label_arr[idx] for idx in order]

        leg = fig.legend(
            handles=legend_items,
            labels=label_arr,
            bbox_to_anchor=bbox_to_anchor,
            loc=loc,
            fancybox=True,
            shadow=False,
            ncol=ncol_legend,
            framealpha=0.8,
            frameon=True,
            fontsize=fsize,
        )
        # leg = fig.legend(
        #     # label_arr,   # if commented in might cause problems with sns
        #     bbox_to_anchor=bbox_to_anchor,
        #     loc=loc,
        #     fancybox=True,
        #     shadow=False,
        #     ncol=ncol_legend,
        #     framealpha=0.8,
        #     frameon=True,
        #     fontsize=fsize,
        # )

        leg.get_frame().set_linewidth(0.0)
        leg.set_zorder(pst.MAX_ZORDER)
    return ax


def get_available_mpl_colormaps():
    """
    Get a list of all available colormaps in matplotlib.

    Returns:
        list: A list of strings containing the names of available colormaps.
    """
    colormaps = plt.colormaps()
    return np.array(colormaps)


def get_available_palettable_colormaps():
    """
    Get a list of all available colormaps in palettable.

    Returns:
        list: A list of strings containing the names of available colormaps.
    """
    import palettable as pt
    import cmocean as cmo
    import cmweather
    diverging_lst = [cmap for cmap in dir(
        pt.colorbrewer.diverging) if not cmap.startswith("__")]
    qualitative_lst = [cmap for cmap in dir(
        pt.colorbrewer.qualitative) if not cmap.startswith("__")]
    sequential_lst = [cmap for cmap in dir(
        pt.colorbrewer.sequential) if not cmap.startswith("__")]
    cmocean_div = [cmap for cmap in dir(
        pt.cmocean.diverging) if not cmap.startswith("__")]
    cmocean_seq = [cmap for cmap in dir(
        pt.cmocean.sequential) if not cmap.startswith("__")]
    scientific_seq = [cmap for cmap in dir(
        pt.scientific.sequential) if not cmap.startswith("__")]
    scientific_div = [cmap for cmap in dir(
        pt.scientific.diverging) if not cmap.startswith("__")]

    return (np.array(diverging_lst + qualitative_lst + sequential_lst +
                     cmocean_div + cmocean_seq +
                     scientific_seq + scientific_div),
            diverging_lst,
            qualitative_lst,
            sequential_lst,
            cmocean_div,
            cmocean_seq,
            scientific_seq,
            scientific_div)


def get_cmap(cmap, levels=None, all=pst.enable_all_cmaps):
    mpl_cmaps = get_available_mpl_colormaps()
    if cmap in mpl_cmaps:
        colormap = cmap
    elif not all:
        raise ValueError(
            f'Colormap {cmap} not found. Please choose from {mpl_cmaps}!')
    if all and cmap not in mpl_cmaps:
        import palettable as pt
        pt_cmaps, d_cmaps, q_cmaps, s_cmaps, cmocean_div, cmocean_seq, scientific_seq, scientific_div = get_available_palettable_colormaps()

        if cmap in pt_cmaps:
            cmap_strs = cmap.split("_")
            reverse = True if 'r' in cmap_strs else False
            if cmap in d_cmaps:
                colormap = pt.colorbrewer.get_map(
                    cmap_strs[0], 'diverging',  number=int(cmap_strs[1]),
                    reverse=reverse)
            elif cmap in s_cmaps:
                colormap = pt.colorbrewer.get_map(
                    cmap_strs[0], 'sequential',  number=int(cmap_strs[1]), reverse=reverse)
            elif cmap in q_cmaps:
                colormap = pt.colorbrewer.get_map(
                    cmap_strs[0], 'qualitative',  number=int(cmap_strs[1]), reverse=reverse)
            elif cmap in cmocean_div:
                colormap = pt.cmocean.diverging.get_map(
                    name=cmap, reverse=reverse)
            elif cmap in cmocean_seq:
                colormap = pt.cmocean.sequential.get_map(
                    cmap, reverse=reverse)
            elif cmap in scientific_seq:
                colormap = pt.scientific.sequential.get_map(
                    cmap, reverse=reverse)
            elif cmap in scientific_div:
                colormap = pt.scientific.diverging.get_map(
                    cmap, reverse=reverse)
        else:
            raise ValueError(
                f'Colormap {cmap} not found. Please choose from {mpl_cmaps} or {pt_cmaps}')
        colormap = colormap.mpl_colormap

    n_colors = len(levels) if levels is not None else None
    cmap = plt.get_cmap(colormap, n_colors)

    return cmap


def set_title(title, ax=None, fig=None, **kwargs):
    y_title = kwargs.pop('y_title', 1.05)
    vertical_title = kwargs.pop('vertical_title', None)
    x_title_offset = kwargs.pop('x_title_offset', -0.33)
    title_color = kwargs.pop('title_color', 'black')
    # fw = kwargs.pop('title_fontweight', "normal")
    fw = kwargs.pop('title_fontweight', "bold")
    fsize = kwargs.pop('title_fsize', pst.BIGGER_SIZE)
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
                         fontsize=fsize)
    if vertical_title is not None:
        fsize = kwargs.pop('vertical_title_fsize', pst.BIGGER_SIZE)
        ax.text(x=x_title_offset, y=.5, s=vertical_title,
                transform=ax.transAxes,
                color=title_color, rotation='vertical',
                verticalalignment="center",
                fontweight=fw, size=fsize)
    return kwargs


def has_labels(ax):
    """
    Check if the given Matplotlib axis has x and y labels set.

    Parameters:
    ax (matplotlib.axes.Axes): The axis to check for labels.

    Returns:
    bool: True if either x or y labels are set, False otherwise.
    """
    return bool(ax.get_xlabel()) or bool(ax.get_ylabel())


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
    yticks = kwargs.pop('yticks', None)
    xticklabels = kwargs.pop('xticklabels', None)
    yticklabels = kwargs.pop('yticklabels', None)
    ylabel = kwargs.pop("ylabel", None)
    ylabel_color = kwargs.pop("ylabel_color", "k")
    xlabel = kwargs.pop("xlabel", None)
    x_label_color = kwargs.pop("x_label_color", "k")
    xlabel_pos = kwargs.pop("xlabel_pos", None)
    ylabel_pos = kwargs.pop("ylabel_pos", None)
    set_xint = kwargs.pop("set_xint", False)
    set_yint = kwargs.pop("set_yint", False)
    set_yaxis = kwargs.pop("set_yaxis", True)
    set_xaxis = kwargs.pop("set_xaxis", True)
    set_twinx = kwargs.pop("set_twinx", False)
    set_twiny = kwargs.pop("set_twiny", False)
    set_spines = kwargs.pop("set_spines", False)
    spines_color = kwargs.pop("spines_color", "black")
    reset_axis = kwargs.pop("reset_axis", False)
    face_color = kwargs.pop("face_color", 'none')
    set_ticks = kwargs.pop("set_ticks", True)
    set_xticks = kwargs.pop("set_xticks", True)
    set_yticks = kwargs.pop("set_yticks", True)
    top_ticks = kwargs.pop("top_ticks", False)
    right_ticks = kwargs.pop("right_ticks", False)
    left_ticks = kwargs.pop("left_ticks", True)

    ax.set_facecolor(face_color)
    if set_twinx:
        ax.set_zorder(1)
        ax = ax.twinx()
        ax.set_zorder(1)
    if set_twiny:
        ax = ax.twiny()
    if plot_type != 'polar':
        ax.spines['bottom'].set_visible(set_xaxis)
        ax.spines['left'].set_visible(set_yaxis)
        if not set_yaxis:
            ylabel = None
            set_yticks = False
        if not set_xaxis:
            xlabel = None
            set_xticks = False

        if not set_twinx:
            ax.spines["right"].set_visible(False)
        else:
            ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(set_spines)
        if set_spines:
            ax.spines['bottom'].set_color(spines_color)
            ax.spines['top'].set_color(spines_color)
            ax.spines['left'].set_color(spines_color)
            ax.spines['right'].set_color(spines_color)

        ax.grid(set_grid)
        ax.tick_params(
            direction="out",
            length=pst.SMALL_SIZE / 2, width=1,
            colors=ylabel_color, grid_alpha=0.5
        )

        if not has_labels(ax) or reset_axis:
            ax.set_xlabel(xlabel, color=x_label_color)
            ax.set_ylabel(ylabel, color=ylabel_color)

            if xlabel_pos is not None:
                if xlabel_pos == "right":
                    ax.xaxis.set_label_coords(1.0, -0.2)
                else:
                    ax.xaxis.set_label_coords(xlabel_pos)
            if ylabel_pos is not None:
                ax.yaxis.set_label_coords(x=ylabel_pos[0],
                                          y=ylabel_pos[1])

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        rot = kwargs.pop("rot", 0)
        rot_y = kwargs.pop("rot_y", 0)

        if set_ticks is False:
            left_ticks = False
            right_ticks = False
        if set_twinx:
            left_ticks = False
            right_ticks = True
        ax.tick_params(axis="x", labelrotation=rot,
                       bottom=set_xticks,
                       top=top_ticks,
                       )
        ax.tick_params(axis="y", labelrotation=rot_y,
                       left=left_ticks if set_yticks else False,
                       right=right_ticks,
                       )
        if xticks is not None and set_xticks:
            ax.set_xticks(xticks)

        if xticklabels is not None and set_xticks:
            ax.set_xticklabels(xticklabels)
        if set_xint:
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        if set_yint:
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

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
    yscale = kwargs.pop("yscale", None)
    if yscale == 'log':
        ylog = True
    elif yscale == 'symlog':
        ysymlog = True

    ysymlog = kwargs.pop("ysymlog", False)
    if ylog and ysymlog:
        raise ValueError('ylog and ysymlog cannot be True at the same time!')
    if ylog is True:
        ax.set_yscale("log")
    if ysymlog is True:
        ax.set_yscale("symlog")

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

    if not set_ticks or not set_xticks or not set_yticks:
        if not set_xticks:
            ax.xaxis.set_ticklabels([])
        if not set_yticks:
            ax.yaxis.set_ticklabels([])

    yticks = kwargs.pop('yticks', None)
    if yticks is not None:
        ax.set_yticks(yticks)
        yticklabels = kwargs.pop('yticklabels', None)
        yticklabels = yticks if yticklabels is None else yticklabels
        ax.set_yticklabels(yticklabels)

    unset_yticks = kwargs.pop('unset_yticks', False)
    if unset_yticks:
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


def truncate_colormap(cmap='terrain_r', minval=0.0, maxval=1.0, n=100):
    cmap = plt.get_cmap(cmap)
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def create_cmap(cmap, levels=None, **kwargs):
    # set colormap
    if isinstance(cmap, str):
        cmap = get_cmap(cmap, levels=levels)
    elif not isinstance(cmap, mpl.colors.Colormap) or not isinstance(cmap, mpl.colors.LinearSegmentedColormap):
        raise ValueError(
            f'cmap has to be of type str or mpl.colors.Colormap but is of type {type(cmap)}!')
    if levels is not None:
        n_colors = len(levels)
        # Set center of colormap to specific color
        centercolor = kwargs.pop('centercolor', None)
        leftcolor = kwargs.pop('leftcolor', None)
        if centercolor is not None:
            colors = np.array([mpl.colors.rgb2hex(cmap(i))
                               for i in range(n_colors)])

            centercolor = '#FFFFFF' if centercolor == 'white' else centercolor
            idx = [len(colors) // 2 - 1, len(colors) // 2]
            colors[idx] = centercolor
            cmap = mpl.colors.ListedColormap(colors)
        if leftcolor is not None:
            colors = np.array([mpl.colors.rgb2hex(cmap(i))
                               for i in range(n_colors)])
            leftcolor = '#FFFFFF' if leftcolor == 'white' else leftcolor
            colors[0] = leftcolor
            cmap = mpl.colors.ListedColormap(colors)
        norm = mpl.colors.BoundaryNorm(
            levels, ncolors=cmap.N, clip=True)
    else:
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


def check_geoaxis(ax, check_axis=True):
    import cartopy.mpl.geoaxes as cmga

    if check_axis:
        if isinstance(ax, cmga.GeoAxes) or isinstance(ax, cmga.GeoAxesSubplot):
            return True

    return False


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

    return extend, kwargs


def add_colorbar(im, fig, label,
                 x_pos=0, y_pos=0,
                 width=None, height=None,
                 set_cax=False,
                 multi_plots=False,
                 **kwargs):
    if multi_plots:
        # find out number of rows and columns
        grid_spec = fig.axes[0].get_subplotspec().get_gridspec()
        num_rows, num_cols = grid_spec.get_geometry()
        num_cols = len(fig.axes) // num_rows
        bbox0 = fig.axes[0].get_position(fig)
        bbox1 = fig.axes[num_cols-1].get_position(fig)  # last column
        bboxlast = fig.axes[-1].get_position(fig)
        if width is None:
            width = (bbox1.x1 - bbox0.x0)/1.1
            delta_width = (bbox1.x1 - bbox0.x0) - width
            height = bboxlast.height / 10  # 10% of height
        x_pos = bbox0.x0 + delta_width/2
        y_pos = bboxlast.y0 - height*1.5

        positions = [x_pos, y_pos, width, height]
    else:
        width = 0.8 if width is None else width
        height = 0.05 if height is None else height
        positions = [x_pos, y_pos, width, height]
    ax_cbar = fig.add_axes(positions)
    orientation = kwargs.pop('orientation', 'horizontal')
    fsize = kwargs.pop('fontsize', pst.BIGGER_SIZE)
    cbar = make_colorbar(ax=ax_cbar,
                         im=im['im'],
                         fig=fig,
                         orientation=orientation,
                         label=label,
                         set_cax=set_cax,
                         ticks=im['ticks'],
                         fontsize=fsize,
                         **kwargs,
                         )
    return cbar


def make_colorbar(ax, im, fig=None, **kwargs):
    ticks = kwargs.pop('ticks', None)
    extend = kwargs.pop('extend', 'neither')
    sci = kwargs.pop("sci", None)
    shift_ticks = kwargs.pop("shift_ticks", None)
    fsize = kwargs.pop('fontsize', None)
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
    set_int = kwargs.pop("set_int", False)
    if round_dec is not None:
        ticks = np.around(ticks, round_dec)
    if set_int:
        ticks = np.round(ticks).astype(int)
    if fig is None:
        fig = ax.get_figure()

    orientation = kwargs.pop("orientation", "horizontal")
    set_cax = kwargs.pop('set_cax', False)
    if set_cax:
        if orientation == "vertical":
            loc = "right"
            pad = kwargs.pop('pad', -2.5)  # distance from right border
            # divider = make_axes_locatable(ax)
            # cax = divider.append_axes(loc, "5%", pad=pad, axes_class=plt.Axes)
            cax = inset_axes(ax,
                             width="5%",
                             height="100%",
                             loc=loc,
                             borderpad=pad)

        elif orientation == "horizontal":
            loc = "lower center"
            # defines distance to lower y-range (different for maps and xy-plots) is passed to make_colorbar
            pad = kwargs.pop('pad', -4.5)
            cax = inset_axes(ax,
                             width="100%",
                             height="5%",
                             loc=loc,
                             borderpad=pad)
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
            extend=extend,
            **kwargs,
        )
    else:
        norm = kwargs.pop('norm', None)
        if norm == 'log':
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
            extend=extend,
            **kwargs
        )

    if shift_ticks is not None:
        set_rot = kwargs.pop('set_rot', False)
        cbar.set_ticks(ticks[:-1]+0.5)
        if set_rot:
            cbar.ax.set_xticklabels(ticks[:-1], rotation=45)
        cbar.set_ticklabels(ticks[:-1])

    if fsize is not None:
        cbar.ax.tick_params(labelsize=fsize)
        cbar.set_label(label=label, size=fsize)

    return cbar


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
                                    linestyle=ls,
                                    ),
                    fontsize=kwargs.pop('fontsize', pst.BIGGER_SIZE),
                    weight=kwargs.pop('weight', 'bold'),
                    color=color,
                    zorder=zorder,)
    else:
        ax.arrow(x1, y1, x2-x1, y2-y1,
                 #  xycoords='data',
                 zorder=zorder,
                 **arrowprops,
                 transform=ccrs.PlateCarree()._as_mpl_transform(
                     ax) if trafo_axis else ax.transData,
                 )


def plot_hline(ax, y, transform=None, **kwargs):
    lw = kwargs.pop("lw", 1)
    ls = kwargs.pop("ls", '--')
    color = kwargs.pop("color", "k")
    zorder = kwargs.pop('zorder', 10)
    label = kwargs.pop('label', None)
    if transform is None:
        ax.axhline(y=y, color=color,
                   zorder=zorder,
                   lw=lw,
                   linestyle=ls,
                   label=label)
    else:
        ax.axhline(y=y, color=color,
                   zorder=zorder,
                   lw=lw,
                   linestyle=ls,
                   transform=transform,
                   label=label)

    if label is not None:
        # labels and legend items are already within ax-object
        ax = set_legend(ax=ax, **kwargs)

    return {'im': ax}


def enumerate_subplots(axs, pos_x=-0.12,
                       pos_y=1.06,
                       fontsize=pst.BIGGER_SIZE):
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
    if isinstance(pos_x, float) or isinstance(pos_x, int):
        pos_x = [pos_x] * len(axs.flatten())
    if isinstance(pos_y, float) or isinstance(pos_y, int):
        pos_y = [pos_y] * len(axs.flatten())

    for n, ax in enumerate(axs.flatten()):
        li = n % 26  # letter index
        fac = n // 26  # factor
        lw_case = f"{string.ascii_lowercase[li]}."
        up_case = f"{string.ascii_uppercase[li]}{fac}."
        plot_text(
            ax=ax,
            xpos=pos_x[n],
            ypos=pos_y[n],
            text=lw_case if n < 26 else up_case,
            size=fontsize,
            weight="bold",
            transform=True,
            box=False,
            check_axis=False,
        )

    return axs


def plot_text(ax, text, xpos=0, ypos=0,
              **kwargs):

    color = kwargs.pop("color", "k")
    zorder = kwargs.pop('zorder', 10)
    rot = kwargs.pop("rot", 0)
    fsize = kwargs.pop('fsize', pst.BIGGER_SIZE)
    weight = kwargs.pop('weight', "normal")
    trafo_axis = kwargs.pop('transform', False)
    check_axis = kwargs.pop('check_axis', True)
    box = kwargs.pop('box', False)
    plot_box = dict(facecolor='white', edgecolor='black',
                    pad=10.0) if box else None
    if check_geoaxis(ax, check_axis):
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


def check_projection(ax, projection):
    if ax is not None and projection is not None:
        raise ValueError(
            f'Axis already given, projection {projection} will have no effect.')
    else:
        # Set default to PlateCarree
        projection = 'PlateCarree' if projection is None else projection
    return projection


def get_lon_lat_ticklabels(array_lon=None, array_lat=None, deg_label='°E'):
    lon = np.arange(0, 360, 30) if array_lon is None else array_lon
    lat = np.arange(-90, 90, 30) if array_lat is None else array_lat
    lon_labels = [f'{lon}°E' if lon > 0 and lon <
                  180 else f'{lon}°W' for lon in lon]
    lat_labels = [f'{lat}°N' if lat >= 0 else f'{lat}°S' for lat in lat]
    return lon_labels, lat_labels


def get_grid_figure(fig=None, ax_ratios=None, **kwargs):
    import matplotlib.gridspec as gridspec
    figsize = kwargs.pop('figsize', (10, 6))
    nrows = kwargs.pop('nrows', 2)
    ncols = kwargs.pop('ncols', 1)
    hspace = kwargs.pop('hspace', 0.4)
    wspace = kwargs.pop('wspace', 0.4)

    # Create a gridspec instance
    if fig is None:
        fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows, ncols, hspace=hspace,
                           wspace=wspace,
                           figure=fig)
    ax_arr = []
    for ax_ratio in ax_ratios:
        ax_arr.append(plt.subplot(gs[ax_ratio[0], ax_ratio[1]]))

    return {'fig': fig, 'ax': ax_arr}
