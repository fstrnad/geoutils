"""Axis preparation, titles, legends and line/text annotations."""
from matplotlib.offsetbox import AnchoredText
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import geoutils.plotting.style as pst




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
        box_loc = (-0.06, -0.2) if loc == 'under' else box_loc_default
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
        fsize = kwargs.pop('vertical_title_fsize', pst.MEDIUM_SIZE)
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


def prepare_axis(ax, **kwargs):
    """Prepares an axis for any type x to y plot

    Args:
        ax (ax-object): axis object from matplotlib

    Returns:
        ax: ax matplotlib object
    """
    set_grid = kwargs.get('set_grid', False)
    xlim = kwargs.get("xlim", None)
    ylim = kwargs.get("ylim", None)
    xticks = kwargs.get('xticks', None)
    yticks = kwargs.get('yticks', None)
    xticklabels = kwargs.get('xticklabels', None)
    yticklabels = kwargs.get('yticklabels', None)
    ylabel = kwargs.get("ylabel", None)
    ylabel_color = kwargs.get("ylabel_color", "k")
    xlabel = kwargs.get("xlabel", None)
    x_label_color = kwargs.get("x_label_color", "k")
    xlabel_pos = kwargs.get("xlabel_pos", None)
    ylabel_pos = kwargs.get("ylabel_pos", None)
    set_xint = kwargs.get("set_xint", False)
    set_yint = kwargs.get("set_yint", False)
    set_yaxis = kwargs.get("set_yaxis", True)
    set_xaxis = kwargs.get("set_xaxis", True)
    set_twinx = kwargs.get("set_twinx", False)
    set_twiny = kwargs.get("set_twiny", False)
    set_spines = kwargs.get("set_spines", False)
    spines_color = kwargs.get("spines_color", "black")
    reset_axis = kwargs.get("reset_axis", False)
    set_ticks = kwargs.get("set_ticks", True)
    set_xticks = kwargs.get("set_xticks", True)
    set_yticks = kwargs.get("set_yticks", True)
    top_ticks = kwargs.get("top_ticks", False)
    right_ticks = kwargs.get("right_ticks", False)
    left_ticks = kwargs.get("left_ticks", True)
    # face_color = kwargs.pop("face_color", 'none')
    # ax.set_facecolor(face_color)

    if set_twinx:
        ax.set_zorder(1)
        ax = ax.twinx()
        ax.set_zorder(1)
    if set_twiny:
        ax = ax.twiny()

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

    if not has_labels(ax):
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

    title = kwargs.pop("title", None)
    set_title(title=title, ax=ax, **kwargs)

    ylog = kwargs.pop("ylog", False)
    ysymlog = kwargs.pop("ysymlog", False)
    yscale = kwargs.pop("yscale", None)
    if yscale == 'log':
        ylog = True
    elif yscale == 'symlog':
        ysymlog = True

    if ylog and ysymlog:
        raise ValueError('ylog and ysymlog cannot be True at the same time!')
    if ylog is True:
        ax.set_yscale("log")
    if ysymlog is True:
        ax.set_yscale("symlog")

    xlog = kwargs.pop("xlog", False)
    if xlog is True:
        ax.set_xscale("log")

    log = kwargs.pop("log", False)
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


def check_geoaxis(ax, check_axis=True):
    import cartopy.mpl.geoaxes as cmga

    if check_axis:
        if isinstance(ax, cmga.GeoAxes) or isinstance(ax, cmga.GeoAxesSubplot):
            return True

    return False


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


def remove_colorbars(ax):
    fig = ax.figure

    # Find and remove a colorbar linked to this axis
    for cb_ax in fig.axes[:]:      # copy list because we may modify it
        if isinstance(cb_ax, mpl.colorbar.Colorbar):
            # If this colorbar belongs to our axis, remove it
            if cb_ax.ax == ax:
                cb_ax.remove()
                print('Colorbar removed.')


    # Clear the axis
    ax.cla()
    ax.set_axis_off()
    return ax


def full_reset_ax(ax, fig=None):
    ax.cla()                  # clear all artists (content)
    ax.set_xticks([])         # remove ticks
    ax.set_yticks([])
    ax.set_xticklabels([])    # remove tick labels
    ax.set_yticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    ax = remove_colorbars(ax=ax)

    return ax
