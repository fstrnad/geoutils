"""Axis preparation, titles, legends and line/text annotations."""
import warnings
from dataclasses import dataclass, fields
from matplotlib.offsetbox import AnchoredText
import matplotlib as mpl
import numpy as np
import cartopy.crs as ccrs
import geoutils.plotting.style as pst

# keyword options understood by set_title (used to flag typos)
TITLE_OPTIONS = frozenset({
    'title_color', 'title_fontweight', 'title_fsize', 'title_loc', 'vertical_title',
    'vertical_title_fsize', 'x_title_offset', 'y_title',
})




@pst.styled
def set_legend(ax,
               fig=None,
               label_arr=[],
               legend_items=None,
               **kwargs):
    if fig is None:
        fig = ax
    loc = kwargs.pop("loc", "best")
    box_loc_default = (0.96, 1)
    box_loc = kwargs.pop('box_loc', None)
    if box_loc is None:
        box_loc = (-0.06, -0.2) if loc == 'under' else box_loc_default
    bbox_to_anchor = box_loc if loc == "outside" or loc == 'under' else None
    ncol_legend = kwargs.pop('ncol_legend', 1)
    ncol_legend = 2 if ncol_legend == 1 and loc == 'under' else ncol_legend
    loc = 'upper left' if loc == 'outside' or loc == 'under' else loc  # loc is set above
    fsize = kwargs.pop("fontsize", pst.get_preset().tick)
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
    y_title = kwargs.pop('y_title', None)  # None: let the layout place it
    vertical_title = kwargs.pop('vertical_title', None)
    x_title_offset = kwargs.pop('x_title_offset', -0.33)
    title_color = kwargs.pop('title_color', pst.INK)
    # fw = kwargs.pop('title_fontweight', "normal")
    fw = kwargs.pop('title_fontweight', "bold")
    fsize = kwargs.pop('title_fsize', None)
    title_loc = kwargs.pop('title_loc', 'left')
    if title is not None:
        if ax is not None:
            ax.set_title(title, loc=title_loc, color=title_color,
                         fontweight=fw, size=fsize or pst.get_preset().title,
                         y=y_title)
        elif fig is not None:
            left = title_loc == 'left'
            fig.suptitle(title,
                         x=0.01 if left else 0.5,
                         ha='left' if left else 'center',
                         color=title_color,
                         y=y_title,
                         fontweight=fw,
                         fontsize=fsize or pst.get_preset().figure_title)
    if vertical_title is not None:
        fsize = kwargs.pop('vertical_title_fsize', pst.get_preset().label)
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


@dataclass(frozen=True)
class AxisSpec:
    """Appearance of an x-y axis as set by ``prepare_axis``."""
    title: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    x_label_color: str = pst.INK
    ylabel_color: str = pst.INK
    xlabel_pos: object = None
    ylabel_pos: object = None
    xlim: tuple | None = None
    ylim: tuple | None = None
    xticks: object = None
    yticks: object = None
    xticklabels: object = None
    yticklabels: object = None
    rot: float = 0
    rot_y: float = 0
    set_grid: bool = False
    set_spines: bool = False
    spines_color: str = "black"
    set_xaxis: bool = True
    set_yaxis: bool = True
    set_ticks: bool = True
    set_xticks: bool = True
    set_yticks: bool = True
    top_ticks: bool = False
    right_ticks: bool = False
    left_ticks: bool = True
    unset_yticks: bool = False
    set_xint: bool = False
    set_yint: bool = False
    set_twinx: bool = False
    set_twiny: bool = False
    xlog: bool = False
    ylog: bool = False
    ysymlog: bool = False
    yscale: str | None = None
    log: bool = False
    sci_x: int | None = None
    invert_x: bool = False
    invert_y: bool = False

    @classmethod
    def from_kwargs(cls, kwargs):
        """Build a spec from flat keywords, popping the ones it knows.

        ``flip_x`` / ``flip_y`` / ``sci`` are deprecated aliases of
        ``invert_x`` / ``invert_y`` / ``sci_x``.
        """
        for old, new in (('flip_x', 'invert_x'), ('flip_y', 'invert_y'),
                         ('sci', 'sci_x')):
            if old in kwargs:
                warnings.warn(f"'{old}' is deprecated, use '{new}'",
                              DeprecationWarning, stacklevel=3)
                kwargs.setdefault(new, kwargs.pop(old))
        return cls(**{f.name: kwargs.pop(f.name)
                      for f in fields(cls) if f.name in kwargs})


# AxisSpec fields plus the deprecated aliases and the no-op reset_axis
AXIS_OPTIONS = frozenset(f.name for f in fields(AxisSpec)) | {
    'flip_x', 'flip_y', 'sci', 'reset_axis'}


def prepare_axis(ax, spec=None, **kwargs):
    """Prepare an axis for any x-y plot.

    Options come from ``spec`` (an ``AxisSpec``) or, for convenience, from the
    same names as flat keywords. Returns the axis (a new one for ``set_twinx``
    / ``set_twiny``) and the keywords that were not axis options.
    """
    if spec is None:
        spec = AxisSpec.from_kwargs(kwargs)
    set_xticks, set_yticks = spec.set_xticks, spec.set_yticks
    xlabel, ylabel = spec.xlabel, spec.ylabel

    if spec.set_twinx:
        ax.set_zorder(1)
        ax = ax.twinx()
        ax.set_zorder(1)
    if spec.set_twiny:
        ax = ax.twiny()

    ax.spines['bottom'].set_visible(spec.set_xaxis)
    ax.spines['left'].set_visible(spec.set_yaxis)
    if not spec.set_yaxis:
        ylabel = None
        set_yticks = False
    if not spec.set_xaxis:
        xlabel = None
        set_xticks = False

    ax.spines["right"].set_visible(False)
    if spec.set_twinx:
        ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(spec.set_spines)
    if spec.set_spines:
        for side in ('bottom', 'top', 'left', 'right'):
            ax.spines[side].set_color(spec.spines_color)

    ax.grid(spec.set_grid)
    ax.tick_params(
        direction="out",
        length=pst.get_preset().tick_length,
        width=pst.get_preset().axes_width,
        colors=spec.ylabel_color, grid_alpha=0.5
    )

    if not has_labels(ax):
        ax.set_xlabel(xlabel, color=spec.x_label_color)
        ax.set_ylabel(ylabel, color=spec.ylabel_color)

        if spec.xlabel_pos is not None:
            if spec.xlabel_pos == "right":
                ax.xaxis.set_label_coords(1.0, -0.2)
            else:
                ax.xaxis.set_label_coords(spec.xlabel_pos)
        if spec.ylabel_pos is not None:
            ax.yaxis.set_label_coords(x=spec.ylabel_pos[0],
                                      y=spec.ylabel_pos[1])

        if spec.xlim is not None:
            ax.set_xlim(spec.xlim)
        if spec.ylim is not None:
            ax.set_ylim(spec.ylim)

        left_ticks, right_ticks = spec.left_ticks, spec.right_ticks
        if spec.set_ticks is False:
            left_ticks = right_ticks = False
        if spec.set_twinx:
            left_ticks, right_ticks = False, True
        ax.tick_params(axis="x", labelrotation=spec.rot,
                       bottom=set_xticks, top=spec.top_ticks)
        ax.tick_params(axis="y", labelrotation=spec.rot_y,
                       left=left_ticks if set_yticks else False,
                       right=right_ticks)
        if spec.xticks is not None and set_xticks:
            ax.set_xticks(spec.xticks)
        if spec.xticklabels is not None and set_xticks:
            ax.set_xticklabels(spec.xticklabels)
        if spec.set_xint:
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        if spec.set_yint:
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

    set_title(title=spec.title, ax=ax, **kwargs)

    ylog = spec.ylog or spec.yscale == 'log'
    ysymlog = spec.ysymlog or spec.yscale == 'symlog'
    if ylog and ysymlog:
        raise ValueError('ylog and ysymlog cannot be True at the same time!')
    if ylog:
        ax.set_yscale("log")
    if ysymlog:
        ax.set_yscale("symlog")
    if spec.xlog:
        ax.set_xscale("log")

    if spec.log:
        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.xaxis.set_major_locator(
            mpl.ticker.LogLocator(base=10.0, numticks=10))
        ax.xaxis.set_minor_locator(mpl.ticker.LogLocator(
            base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    if not spec.set_ticks or not set_xticks or not set_yticks:
        if not set_xticks:
            ax.xaxis.set_ticklabels([])
        if not set_yticks:
            ax.yaxis.set_ticklabels([])

    if spec.yticks is not None:
        ax.set_yticks(spec.yticks)
        ax.set_yticklabels(spec.yticks if spec.yticklabels is None
                           else spec.yticklabels)
    if spec.unset_yticks:
        ax.yaxis.set_ticklabels([])

    if spec.sci_x is not None:
        ax.ticklabel_format(style="sci", axis="x",
                            scilimits=(spec.sci_x, spec.sci_x))
    if spec.invert_x:
        ax.invert_xaxis()
    if spec.invert_y:
        ax.invert_yaxis()

    return ax, kwargs


def check_geoaxis(ax, check_axis=True):
    import cartopy.mpl.geoaxes as cmga

    if check_axis:
        if isinstance(ax, cmga.GeoAxes) or isinstance(ax, cmga.GeoAxesSubplot):
            return True

    return False


def plot_line(ax, x_range, y_range, **kwargs):
    lw = kwargs.pop("lw", pst.get_preset().line_width)
    color = kwargs.pop("color", "k")
    zorder = kwargs.pop('zorder', 1)

    ax.plot(x_range, y_range,
            linewidth=lw,
            color=color,
            zorder=zorder
            )

    return ax


def plot_vline(ax, x, **kwargs):
    lw = kwargs.pop("lw", pst.get_preset().line_width)
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
    lw = kwargs.pop("lw", pst.get_preset().line_width)
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
                    fontsize=kwargs.pop('fontsize', pst.get_preset().title),
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
    lw = kwargs.pop("lw", pst.get_preset().line_width)
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


@pst.styled
def plot_text(ax, text, xpos=0, ypos=0,
              **kwargs):

    color = kwargs.pop("color", pst.INK)
    zorder = kwargs.pop('zorder', 10)
    rot = kwargs.pop("rot", 0)
    fsize = kwargs.pop('fsize', pst.get_preset().label)
    weight = kwargs.pop('weight', "normal")
    trafo_axis = kwargs.pop('transform', False)
    check_axis = kwargs.pop('check_axis', True)
    box = kwargs.pop('box', False)
    ha = kwargs.pop('ha', None)
    va = kwargs.pop('va', 'baseline')
    family = kwargs.pop('family', None)
    plot_box = dict(facecolor='white', edgecolor='black',
                    pad=10.0) if box else None
    if check_geoaxis(ax, check_axis):
        lon_pos = xpos
        lat_pos = ypos
        ax.text(
            lon_pos,
            lat_pos,
            text,
            horizontalalignment=ha or "center",
            verticalalignment=va,
            family=family,
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
            horizontalalignment=ha or "left",
            verticalalignment=va,
            family=family,
            color=color,
            zorder=zorder,
            rotation=rot,
            fontsize=fsize,
            weight=weight,
            transform=ax.transAxes if trafo_axis else ax.transData,
            bbox=plot_box
        )
    return ax


@pst.styled
def text_box(ax, text, pos="upper right", fsize=None, **kwargs):
    fw = kwargs.pop('fw', "normal")
    fsize = pst.get_preset().label if fsize is None else fsize
    anchored_text = AnchoredText(text, loc=pos,
                                 prop=dict(size=fsize,
                                           fontweight=fw))
    ax.add_artist(anchored_text)

    return ax
