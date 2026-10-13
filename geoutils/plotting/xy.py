"""Basic plotting functions for x-y (non-map) plots."""
import copy
from dataclasses import dataclass
import xarray as xr
import pandas as pd
import geoutils.utils.time_utils as tu
import geoutils.utils.statistic_utils as sut
import geoutils.utils.general_utils as gut
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import geoutils.plotting.axes as pax
import geoutils.plotting.colors as pcol
import geoutils.plotting.style as pst
from geoutils.plotting.result import PlotResult


"""
# x - y plots
"""


def _as_series_list(data):
    """Normalize a plot input into a list of series.

    A single series (a 1d ``np.ndarray``, ``pd.Series``/``pd.Index``,
    ``xr.DataArray`` or a flat list/tuple of scalars) is wrapped into a
    one-element list. A list/tuple whose elements are themselves array-like is
    treated as multiple series and returned as-is. This lets callers pass a
    single array directly while still expecting a list for multiple plots.
    """
    if isinstance(data, (list, tuple)):
        if len(data) > 0 and isinstance(
            data[0], (list, tuple, np.ndarray, xr.DataArray, pd.Series, pd.Index)
        ):
            return list(data)
        return [data]
    return [data]


def _pick(values, idx):
    """Return ``values[idx]``, or the last element when ``idx`` is past the end."""
    return values[idx] if idx < len(values) else values[-1]


def _override(kwargs, key, values, idx):
    """``kwargs[key]`` if given (shared by all series), else the per-series pick."""
    value = kwargs.get(key)
    return _pick(values, idx) if value is None else value


@dataclass(frozen=True)
class _XYSpec:
    """Per-series styling and error-band inputs for ``_draw_xy``."""
    z_arr: list
    lw_arr: list
    mk_arr: list
    ls_arr: list
    alpha_arr: list
    color_arr: list | None
    color: object
    series_colors: object
    y_lb_arr: list
    y_ub_arr: list
    y_err_arr: list
    x_err_arr: list
    label_arr: list
    zorder: float
    norm: bool
    standardize: bool
    inv_z_order: bool
    linearize_xaxis: bool


def _series_color(spec, idx):
    if spec.series_colors is not None:
        return spec.series_colors[idx]
    if spec.color is not None:
        return spec.color
    if spec.color_arr is not None:
        return _pick(spec.color_arr, idx)
    return None  # let matplotlib take the next colour of the axes' cycle


def _series_label(label_arr, idx, num_items):
    if len(label_arr) >= num_items:
        return label_arr[idx]
    return label_arr[0] if idx == num_items - 1 else None


def _draw_xy(ax, y_arr, x_arr, spec, kwargs):
    """Draw line/scatter series; returns the last artist and the last label."""
    num_items = len(y_arr)
    y_err_arr, y_lb_arr, y_ub_arr = spec.y_err_arr, spec.y_lb_arr, spec.y_ub_arr
    zorder = spec.zorder
    label = None
    im = None
    for idx in range(num_items):
        if x_arr is None:
            x = np.arange(len(y_arr[idx]))
        else:
            x = x_arr[0] if len(x_arr) == 1 else x_arr[idx]
        if spec.linearize_xaxis:
            x = np.arange(0, len(x))

        y = y_arr[idx] if num_items > 1 else y_arr[0]
        if len(y) != len(x):
            raise ValueError(f"{x} and {y} must have the same length")
        if isinstance(y, xr.DataArray):
            y = y.values
        if isinstance(x, xr.DataArray):
            x = x.values

        zorder += num_items - idx if spec.inv_z_order else idx
        z = spec.z_arr[idx] if len(spec.z_arr) == 1 else None
        if spec.norm is True:
            y = sut.normalize(y)
        if spec.standardize is True:
            y = sut.standardize(y)
            if len(y_err_arr) > 0:
                y_err_arr[idx] = sut.standardize(y_err_arr[idx])
            if len(y_lb_arr) > 0:
                y_lb_arr[idx] = sut.standardize(
                    y_lb_arr[idx]) if y_lb_arr[idx] is not None else None
                y_ub_arr[idx] = sut.standardize(
                    y_ub_arr[idx]) if y_lb_arr[idx] is not None else None

        lw = _override(kwargs, 'lw', spec.lw_arr, idx)
        alpha = _override(kwargs, 'alpha', spec.alpha_arr, idx)
        mk = _override(kwargs, 'mk', spec.mk_arr, idx)
        ls = _override(kwargs, 'ls', spec.ls_arr, idx)
        mk_size = kwargs.get('mk_size', kwargs.get('marker_size', None))
        if kwargs.get('mk_size_arr') is not None:
            mk_size = _pick(kwargs['mk_size_arr'], idx)
        if len(spec.label_arr) > 0:
            label = _series_label(spec.label_arr, idx, num_items)
        c = _series_color(spec, idx)

        if z is not None:
            cmap = kwargs.pop("cmap", "viridis")
            cm = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
            im = ax.scatter(x, y, c=z, label=label, lw=lw, marker=mk,
                            s=mk_size, cmap=cm,
                            vmin=kwargs.pop('vmin', None),
                            vmax=kwargs.pop('vmax', None), alpha=alpha)
            plt.colorbar(im, orientation='horizontal', label=label)
            continue

        im = ax.plot(x, y, lw=lw, marker=mk, markersize=mk_size, ls=ls,
                     color=c, zorder=zorder, alpha=alpha, label=label)
        c = im[0].get_color() if c is None else c

        y_err = y_err_arr[idx] if len(y_err_arr) > 0 else None
        x_err = spec.x_err_arr[idx] if len(spec.x_err_arr) > 0 else None
        if len(y_lb_arr) > idx:
            y_lb, y_ub = y_lb_arr[idx], y_ub_arr[idx]
        elif y_err is not None:
            y_lb = np.array(y + y_err / 2, dtype=float)
            y_ub = np.array(y - y_err / 2, dtype=float)
        else:
            y_lb = y_ub = None

        if y_lb is not None:
            color_arr_ci = kwargs.pop('color_arr_ci', None)
            c = color_arr_ci if color_arr_ci is not None else c
            im = ax.fill_between(x, y_lb, y_ub, color=c,
                                 alpha=kwargs.get('alpha_err', 0.3))

        if kwargs.pop('plot_errorbars', False):
            ax.errorbar(x, y, xerr=x_err, yerr=y_err, label=label, lw=lw,
                        marker=mk, ls=ls, color=c, capsize=2)
    return im, label


def _draw_bar(ax, y_arr, x_arr, label_arr, color_arr, kwargs, kwargs_init):
    import seaborn as sns

    stacked = kwargs.pop('stacked', False)
    edgecolor = kwargs.pop('edge_color', None)
    fill_bar = kwargs.pop('fill_bar', True)
    df = pd.DataFrame(dict(X=x_arr[0]))
    names = [0] if len(label_arr) == 0 else label_arr
    for idx, arr in enumerate(y_arr):
        df[names[idx]] = arr
    if color_arr is None:
        color_arr = sns.color_palette('hls', len(y_arr))

    if stacked:
        im = df.set_index('X').plot(kind='bar', stacked=True,
                                    color=color_arr, ax=ax)
    else:
        tidy = df.melt(id_vars='X').rename(columns=str.title)
        im = sns.barplot(x='X', y='Value', hue='Variable', data=tidy,
                         palette=color_arr, fill=fill_bar, ax=ax,
                         edgecolor=edgecolor)
        ax.set_xlabel(kwargs_init.pop("xlabel", "") or "")
        ax.set_ylabel(kwargs_init.pop("ylabel", "") or "")
    # seaborn resets the axis, so prepare it again
    ax, kwargs = pax.prepare_axis(ax, reset_axis=True, **kwargs_init)
    return im, ax, kwargs


def _draw_density(ax, y_arr, x_arr, label_arr, color_arr, alpha_arr, kwargs):
    import seaborn as sns

    levels = kwargs.pop("levels", 10)
    im = label = None
    for idx in range(len(y_arr)):
        label = label_arr[idx] if idx < len(label_arr) else None
        im = sns.kdeplot(
            x=x_arr[idx], y=y_arr[idx], label=label, ax=ax,
            cmap=color_arr[idx] if color_arr is not None else 'viridis',
            alpha=_pick(alpha_arr, idx), levels=levels, fill=True)
    return im, label


@pst.styled
def plot_2d(
    y=None,
    x=None,
    z_arr=None,
    lw_arr=None,
    mk_arr=None,
    ls_arr=None,
    alpha_arr=None,
    color_arr=None,
    color=None,
    lcmap=None,
    ax=None,
    all_x=False,
    norm=False,
    standardize=False,
    plot_type='xy',
    **kwargs,
):
    """Plot one or several x-y series, bars or densities.

    Parameters
    ----------
    y, x : array-like or list of array-likes
        One series, or a list of series. ``x`` defaults to ``arange(len(y))``;
        a single ``x`` is shared by all series.
    z_arr : list, optional
        Values to colour a scatter plot by (``plot_type='xy'`` with one entry).
    lw_arr, mk_arr, ls_arr, alpha_arr, color_arr : list, optional
        Per-series line width, marker, line style, alpha and colour; the last
        entry is reused for series beyond the end of the list.
    color : colour, optional
        One colour for all series (wins over ``color_arr``).
    lcmap : str or Colormap, optional
        Colour the series along this colormap and add a colorbar.
    ax : matplotlib Axes, optional
        Draw into this axis instead of creating a figure.
    norm, standardize : bool
        Rescale each series to [0, 1] or to zero mean and unit variance.
    plot_type : {'xy', 'scatter', 'bar', 'density'}

    Other keyword arguments
    -----------------------
    label / label_arr, y_lb / y_ub / y_err / x_err, alpha_err, color_arr_ci,
    plot_errorbars, zorder, inv_z_order, linearize_xaxis, all_x, lw, alpha, ls,
    mk, mk_size, mk_size_arr (override the per-series lists), stacked,
    edge_color, fill_bar (bars), levels (density), sci, sci_x, set_legend, and
    everything accepted by ``axes.prepare_axis`` and ``axes.set_legend``.

    Returns
    -------
    PlotResult with ``ax``, ``fig`` and the last drawn artist ``im``.
    """
    if y is None:
        raise ValueError("y must be provided!")

    y_arr = _as_series_list(y)
    x_arr = _as_series_list(x) if x is not None else None

    y_lb = kwargs.pop('y_lb', [])
    y_ub = kwargs.pop('y_ub', [])
    y_lb_arr = y_lb if isinstance(y_lb, list) else [y_lb]
    y_ub_arr = y_ub if isinstance(y_ub, list) else [y_ub]
    if len(y_lb_arr) > 0 and len(y_ub_arr) == 0:
        y_ub_arr = y_lb_arr
    if len(y_ub_arr) > 0 and len(y_lb_arr) == 0:
        y_lb_arr = y_ub_arr

    y_err = kwargs.pop('y_err', [])
    y_err_arr = y_err if isinstance(y_err, list) else [y_err]
    x_err = kwargs.pop('x_err', [])
    x_err_arr = x_err if isinstance(x_err, list) else [x_err]
    zorder = kwargs.pop('zorder', 0)
    kwargs_init = copy.deepcopy(kwargs)

    ax, fig, kwargs = create_axis(ax, plot_type, **kwargs)

    num_items = len(y_arr)
    inv_z_order = kwargs.pop('inv_z_order', False)
    linearize_xaxis = kwargs.pop('linearize_xaxis', False)
    label = kwargs.pop('label', None)
    # both label and label_arr can be used
    label_arr = kwargs.get("label_arr", [])
    if label is not None:
        label_arr = label if isinstance(label, (list, np.ndarray)) else [label]
    alpha_arr = [1] if alpha_arr is None else alpha_arr

    series_colors = None
    if lcmap is not None:
        series_colors = (mpl.colormaps[lcmap] if isinstance(lcmap, str)
                         else lcmap)(np.linspace(0, 1, num_items))

    if plot_type in ('xy', 'scatter'):
        spec = _XYSpec(
            z_arr=[] if z_arr is None else z_arr,
            lw_arr=[1] if lw_arr is None else lw_arr,
            mk_arr=[""] if mk_arr is None else mk_arr,
            ls_arr=['-'] if ls_arr is None else ls_arr,
            alpha_arr=alpha_arr, color_arr=color_arr, color=color,
            series_colors=series_colors, y_lb_arr=y_lb_arr, y_ub_arr=y_ub_arr,
            y_err_arr=y_err_arr, x_err_arr=x_err_arr, label_arr=label_arr,
            zorder=zorder, norm=norm, standardize=standardize,
            inv_z_order=inv_z_order, linearize_xaxis=linearize_xaxis)
        im, label = _draw_xy(ax, y_arr, x_arr, spec, kwargs)
    elif plot_type == 'bar':
        im, ax, kwargs = _draw_bar(ax, y_arr, x_arr, label_arr, color_arr,
                                   kwargs, kwargs_init)
    elif plot_type == 'density':
        im, label = _draw_density(ax, y_arr, x_arr, label_arr, color_arr,
                                  alpha_arr, kwargs)

    if lcmap is not None:
        ticks = np.arange(num_items)
        mappable = mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=ticks.min(), vmax=ticks.max()),
            cmap=lcmap)
        mappable.set_array([])
        fig.colorbar(mappable, ax=ax, ticks=ticks)

    if all_x is True:
        ax.set_xticks(x_arr[0])
    sci = kwargs.pop("sci", None)
    if sci is not None:
        ax.ticklabel_format(style="sci", axis="y", scilimits=(sci, sci))
    sci_x = kwargs.pop("sci_x", None)
    if sci_x is not None:
        ax.ticklabel_format(style="sci", axis="x", scilimits=(sci_x, sci_x))

    make_legend = kwargs.pop('set_legend', True)
    if make_legend and label is not None:
        # labels and legend items are already within ax-object
        ax = pax.set_legend(ax=ax, **kwargs)
    elif plot_type == 'bar':
        ax.legend_.remove()

    return PlotResult(ax=ax, im=im, fig=fig)


def create_axis(ax, plot_type, **kwargs):
    if ax is None:
        figsize = kwargs.pop("figsize",
                             (8, 5))
        fig, ax = plt.subplots(figsize=(figsize), nrows=1, ncols=1)

    else:
        fig = ax.get_figure()

    prepare_axis = kwargs.pop('prepare_axis', True)
    if prepare_axis:
        # sci / sci_x are the callers' y / x formatting, applied after drawing
        held = {k: kwargs.pop(k) for k in ('sci', 'sci_x') if k in kwargs}
        ax, kwargs = pax.prepare_axis(ax=ax, **kwargs)
        kwargs.update(held)

    return ax, fig, kwargs


def fill_between(ax, x, y, y2, thresh=0, larger=True,
                 **kwargs):
    if larger:
        ax.fill_between(x, y, y2=y2,
                        where=(y >= y2 + thresh),
                        **kwargs)
    else:
        ax.fill_between(x, y, y2=y2,
                        where=(y <= y2 + thresh),
                        **kwargs)
    return ax


def plot_lines(ax, te, color="Turquoise"):
    ylo, yhi = ax.get_ylim()
    ax.bar(x=te,
           width=1 * np.ones(len(te)),
           height=(yhi - ylo) * np.ones(len(te)),
           bottom=ylo * np.ones(len(te)),
           #    edgecolor="none",
           facecolor=color,
           zorder=10,
           )
    return {'ax': ax}


@pst.styled
def plot_hist(data, ax=None, fig=None,
              plot_type='xy',
              log=False,
              **kwargs):
    ax, fig, kwargs = create_axis(
        ax=ax, plot_type=plot_type, **kwargs)
    density = kwargs.pop("density", False)
    nbins = kwargs.pop("nbins", None)
    bw = kwargs.pop("bw", None)
    if len(np.shape(data)) > 1:
        data_tmp = np.concatenate(data, axis=0)
        data_arr = data
    else:
        if isinstance(data[0], (list, np.ndarray)):
            data_tmp = data[0]
            data_arr = data
        else:
            data_tmp = data
            data_arr = [data]
    if nbins is None and bw is None:
        data_tmp = gut.remove_nans(data_tmp)
        nbins = sut.__doane(data_tmp)

    hc_arr = []
    bc_arr = []
    label = kwargs.pop("label", None)
    for idx, arr in enumerate(data_arr):
        arr = gut.remove_nans(arr)
        if log:
            hc, bc, be = sut.loghist(arr, nbins=nbins, density=density)
        else:
            hc, bc, be = sut.hist(
                arr,
                nbins=nbins,
                bw=bw,
                min_bw=ax.get_xlim()[0],
                max_bw=ax.get_xlim()[1],
                density=density,
            )
        hc_arr.append(hc)
        bc_arr.append(bc)

    im = plot_2d(x=bc_arr, y=hc_arr,
                 ax=ax,
                 plot_type=plot_type,
                 label=label,
                 prepare_axis=False,
                 **kwargs)

    return dict(
        ax=ax,
        be=be,
        bc=bc_arr[0],
        hc=hc_arr[0],
        fig=fig,
    )


def prepare_ts_x_axis(ax, dates, **kwargs):

    sy, ey = tu.get_start_end_date(data=dates)
    ax, kwargs = pax.prepare_axis(ax, **kwargs)
    ax.tick_params(axis="x", labelrotation=90)
    # Text in the x axis will be displayed in 'YYYY' format.
    fmt_form = mdates.DateFormatter("%m-%Y")
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_form))
    fmt_loc = mdates.YearLocator()
    ax.xaxis.set_major_locator(fmt_loc)
    ax.tick_params(
        direction="out", length=pst.SMALL_SIZE / 2, width=1, colors="k", grid_alpha=0.5
    )
    ax.set_xlim(sy, ey)

    return ax


#  ########################  Null Model #########################


@pst.styled
def plot_null_model(
    arr, title="Null Model for Event Synchronization",
):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax, kwargs = pax.prepare_axis(ax)
    ax.set_title(title)
    im = ax.imshow(arr, interpolation="None", cmap="coolwarm", origin="lower")
    # ax.set_aspect('equal')
    ax.set_xlabel("number of events j")
    ax.set_ylabel("number of events i")
    pcol.make_colorbar(ax, im, label="# Sync Events")
    return ax
