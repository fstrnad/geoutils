"""Basic plotting functions for maps"""
# import matplotlib.cm as cm
import copy
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
import geoutils.plotting.plotting_utils as put
import geoutils.plotting.plot_settings as pst

from importlib import reload

"""
# x - y plots
"""
reload(put)
reload(pst)


def plot_2d(
    y=None,
    x=None,
    z_arr=[],
    lw_arr=[1],
    mk_arr=[""],
    ls_arr=['-'],
    alpha_arr=[1],
    color_arr=None,
    color=None,
    lcmap=None,
    ax=None,
    all_x=False,
    norm=False,
    standardize=False,
    ts_axis=False,
    plot_type='xy',
    set_axis=True,
    **kwargs,
):
    reload(sut)
    reload(gut)
    if y is None:
        raise ValueError("y must be provided!")

    if isinstance(y, (list)):
        y_arr = y
    else:
        y_arr = [y]
    if x is not None:
        if isinstance(x, (list)):
            x_arr = x
        else:
            x_arr = [x]
    else:
        x_arr = None

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

    if ax is None:
        figsize = kwargs.pop("figsize",
                             (8, 5))
        set_axis = True
        if plot_type != 'polar':
            fig, ax = plt.subplots(figsize=(figsize), nrows=1, ncols=1)
        else:
            fig, ax = plt.subplots(figsize=(figsize), nrows=1, ncols=1,
                                   subplot_kw={'projection': 'polar'})
    else:
        fig = ax.get_figure()
    zorder = kwargs.pop('zorder', 0)
    kwargs_init = copy.deepcopy(kwargs)
    if set_axis:
        if ts_axis:
            ax = prepare_ts_x_axis(ax, dates=x_arr[0], **kwargs)
        else:
            ax, kwargs = put.prepare_axis(
                ax, plot_type=plot_type, **kwargs)

    num_items = len(y_arr)
    inverted_z_order = kwargs.pop('inv_z_order', False)
    linearize_xaxis = kwargs.pop('linearize_xaxis', False)
    label = kwargs.pop('label', None)
    # to account that both label or label_arr can be used
    label_arr = kwargs.get("label_arr", [])
    if label is not None:
        if not isinstance(label, (list, np.ndarray)):
            label_arr = [label]
        else:
            label_arr = label

    if plot_type == 'xy' or plot_type == 'scatter':
        if lcmap is not None:
            lcmap, evenly_spaced_interval, ccolors = put.get_arr_colorbar(
                cmap=lcmap, num_items=num_items,)

        for idx in range(num_items):
            if x_arr is None:
                x = np.arange(len(y_arr[idx]))
            else:
                x = x_arr[0] if len(x_arr) == 1 else x_arr[idx]
            if linearize_xaxis:
                x = np.arange(0, len(x))

            y = y_arr[idx] if len(y_arr) > 1 else y_arr[0]

            if len(y) != len(x):
                raise ValueError(
                    f"{x} and {y} must have the same length")
            if isinstance(y, xr.DataArray):
                y = y.values
            if isinstance(x, xr.DataArray):
                x = x.values

            zorder += len(y_arr) - idx if inverted_z_order else idx
            z = z_arr[idx] if len(z_arr) == 1 else None
            if norm is True:
                y = sut.normalize(y)
            if standardize is True:
                y = sut.standardize(y)
                if len(y_err_arr) > 0:
                    y_err_arr[idx] = sut.standardize(y_err_arr[idx])
                if len(y_lb_arr) > 0:
                    y_lb_arr[idx] = sut.standardize(
                        y_lb_arr[idx]) if y_lb_arr[idx] is not None else None
                    y_ub_arr[idx] = sut.standardize(
                        y_ub_arr[idx]) if y_lb_arr[idx] is not None else None

            lw = kwargs.get('lw', None)
            alpha = kwargs.get('alpha', None)

            if lw is None:
                lw = lw_arr[idx] if idx < len(lw_arr) else lw_arr[-1]
            if alpha is None:
                alpha = alpha_arr[idx] if idx < len(alpha_arr) else alpha_arr[-1]
            mk = kwargs.get('mk', None)
            if mk is None:
                mk = mk_arr[idx] if idx < len(mk_arr) else mk_arr[-1]

            mk_size = kwargs.get('marker_size', None)
            mk_size = kwargs.get('mk_size', None)
            mk_size_arr = kwargs.get('mk_size_arr', None)
            if mk_size_arr is not None:
                mk_size = mk_size_arr[idx] if idx < len(
                    mk_size_arr) else mk_size_arr[-1]

            ls = kwargs.get('ls', None)
            if ls is None:
                ls = ls_arr[idx] if idx < len(ls_arr) else ls_arr[-1]

            if len(label_arr) > 0:
                if len(label_arr) >= len(y_arr):
                    label = label_arr[idx]
                elif len(label_arr) == idx-1:  # only plot last label in case of only 1 for multiple datasets
                    label = label_arr[0]
                else:
                    label = None

            if lcmap is None:
                if color is None:
                    if color_arr is None:
                        if len(y_arr) <= len(pst.colors):
                            c = pst.colors[idx]
                        else:
                            c = pst.colors[0]
                    else:
                        if idx < len(color_arr):
                            c = color_arr[idx]
                        else:
                            c = color_arr[-1]
                else:
                    c = color
            else:
                c = ccolors[idx]
            if ts_axis:
                x_0 = np.array(x[0], dtype="datetime64[D]")
                x_end = np.array(x[-1], dtype="datetime64[D]") + \
                    np.timedelta64(int(1), "D")
                x_ts = np.arange(x_0, x_end, dtype="datetime64[D]")
                y_ids = np.nonzero(
                    np.in1d(x_ts, np.array(x, dtype="datetime64[D]")))[0]

                y_ts = np.empty(x_ts.shape)
                y_ts[:] = np.nan
                y_ts[y_ids] = y
                im = ax.plot(x_ts, y_ts, label=label, lw=lw,
                             marker=mk, ls=ls, color=c, alpha=alpha)
            else:
                if z is not None:
                    cmap = kwargs.pop("cmap", "viridis")
                    vmin = kwargs.pop('vmin', None)
                    vmax = kwargs.pop('vmax', None)
                    cm = plt.cm.get_cmap(cmap)
                    im = ax.scatter(x, y, c=z,
                                    label=label,
                                    lw=lw,
                                    marker=mk,
                                    markersize=mk_size,
                                    cmap=cm,
                                    vmin=vmin, vmax=vmax,
                                    alpha=alpha
                                    )
                    plt.colorbar(im, orientation='horizontal', label=label)
                else:
                    im = ax.plot(x, y,
                                 lw=lw,
                                 marker=mk,
                                 markersize=mk_size,
                                 ls=ls,
                                 color=c,
                                 zorder=zorder,
                                 alpha=alpha,
                                 label=label
                                 )

                if len(y_lb_arr) > idx:
                    y_lb = y_lb_arr[idx] if y_lb_arr[idx] is not None else None
                    y_ub = y_ub_arr[idx] if y_ub_arr[idx] is not None else None
                elif len(y_err_arr) > 0:
                    y_err = y_err_arr[idx] if len(y_err_arr) > 0 else None
                    x_err = x_err_arr[idx] if len(x_err_arr) > 0 else None
                    y_lb = np.array(y + y_err / 2, dtype=float)
                    y_ub = np.array(y - y_err / 2, dtype=float)
                else:
                    y_lb = None
                    y_ub = None

                if y_lb is not None:
                    color_arr_ci = kwargs.pop('color_arr_ci', None)
                    c = color_arr_ci if color_arr_ci is not None else c
                    im = ax.fill_between(
                        x,
                        y_lb,
                        y_ub,
                        color=c,
                        alpha=0.5,
                    )

                plot_errorbar = kwargs.pop('plot_errorbars', False)
                if plot_errorbar:
                    ax.errorbar(x, y, xerr=x_err, yerr=y_err,
                                label=label, lw=lw, marker=mk, ls=ls, color=c,
                                capsize=2)

    # ############# Plotting  bar ################
    elif plot_type == 'bar':
        import seaborn as sns

        # Bar plot
        stacked = kwargs.pop('stacked', False)
        df = pd.DataFrame(dict(
            X=x_arr[0]
        ))
        edgecolor = kwargs.pop('edge_color', None)
        fill_bar = kwargs.pop('fill_bar', True)
        label_arr_tmp = [0] if len(label_arr) == 0 else label_arr

        for idx, arr in enumerate(y_arr):
            df[label_arr_tmp[idx]] = arr

        if color_arr is None:
            color_arr = sns.color_palette('hls', len(y_arr))

        tidy = df.melt(id_vars='X').rename(columns=str.title)
        if stacked:
            im = df.set_index('X').plot(kind='bar',
                                        stacked=True, color=color_arr, ax=ax)
        else:
            im = sns.barplot(x='X',
                             y='Value',
                             hue='Variable',
                             data=tidy,
                             palette=color_arr,
                             fill=fill_bar,
                             ax=ax,
                             edgecolor=edgecolor,
                             )
        # set again axis, because sns changes it
        ax, kwargs = put.prepare_axis(ax, reset_axis=True, **kwargs_init)
    # ############# Plotting  density ################
    elif plot_type == 'density':
        levels = kwargs.pop("levels", 10)
        for idx in range(num_items):
            label = label_arr[idx] if idx < len(label_arr) else None
            color = color_arr[idx] if color_arr is not None else 'viridis'
            im = sns.kdeplot(x=x_arr[idx], y=y_arr[idx], label=label,
                             cmap=color, ax=ax,
                             alpha=alpha,
                             levels=levels, fill=True)

    if lcmap is not None:
        norm = mpl.colors.Normalize(
            vmin=evenly_spaced_interval.min(), vmax=evenly_spaced_interval.max()
        )
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
        cmap.set_array([])
        fig.colorbar(cmap, ticks=evenly_spaced_interval)

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
        ax = put.set_legend(ax=ax, **kwargs)
    else:
        if plot_type == 'bar':
            ax.legend_.remove()

    return {"ax": ax, "im": im, "fig": fig}


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


def plot_hist(data, ax=None, fig=None,
              plot_type='xy',
              log=False,
              **kwargs):
    reload(sut)
    reload(gut)

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
                 ax=ax, plot_type=plot_type,
                 label=label, **kwargs)

    return dict(
        ax=ax,
        be=be,
        bc=bc_arr[0],
        hc=hc_arr[0],
        fig=fig,
    )


def prepare_ts_x_axis(ax, dates, **kwargs):

    sy, ey = tu.get_start_end_date(data=dates)
    ax, kwargs = put.prepare_axis(ax, **kwargs)
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


def plot_null_model(
    arr, title="Null Model for Event Synchronization",
):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax, kwargs = put.prepare_axis(ax)
    ax.set_title(title)
    im = ax.imshow(arr, interpolation="None", cmap="coolwarm", origin="lower")
    # ax.set_aspect('equal')
    ax.set_xlabel("number of events j")
    ax.set_ylabel("number of events i")
    put.make_colorbar(ax, im, label="# Sync Events")
    return ax
