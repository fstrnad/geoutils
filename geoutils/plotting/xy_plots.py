"""Basic plotting functions for maps"""
# import matplotlib.cm as cm
import copy
import xarray as xr
import pandas as pd
import geoutils.utils.time_utils as tu
import seaborn as sns
import geoutils.utils.statistic_utils as sut
import geoutils.utils.general_utils as gut
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# plt.style.use('./src/matplotlib_style.py')
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


def set_legend(ax,
               fig=None,
               label_arr=[],
               legend_items=None,
               **kwargs):
    if fig is None:
        fig = ax
    loc = kwargs.pop("loc", "upper right")
    box_loc = kwargs.pop('box_loc', (0.96, 1))
    box_loc = (0.1, 0) if loc == 'under' else box_loc
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


def plot_xy(
    y_arr,
    x_arr=None,
    z_arr=[],
    x_err_arr=[],
    y_err_arr=[],
    y_lb_arr=[],
    y_ub_arr=[],
    label_arr=[],
    lw_arr=[1],
    mk_arr=[""],
    ls_arr=['-'],
    color_arr=None,
    color=None,
    lcmap=None,
    ax=None,
    all_x=False,
    norm=False,
    standardize=False,
    ts_axis=False,
    plot_type='xy',
    set_axis=False,
    **kwargs,
):
    reload(sut)
    reload(gut)
    if not isinstance(y_arr[0], (list, np.ndarray, xr.DataArray)):
        y_arr = [y_arr]
    if x_arr is not None and isinstance(x_arr[0], (list, np.ndarray, xr.DataArray)):
        if len(y_arr) != len(x_arr):
            raise ValueError(
                f"x and y arrays must have the same length, but are {len(x_arr)} and {len(y_arr)}!")
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
    filled = kwargs.pop('filled', False)
    kwargs_init = copy.deepcopy(kwargs)
    if set_axis:
        if ts_axis:
            ax = prepare_ts_x_axis(ax, dates=x_arr[0], **kwargs)
        else:
            ax, kwargs = put.prepare_axis(
                ax, plot_type=plot_type, **kwargs)

    num_items = len(y_arr)
    alpha = kwargs.pop('alpha', 1)
    inverted_z_order = kwargs.pop('inv_z_order', False)
    linearize_xaxis = kwargs.pop('linearize_xaxis', False)

    # ############# Plotting  Scatter ################
    if plot_type == 'xy' or plot_type == 'scatter':
        if lcmap is not None:
            lcmap, evenly_spaced_interval, ccolors = put.get_arr_colorbar(
                cmap=lcmap, num_items=num_items,)

        for idx in range(num_items):
            if x_arr is None:
                x_arr = [np.arange(len(y_arr[idx]))]
            if isinstance(x_arr[0], (list, np.ndarray, xr.DataArray)):
                x = x_arr[idx]
            else:
                x = x_arr
            if linearize_xaxis:
                x = np.arange(0, len(x))

            y = y_arr[idx] if len(y_arr) > 1 else y_arr[0]
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

            lw = lw_arr[idx] if idx < len(lw_arr) else lw_arr[-1]
            mk = mk_arr[idx] if idx < len(mk_arr) else mk_arr[-1]
            ls = ls_arr[idx] if idx < len(ls_arr) else ls_arr[-1]
            label = label_arr[idx] if idx < len(label_arr) else None
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
                                    cmap=cm,
                                    vmin=vmin, vmax=vmax,
                                    alpha=alpha
                                    )
                    plt.colorbar(im, orientation='horizontal', label=label)
                else:
                    im = ax.plot(x, y,
                                 lw=lw,
                                 marker=mk,
                                 ls=ls,
                                 color=c,
                                 zorder=zorder,
                                 alpha=alpha,
                                 label=label
                                 )
                    if filled:
                        alpha_fill = kwargs.get('alpha_fill', 1)
                        offset_fill = kwargs.get('offset_fill', 0)
                        if offset_fill == 0:
                            ax.fill_between(
                                x, y, color=c, alpha=alpha_fill, zorder=zorder)
                        else:
                            y_linear = np.full_like(x, offset_fill)

                            ax.fill_between(x, y, y_linear, where=(y <= y_linear),
                                            color=c, alpha=alpha_fill,
                                            zorder=zorder)

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
                        # label=label,
                    )

                plot_errorbar = kwargs.pop('plot_errorbars', False)
                if plot_errorbar:
                    ax.errorbar(x, y, xerr=x_err, yerr=y_err,
                                label=label, lw=lw, marker=mk, ls=ls, color=c,
                                capsize=2)

    # ############# Plotting  bar ################
    elif plot_type == 'bar':
        # Bar plot
        stacked = kwargs.pop('stacked', False)
        df = pd.DataFrame(dict(
            X=x_arr
        ))
        edgecolor = kwargs.pop('edge_color', None)
        fill_bar = kwargs.pop('fill_bar', True)
        label_arr_tmp = [0] if len(label_arr) == 0 else label_arr

        for idx, arr in enumerate(y_arr):
            # print(idx, len(label_arr_tmp), len(y_arr))
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
    if make_legend and len(label_arr) > 0:
        # labels and legend items are already within ax-object
        ax = set_legend(ax=ax, **kwargs)
    else:
        if plot_type == 'bar':
            ax.legend_.remove()

    return {"ax": ax, "im": im, "fig": fig}


def bar_plot_xy(x_arr, y_arr, label_arr,
                ax=None,  fig=None, log=False, color_arr=None, **kwargs):
    if ax is None:
        figsize = kwargs.pop("figsize", (6, 4))
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    df = pd.DataFrame(dict(
        X=x_arr
    ))
    for idx, arr in enumerate(y_arr):
        df[label_arr[idx]] = arr

    mk_legend = kwargs.pop('set_legend', True)

    if color_arr is None:
        color_arr = sns.color_palette('hls', len(y_arr))

    tidy = df.melt(id_vars='X').rename(columns=str.title)
    ax = sns.barplot(x='X', y='Value',
                     hue='Variable',
                     data=tidy,
                     palette=color_arr,
                     ax=ax,
                     )
    ax, kwargs = put.prepare_axis(ax, log=log, **kwargs)

    # ax.set_xticks(np.arange(len(x_arr)))
    # ax.set_xticklabels(x_arr)
    if mk_legend:
        ax = set_legend(ax, label_arr=label_arr, **kwargs)
    else:
        ax.legend_.remove()

    sci = kwargs.pop("sci", None)
    if sci is not None:
        ax.ticklabel_format(style="sci", axis="x", scilimits=(sci, sci))
    return {'ax': ax,
            'fig': fig,
            }


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


def plot_hist(data, ax=None, fig=None, label_arr=None, log=False, color_arr=None, **kwargs):
    reload(sut)
    reload(gut)

    density = kwargs.pop("density", False)
    bar = kwargs.pop("bar", False)
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

    if ax is None:
        figsize = kwargs.pop("figsize", (6, 4))
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax, kwargs = put.prepare_axis(ax, log=log, **kwargs)

    hc_arr = []
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
        label = label_arr[idx] if label_arr is not None else []
        c = pst.colors[idx] if color_arr is None else color_arr[idx]
        if bar:
            width = (bc[1] - bc[0]) * 0.4
            x_pos = bc + width * idx
            ax.bar(
                x_pos,
                hc,
                yerr=None,
                ecolor=c,
                capsize=2,
                width=width,
                color=c,
                label=label,
            )
        else:
            ax.plot(bc, hc, "x", color=c, label=label, ls='-')
    if label_arr is not None:
        ax = set_legend(ax, label_arr=label_arr, **kwargs)

    sci = kwargs.pop("sci", None)
    if sci is not None:
        ax.ticklabel_format(style="sci", axis="x", scilimits=(sci, sci))

    return dict(
        ax=ax,
        be=be,
        bc=bc,
        hc=hc_arr,
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
