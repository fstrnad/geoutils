"""Basic plotting functions for maps"""
# import matplotlib.cm as cm
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


def set_legend(ax,
               fig=None,
               label_arr=[],
               legend_items=None,
               **kwargs):
    if fig is None:
        fig = ax
    loc = kwargs.pop("loc", "upper right")
    box_loc = kwargs.pop('box_loc', (0.96, 1))
    box_loc = (0, 0) if loc == 'under' else box_loc
    bbox_to_anchor = box_loc if loc == "outside" or loc == 'under' else None
    loc = 'upper left' if loc == 'outside' or loc == 'under' else loc  # loc is set above
    ncol_legend = kwargs.pop('ncol_legend', 1)
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

    return ax


def plot_xy(
    x_arr,
    y_arr,
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
    color_arr_ci=None,
    color=None,
    lcmap=None,
    ax=None,
    all_x=False,
    norm=False,
    stdize=False,
    ts_axis=False,
    kde=False,
    plot_type='xy',
    **kwargs,
):
    reload(sut)
    reload(gut)
    if ax is None:
        figsize = kwargs.pop("figsize", (8, 5))
        if plot_type != 'polar':
            fig, ax = plt.subplots(figsize=(figsize), nrows=1, ncols=1)
        else:
            fig, ax = plt.subplots(figsize=(figsize), nrows=1, ncols=1,
                                   subplot_kw={'projection': 'polar'})
    else:
        fig = ax.get_figure()

    if plot_type != 'bar':
        if ts_axis:
            ax = prepare_ts_x_axis(ax, dates=x_arr[0], **kwargs)
        else:
            ax, kwargs = put.prepare_axis(ax, plot_type=plot_type, **kwargs)

        num_items = len(y_arr) if len(y_arr) >= len(x_arr) else len(x_arr)
        if lcmap is not None:
            evenly_spaced_interval = np.linspace(0, 1, num_items)
            lcmap = plt.get_cmap(lcmap)
            ccolors = [lcmap(x) for x in evenly_spaced_interval]

        alpha = kwargs.pop('alpha', 1)
        fill_between = kwargs.pop('fill_between', False)
        inverted_z_order = kwargs.pop('inv_z_order', False)
        for idx in range(num_items):
            x = x_arr[idx] if len(x_arr) > 1 else x_arr[0]
            # if type(x) == xr.DataArray and ts_axis is False:
            #     x = np.arange(0, len(x))

            y = y_arr[idx] if len(y_arr) > 1 else y_arr[0]
            zorder = len(y_arr) - idx if inverted_z_order else idx
            z = z_arr[idx] if len(z_arr) == 1 else None
            if norm is True:
                y = sut.normalize(y)
            if stdize is True:
                y = sut.standardize(y)
                if len(y_err_arr) > 0:
                    y_err_arr[idx] = sut.normalize(y_err_arr[idx])
                if len(y_lb_arr) > 0:
                    y_lb_arr[idx] = sut.normalize(y_lb_arr[idx])
                    y_ub_arr[idx] = sut.normalize(y_ub_arr[idx])

            lw = lw_arr[idx] if idx < len(lw_arr) else lw_arr[-1]
            mk = mk_arr[idx] if idx < len(mk_arr) else mk_arr[-1]
            ls = ls_arr[idx] if idx < len(ls_arr) else ls_arr[-1]
            label = label_arr[idx] if idx < len(label_arr) else None
            if lcmap is None:
                if color is None:
                    if color_arr is None:
                        c = pst.colors[idx]
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

                if kde:
                    levels = kwargs.pop("levels", 10)
                    cmap = kwargs.pop("cmap", "viridis")
                    cbar = kwargs.pop("cbar", True)
                    im = sns.kdeplot(
                        x=x,
                        y=y,
                        ax=ax,
                        fill=True,
                        levels=levels,
                        label=label,
                        cmap=cmap,
                        cbar=cbar,
                    )
                elif z is not None:
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
                if len(x_err_arr) > idx or len(y_err_arr) > idx or len(y_lb_arr) > idx:

                    if len(y_lb_arr) > idx:
                        y_lb = y_lb_arr[idx]
                        y_ub = y_ub_arr[idx]
                    else:
                        y_err = y_err_arr[idx] if len(y_err_arr) > 0 else None
                        x_err = x_err_arr[idx] if len(x_err_arr) > 0 else None
                        y_lb = np.array(y + y_err / 2, dtype=float)
                        y_ub = np.array(y + y_err / 2, dtype=float)

                    if fill_between:
                        c = color_arr_ci[idx] if color_arr_ci is not None else c
                        im = ax.fill_between(
                            x,
                            y_lb,
                            y_ub,
                            color=c,
                            alpha=0.5,
                            # label=label,
                        )

                    ax.errorbar(x, y, xerr=x_err, yerr=y_err,
                                label=label, lw=lw, marker=mk, ls=ls, color=c,
                                capsize=2)
    else:
        # Bar plot
        stacked = kwargs.pop('stacked', False)
        df = pd.DataFrame(dict(
            X=x_arr
        ))

        for idx, arr in enumerate(y_arr):
            df[label_arr[idx]] = arr

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
                             ax=ax,
                             )
        ax, kwargs = put.prepare_axis(ax, **kwargs)

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


def plot_hist(data, ax=None, label_arr=None, log=False, color_arr=None, **kwargs):
    reload(sut)
    reload(gut)
    if ax is None:
        figsize = kwargs.pop("figsize", (6, 4))
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax, kwargs = put.prepare_axis(ax, log=log, **kwargs)

    density = kwargs.pop("density", False)
    bar = kwargs.pop("bar", True)
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
            ax.plot(bc, hc, "x", color=c, label=label)
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


def violinplot(data, x, y, ax=None, **kwargs):
    hue = kwargs.pop('hue', None)
    if ax is None:
        figsize = kwargs.pop("figsize", (6, 4))
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax, kwargs = put.prepare_axis(ax=ax, **kwargs)

    ax = sns.violinplot(ax=ax, data=data, x=x, y=y, hue=hue,)

    ax = set_legend(ax, **kwargs)

    return ax


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


#  ################# Histogramms  #################


def plot_cnt_occ_ensemble(
    ds,
    mean_cnt_arr,
    std_cnt_arr=None,
    savepath=None,
    label_arr=None,
    polar=False,
    **kwargs,
):
    figsize = (10, 6)
    if polar is True:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={
                               "projection": "polar"})
        ax.margins(y=0)
        x_pos = np.deg2rad(np.linspace(0, 360, 13))
        ax.set_xticks(x_pos)
        ax.set_xticklabels(ds.months + [""],)
        ax.set_rlabel_position(60)  # get radial labels away from plotted line
        ax.set_rticks([0.0, 0.1, 0.2, 0.3, 0.4])  # Less radial ticks
        # rotate the axis arbitrarily, just replace pi with the angle you want.
        ax.set_theta_offset(np.pi)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        put.prepare_axis(ax, **kwargs)
        ax.set_xlabel("Month")
        ax.set_ylabel("Relative Frequency")
        x_pos = ds.months

    if std_cnt_arr is None:
        std_cnt_arr = np.zeros_like(mean_cnt_arr)
    if len(mean_cnt_arr) != len(std_cnt_arr):
        raise ValueError(
            f"Mean len {len(mean_cnt_arr)} != Std len {len(std_cnt_arr)}")

    sum_mean_cnt = np.zeros_like(mean_cnt_arr[0])
    for idx in range(len(mean_cnt_arr)):
        mean_cnt = np.array(mean_cnt_arr[idx], dtype=float)
        std_cnt = np.array(std_cnt_arr[idx], dtype=float)
        if polar is True:
            mean_cnt = np.append(mean_cnt, np.array([mean_cnt[0]]), axis=0)
            std_cnt = np.append(std_cnt, np.array([std_cnt[0]]), axis=0)
        if label_arr is None:
            label = None
        else:
            label = label_arr[idx]

        width = 1 / len(mean_cnt_arr) - 0.1
        x_pos = np.arange(len(mean_cnt)) + width * idx
        ax.bar(
            x_pos,
            mean_cnt,
            yerr=(std_cnt),
            ecolor=pst.colors[idx],
            capsize=10,
            width=width,
            color=pst.colors[idx],
            label=label,
        )
        sum_mean_cnt += mean_cnt
        # print(sum_mean_cnt)
    # off_set = len(mean_cnt_arr)
    ax.set_xticks(x_pos)  # + width/off_set
    ax.set_xticklabels(ds.months)
    ax.grid(True)

    if label_arr is not None:
        set_legend(ax, label_arr=label_arr, **kwargs)

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")

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
