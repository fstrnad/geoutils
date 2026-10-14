"""Chart primitives that are not x-y lines: bars, box plots, hexbin, heatmaps, histograms.

All functions draw in the geoutils look, accept ``ax`` to draw into an existing
axis, pass axis options (``title``, ``xlabel``, ``xlim``, ``ylog``, ...) to
``axes.prepare_axis`` and return a ``PlotResult``.
"""
import matplotlib as mpl
import numpy as np

import geoutils.plotting.axes as pax
import geoutils.plotting.colors as pcol
import geoutils.plotting.style as pst
from geoutils.plotting.result import PlotResult
from geoutils.plotting.xy import create_axis

DEFAULT_BAR_SLOT = 0.8  # share of the spacing between bar positions that bars fill
BOX_FACE_ALPHA = 0.35
FLIER_SIZE = 3
FLIER_ALPHA = 0.3
ANNOTATION_FLIP = 0.5  # normalised cell value above which annotations turn white


def _as_list(value):
    if value is None:
        return None
    return list(value) if isinstance(value, (list, tuple, np.ndarray)) else [value]


def _pick(values, idx):
    """``values[idx]``, the last entry once ``idx`` is past the end, None if empty."""
    if not values:
        return None
    return values[idx] if idx < len(values) else values[-1]


def _palette(n):
    cycle = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    return [cycle[i % len(cycle)] for i in range(n)]


def _finish_legend(ax, labels, kwargs):
    if labels:
        pax.set_legend(ax=ax, **kwargs)


@pst.styled
def plot_bar(values, labels=None, x=None, ax=None, series_labels=None,
             colors=None, hatch=None, edgecolor=None, width=None,
             horizontal=False, value_fmt=None, alpha=1.0, **kwargs):
    """Bars for one series, or grouped bars for several.

    Parameters
    ----------
    values : 1d array (one series) or list of 1d arrays (one group member each).
    labels : tick label per category; ``x`` gives numeric positions instead
        (default ``arange``).
    series_labels : legend entry per series.
    colors : one colour per series, or per bar when there is a single series.
    hatch : hatch per series, the last one is reused.
    width : width of one category slot in data units; grouped bars share it.
    horizontal : draw ``barh``.
    value_fmt : ``%``-format; adds the bar value as a label, e.g. ``"%.2f"``.
    """
    series = np.atleast_2d(np.asarray(values, dtype=float))
    n_series, n_cat = series.shape
    pos = np.arange(n_cat, dtype=float) if x is None else np.asarray(x, dtype=float)
    if width is None:
        spacing = np.min(np.diff(np.sort(pos))) if n_cat > 1 else 1.0
        width = DEFAULT_BAR_SLOT * spacing
    bar_width = width / n_series

    colors = _as_list(colors)
    per_bar_colors = n_series == 1 and colors is not None and len(colors) == n_cat
    if colors is None:
        colors = _palette(n_series)
    hatch = _as_list(hatch) or []
    series_labels = _as_list(series_labels) or []

    ax, fig, kwargs = create_axis(ax, "bar", **kwargs)
    draw = ax.barh if horizontal else ax.bar
    containers = []
    for idx in range(n_series):
        offset = (idx - (n_series - 1) / 2) * bar_width
        color = colors if per_bar_colors else _pick(colors, idx)
        container = draw(pos + offset, series[idx], bar_width, color=color,
                         alpha=alpha, hatch=_pick(hatch, idx), edgecolor=edgecolor,
                         label=_pick(series_labels, idx) if idx < len(series_labels) else None)
        if value_fmt is not None:
            ax.bar_label(container, fmt=value_fmt, fontsize=pst.TINY_SIZE, padding=2)
        containers.append(container)

    if labels is not None:
        (ax.set_yticks if horizontal else ax.set_xticks)(pos, labels)
    _finish_legend(ax, series_labels, kwargs)
    return PlotResult(ax=ax, fig=fig, im=containers)


@pst.styled
def plot_box(data, positions=None, labels=None, ax=None, colors=None, widths=0.6,
             showfliers=True, **kwargs):
    """One box per group of samples; whiskers at 1.5 IQR, fliers as faint dots."""
    data = list(data)
    positions = np.arange(len(data)) if positions is None else np.asarray(positions)
    ax, fig, kwargs = create_axis(ax, "box", **kwargs)
    colors = _as_list(colors) or _palette(1)

    parts = ax.boxplot(
        data, positions=positions, widths=widths, patch_artist=True,
        showfliers=showfliers,
        flierprops={"marker": ".", "markersize": FLIER_SIZE, "alpha": FLIER_ALPHA},
        medianprops={"color": pst.INK})
    for idx, box in enumerate(parts["boxes"]):
        box.set_facecolor(mpl.colors.to_rgba(_pick(colors, idx), BOX_FACE_ALPHA))
        box.set_edgecolor(_pick(colors, idx))
    if labels is not None:
        ax.set_xticks(positions, labels)
    return PlotResult(ax=ax, fig=fig, im=parts)


@pst.styled
def plot_hexbin(x, y, ax=None, gridsize=60, log=False, extent=None, cmap="Blues",
                label=None, mincnt=1, **kwargs):
    """Density of a scatter as hexagonal bins; a colorbar is drawn when ``label`` is given.

    ``extent`` is ``(xmin, xmax, ymin, ymax)``. ``log`` colours by log10(count).
    """
    ax, fig, kwargs = create_axis(ax, "hexbin", **kwargs)
    hb = ax.hexbin(x, y, gridsize=gridsize, bins="log" if log else None,
                   cmap=cmap, extent=extent, mincnt=mincnt)
    if extent is not None:
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])
    cbar = None
    if label is not None:
        cbar = pcol.make_colorbar(ax=ax, im=hb, label=label,
                                  orientation="vertical", set_cax=True)
    return PlotResult(ax=ax, fig=fig, im=hb, colorbar=cbar)


@pst.styled
def plot_heatmap(matrix, ax=None, cmap="Blues", vmin=None, vmax=None,
                 xticklabels=None, yticklabels=None, annotate=None,
                 origin="upper", aspect="equal", label=None, **kwargs):
    """A matrix as coloured cells, optionally with a text in each cell.

    ``annotate`` is a matrix of strings; its text turns white on dark cells.
    A colorbar is drawn when ``label`` is given.
    """
    matrix = np.asarray(matrix, dtype=float)
    ax, fig, kwargs = create_axis(ax, "heatmap", **kwargs)
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, origin=origin,
                   aspect=aspect)
    ax.grid(False)
    if xticklabels is not None:
        ax.set_xticks(range(matrix.shape[1]), xticklabels)
    if yticklabels is not None:
        ax.set_yticks(range(matrix.shape[0]), yticklabels)

    if annotate is not None:
        low, high = im.get_clim()
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                share = (matrix[row, col] - low) / (high - low) if high > low else 0.0
                ax.text(col, row, annotate[row][col], ha="center", va="center",
                        fontsize=pst.LABEL_SIZE,
                        color="white" if share > ANNOTATION_FLIP else pst.INK)
    cbar = None
    if label is not None:
        cbar = pcol.make_colorbar(ax=ax, im=im, label=label,
                                  orientation="vertical", set_cax=True)
    return PlotResult(ax=ax, fig=fig, im=im, colorbar=cbar)


@pst.styled
def plot_histogram(data, bins=30, density=False, histtype="bar", ax=None,
                   labels=None, colors=None, ls=None, lw=1.8, alpha=None,
                   **kwargs):
    """Histograms of one or several samples on shared bins.

    Parameters
    ----------
    data : 1d array or list of 1d arrays; non-finite values are dropped.
    bins : edges or a count; the same edges are used for every sample.
    histtype : ``"bar"`` (filled, default alpha 0.85) or ``"step"`` (outline).
    ls : line style per sample (``step`` only); the last one is reused.
    """
    samples = [data] if np.ndim(data[0]) == 0 else list(data)
    labels = _as_list(labels) or []
    colors = _as_list(colors) or _palette(len(samples))
    ls = _as_list(ls) or ["-"]
    if alpha is None:
        alpha = 0.85 if histtype == "bar" else 1.0

    ax, fig, kwargs = create_axis(ax, "hist", **kwargs)
    if isinstance(bins, (int, np.integer)):
        finite = np.concatenate([np.asarray(s, float)[np.isfinite(s)] for s in samples])
        bins = np.linspace(finite.min(), finite.max(), bins + 1)
    for idx, sample in enumerate(samples):
        values = np.asarray(sample, dtype=float)
        ax.hist(values[np.isfinite(values)], bins=bins, density=density,
                histtype=histtype, color=_pick(colors, idx), alpha=alpha,
                linewidth=lw, linestyle=_pick(ls, idx),
                label=labels[idx] if idx < len(labels) else None)
    _finish_legend(ax, labels, kwargs)
    return PlotResult(ax=ax, fig=fig)
