"""Figure creation, subplot labelling and saving."""
import string
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import geoutils.plotting.style as pst
import geoutils.utils.general_utils as gut
from geoutils.plotting.axes import plot_text
from geoutils.plotting.result import PlotResult

_IMAGE_SUFFIXES = {'.png', '.pdf', '.jpg', '.jpeg', '.svg'}


def create_twinaxis(ax):
    ax2 = ax.twinx()
    return ax2


def mk_plot_dir(savepath):
    Path(savepath).parent.mkdir(parents=True, exist_ok=True)


def save_fig(savepath, fig=None,
             extension='png',
             dpi='figure'):
    savepath = Path(savepath)
    if savepath.suffix.lower() not in _IMAGE_SUFFIXES:
        savepath = savepath.with_name(f"{savepath.name}.{extension}")
    mk_plot_dir(savepath)

    if savepath.suffix.lower() == '.png' and dpi == 'figure':
        dpi = 300
    gut.myprint(f'Saving figure to {savepath}')
    fig = plt.gcf() if fig is None else fig
    fig.savefig(savepath, bbox_inches='tight', dpi=dpi)
    return savepath


def create_plot(nrows=1, ncols=1, **kwargs):

    figsize = kwargs.pop("figsize", None)
    if figsize is None:
        figsize = (6 * ncols, 4 * nrows)

    if nrows > 1 or ncols > 1:
        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            constrained_layout=True,
            ** kwargs,
        )
        axs = axs.flatten()
        enumerate_subplots(axs)
    else:
        fig, axs = plt.subplots(figsize=figsize, **kwargs)

    return PlotResult(ax=axs, fig=fig)


def enumerate_subplots(axs, pos_x=-0.12,
                       pos_y=1.1,
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
