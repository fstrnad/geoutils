from pathlib import Path
import matplotlib.pyplot as plt
import geoutils.utils.general_utils as gut

# plotting utils
import geoutils.plotting.plot_settings
import geoutils.plotting.map_plots
import geoutils.plotting.xy_plots
import geoutils.plotting.plotting_utils
from geoutils.plotting.plot_result import PlotResult


from geoutils.plotting.plot_settings import *
from geoutils.plotting.map_plots import *
from geoutils.plotting.xy_plots import *
from geoutils.plotting.plotting_utils import *

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
