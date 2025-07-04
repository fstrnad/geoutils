from importlib import reload
import os
import matplotlib.pyplot as plt
import geoutils.utils.general_utils as gut


# plotting utils
import geoutils.plotting.plot_settings
import geoutils.plotting.map_plots
import geoutils.plotting.xy_plots
import geoutils.plotting.plotting_utils

reload(geoutils.plotting.plot_settings)
reload(geoutils.plotting.plotting_utils)
reload(geoutils.plotting.map_plots)
reload(geoutils.plotting.xy_plots)

from geoutils.plotting.plot_settings import *
from geoutils.plotting.map_plots import *
from geoutils.plotting.xy_plots import *
from geoutils.plotting.plotting_utils import *


def create_twinaxis(ax):
    ax2 = ax.twinx()
    return ax2


def mk_plot_dir(savepath):
    if os.path.exists(savepath):
        return
    else:
        dirname = os.path.dirname(savepath)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        return


def save_fig(savepath, fig=None,
             extension='png',
             dpi='figure'):
    mk_plot_dir(savepath)

    if not gut.check_contains_substring(
        main_string=savepath,
        sub_string=['png', 'pdf', 'jpg', 'jpeg', 'svg']
    ):
        savepath += f".{extension}"

    gut.myprint(f'Saving figure to {savepath}')
    if fig is None:
        if extension == 'png':
            dpi = 300
        plt.savefig(savepath,
                    bbox_inches='tight',
                    dpi=dpi)
    else:
        fig.savefig(savepath,
                    bbox_inches='tight',
                    dpi=dpi)


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
        fig = plt.figure(
            figsize=figsize
        )
        axs = fig.axes

    return {
        "ax": axs,
        "fig": fig,
    }
