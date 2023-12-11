from importlib import reload
import os
import matplotlib.pyplot as plt
import geoutils.utils.general_utils as gut

# plotting utils
import geoutils.plotting.plot_settings
import geoutils.plotting.map_plots
import geoutils.plotting.xy_plots
import geoutils.plotting.plotting_utils

reload(geoutils.plotting.plotting_utils)
reload(geoutils.plotting.plot_settings)
reload(geoutils.plotting.map_plots)
reload(geoutils.plotting.xy_plots)

from geoutils.plotting.map_plots import *
from geoutils.plotting.xy_plots import *
from geoutils.plotting.plotting_utils import *
from geoutils.plotting.plot_settings import *


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
        sub_string=extension
    ):
        savepath += f".{extension}"
    if fig is None:
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
        enumerate_subplots(axs, pos_x=-0.1, pos_y=1.07)
    else:
        fig = plt.figure(
            figsize=figsize
        )
        axs = fig.axes

    return {
        "ax": axs,
        "fig": fig,
    }
