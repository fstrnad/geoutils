"""Figure creation, subplot labelling and saving."""
import string
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import geoutils.plotting.style as pst
import geoutils.utils.general_utils as gut
from geoutils.plotting.axes import plot_text, set_title
from geoutils.plotting.result import PlotResult
from geoutils.plotting.maps.basemap import create_map
from geoutils.plotting.maps.projections import get_projection

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


def create_multi_plot(nrows, ncols, projection=None,
                      lon_range=None, lat_range=None,
                      plot_grid=False,
                      verbose=False, **kwargs):
    figsize = kwargs.pop('figsize', None)
    if figsize is None:
        figsize = (4*ncols, 3*nrows)

    end_idx = kwargs.pop('end_idx', None)
    end_idx = int(nrows*ncols) if end_idx is None else end_idx
    rem_idx = kwargs.pop('rem_idx', None)
    full_length_row = kwargs.pop('full_length_row', False)
    map_axis = kwargs.pop('map_axis', [])
    if len(map_axis) == 0:
        map_axis = np.arange(nrows*ncols)
    if nrows*ncols/end_idx >= nrows:
        nrows = nrows - 1 if nrows > 1 else nrows

    width_ratios = kwargs.pop('width_ratios', np.ones(ncols))
    height_ratios = kwargs.pop('height_ratios', np.ones(nrows))
    gs_rows = nrows
    gs_cols = ncols
    if len(width_ratios) != ncols:
        raise ValueError(
            f'Length of width_ratios {len(width_ratios)} does not match number of columns {ncols}!')
    if len(height_ratios) != nrows:
        raise ValueError(
            f'Length of height_ratios {len(height_ratios)} does not match number of rows {nrows}!')
    fig = plt.figure(figsize=(figsize[0], figsize[1]))

    hspace = kwargs.pop('hspace', 0.)
    wspace = kwargs.pop('wspace', 0.2)

    gs = fig.add_gridspec(gs_rows, gs_cols,
                          height_ratios=height_ratios,
                          width_ratios=width_ratios,
                          hspace=hspace, wspace=wspace)
    proj_arr = kwargs.pop('proj_arr', None)
    if proj_arr is None:
        proj_arr = kwargs.pop('projection_arr', None)

    diff_projs = False

    if projection is not None or proj_arr is not None:
        central_longitude = kwargs.pop("central_longitude", None)
        central_latitude = kwargs.pop("central_latitude", None)
        dateline = kwargs.pop('dateline', False)
        dateline_arr = kwargs.pop('dateline_arr', None)
        if proj_arr is None:
            proj_arr = gut.replicate_object(projection, nrows*ncols)
        else:
            diff_projs = True
            if len(proj_arr) > end_idx:
                raise ValueError(
                    f'Length of projection array {len(proj_arr)} does not match number of pannels {end_idx}!')
    else:
        proj_arr = gut.replicate_object(None, nrows*ncols)
    axs = []

    set_map = kwargs.pop('set_map', False)  # is set later
    run_idx = 0
    for i in range(nrows):
        for j in range(ncols):
            if gut.has_non_none_objects(proj_arr):
                if dateline_arr is not None:
                    dateline = dateline_arr[run_idx]
                projection = proj_arr[run_idx]
                if projection is not None:
                    proj = get_projection(projection=projection,
                                          central_longitude=central_longitude,
                                          central_latitude=central_latitude,
                                          dateline=dateline)

                    if run_idx == end_idx-1 and full_length_row:
                        end_row_idx = kwargs.pop('end_row_idx', ncols)
                        axs.append(fig.add_subplot(gs[i, j:end_row_idx],
                                                   projection=proj,
                                                   ))
                    else:
                        axs.append(fig.add_subplot(gs[i, j],
                                                   projection=proj))
                    if run_idx in map_axis:
                        map_dict = create_map(
                            ax=axs[run_idx],
                            projection=projection,
                            central_longitude=central_longitude,
                            plot_grid=plot_grid,
                            lon_range=lon_range,
                            lat_range=lat_range,
                            dateline=dateline,
                            set_map=set_map,
                            **kwargs
                        )
                        kwargs = map_dict['kwargs']
                else:
                    # Make a row at full length until the end of the columns
                    if run_idx == end_idx-1 and full_length_row:
                        end_row_idx = kwargs.pop('end_row_idx', ncols)
                        axs.append(fig.add_subplot(gs[i, j:end_row_idx]))
                    else:
                        axs.append(fig.add_subplot(gs[i, j]))
            else:
                axs.append(fig.add_subplot(gs[i, j]))

            run_idx += 1
            if run_idx >= end_idx:
                break

    if rem_idx is not None:
        if not isinstance(rem_idx, list):
            raise ValueError(
                f'rem_idx has to be of type list but is of type {type(rem_idx)}!')
        for idx in rem_idx:
            gut.myprint(f'WARNING! Remove axis {idx}!', verbose=verbose)
            if idx < len(axs):
                axs[idx].axis('off')
        enum_axis = gut.remove_elements_by_indices(axs, rem_idx)
    else:
        enum_axis = axs
    if nrows > 1 or ncols > 1:
        label_panels = kwargs.pop('enumerate_subplots', True)
        if label_panels:
            pos_x = kwargs.pop('pos_x', -0.1)
            pos_y = kwargs.pop('pos_y', 1.1)
            if diff_projs and isinstance(pos_x, (int, float)):
                pos_x_proj = kwargs.pop('pos_x_proj', pos_x)
                pos_x = gut.replicate_object(pos_x, end_idx)
                indices_proj = gut.get_not_None_indices(proj_arr)
                pos_x[indices_proj] = pos_x_proj
            enumerate_subplots(axs=enum_axis, pos_x=pos_x, pos_y=pos_y,
                                   )
    else:
        axs = [axs[0]]

    y_title_pos = 1.1 - 0.05*(nrows-1)
    title = kwargs.pop('title', None)
    y_title = kwargs.pop('y_title', y_title_pos)
    if title is not None:
        set_title(title=title, ax=None, fig=fig,
                  y_title=y_title)

    return PlotResult(ax=axs, fig=fig, projection=projection)
