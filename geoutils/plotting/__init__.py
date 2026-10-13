"""Plotting utilities for maps, time series, and 2D arrays.

This is the curated public API. Import the names you need directly, e.g.::

    from geoutils.plotting import plot_map, plot_2d, create_plot, PlotResult

The legacy aggregator ``geoutils.plotting.plots`` (commonly imported ``as cplt``)
still works as a deprecated shim that re-exports the same functions. ``geoutils.plotting.videos`` is
intentionally not imported here because it needs the optional ``video`` extra
(imageio, opencv); import it explicitly if you need it.
"""
from geoutils.plotting.result import PlotResult

# Figure creation and saving
from geoutils.plotting.figure import (
    create_plot,
    create_twinaxis,
    enumerate_subplots,
    mk_plot_dir,
    save_fig,
)

# Maps
from geoutils.plotting.figure import create_multi_plot
from geoutils.plotting.maps.projections import get_projection
from geoutils.plotting.maps.basemap import (
    create_map,
    scale_bar,
    set_extent,
    set_grid,
)
from geoutils.plotting.maps.fields import plot_array, plot_map
from geoutils.plotting.maps.overlays import (
    plot_edges,
    plot_horizontal_line_at_latitude,
    plot_rectangle,
    plot_ring,
    plot_trajectory,
    plot_wind_field,
)

# x-y / time-series plots
from geoutils.plotting.xy import (
    fill_between,
    plot_2d,
    plot_hist,
    plot_null_model,
)

# Colorbars, colormaps, axis helpers, annotations
from geoutils.plotting.colors import (
    add_colorbar,
    create_cmap,
    get_cmap,
    make_colorbar,
    make_colorbar_discrete,
)
from geoutils.plotting.axes import (
    plot_arrow,
    plot_hline,
    plot_line,
    plot_text,
    plot_vline,
    prepare_axis,
    set_legend,
    set_title,
    text_box,
)

# Global font / size settings
from geoutils.plotting.style import (
    change_fsize,
    list_available_fonts,
    list_available_styles,
    reset_fsize,
)

__all__ = [
    "PlotResult",
    # figure
    "create_plot", "create_twinaxis", "mk_plot_dir", "save_fig",
    # maps
    "create_map", "create_multi_plot", "get_projection", "plot_array",
    "plot_edges", "plot_horizontal_line_at_latitude", "plot_map",
    "plot_rectangle", "plot_ring", "plot_trajectory", "plot_wind_field",
    "scale_bar", "set_extent", "set_grid",
    # xy
    "fill_between", "plot_2d", "plot_hist", "plot_null_model",
    # colorbars / cmaps / axes / annotations
    "add_colorbar", "create_cmap", "enumerate_subplots", "get_cmap",
    "make_colorbar", "make_colorbar_discrete", "plot_arrow", "plot_hline",
    "plot_line", "plot_text", "plot_vline", "prepare_axis", "set_legend",
    "set_title", "text_box",
    # settings
    "change_fsize", "list_available_fonts", "list_available_styles",
    "reset_fsize",
]
