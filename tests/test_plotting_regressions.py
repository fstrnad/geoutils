"""Regression tests for bugs found in the plotting review (2026-10-13)."""
import matplotlib as mpl
import numpy as np
import pytest

import geoutils.plotting.plot_settings as pst
import geoutils.plotting.xy_plots as xyp


@pytest.fixture
def xs():
    return np.linspace(0, 10, 20)


def test_plot_2d_density_runs(xs):
    res = xyp.plot_2d(y=xs, x=xs**2, plot_type="density")
    assert res["ax"] is not None


def test_plot_2d_scatter_with_z_arr_runs(xs):
    res = xyp.plot_2d(y=xs, x=xs, z_arr=[xs], plot_type="scatter")
    assert res["im"] is not None


def test_plot_2d_default_lists_not_shared():
    defaults = xyp.plot_2d.__kwdefaults__ or {}
    defaults = list(xyp.plot_2d.__defaults__ or ()) + list(defaults.values())
    assert not any(isinstance(d, list) for d in defaults)


def test_plot_2d_more_series_than_colors(xs):
    n = len(pst.colors) + 3
    res = xyp.plot_2d(y=[xs + i for i in range(n)])
    lines = res["ax"].lines
    assert len(lines) == n
    assert all(a.get_color() != b.get_color() for a, b in zip(lines, lines[1:]))


def test_plot_2d_lcmap_colorbar_uses_given_cmap(xs):
    res = xyp.plot_2d(y=[xs, xs + 1, xs + 2], lcmap="plasma")
    cbar_ax = [a for a in res["fig"].axes if a is not res["ax"]][0]
    assert "plasma" in {c.cmap.name for c in cbar_ax.collections}


def test_plot_2d_marker_size_kwarg_used(xs):
    res = xyp.plot_2d(y=xs, mk_arr=["o"], marker_size=9)
    assert res["ax"].lines[0].get_markersize() == 9


# --- maps ---------------------------------------------------------------

def test_create_map_rem_frame():
    import geoutils.plotting.map_plots as mp
    ax = mp.create_map(rem_frame=True)["ax"]
    assert ax.spines["geo"].get_visible() is False


def test_plot_map_accepts_numpy_scalar_limits(latlon_da):
    import geoutils.plotting.map_plots as mp
    res = mp.plot_map(latlon_da, vmin=np.float32(0), vmax=np.float64(5))
    assert res["im"] is not None


def test_plot_map_all_nan_raises_valueerror(latlon_da):
    import geoutils.plotting.map_plots as mp
    with pytest.raises(ValueError, match="all-NaN"):
        mp.plot_map(latlon_da * np.nan)


def test_plot_map_constant_array_draws(latlon_da):
    import geoutils.plotting.map_plots as mp
    res = mp.plot_map(latlon_da * 0 + 3)
    assert res["im"] is not None
