"""Regression tests for bugs found in the plotting review (2026-10-13)."""
import matplotlib as mpl
import numpy as np
import pytest

import geoutils.plotting.style as pst
import geoutils.plotting.xy as xyp


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
    import geoutils.plotting.maps.fields as mp
    ax = mp.create_map(rem_frame=True)["ax"]
    assert ax.spines["geo"].get_visible() is False


def test_plot_map_accepts_numpy_scalar_limits(latlon_da):
    import geoutils.plotting.maps.fields as mp
    res = mp.plot_map(latlon_da, vmin=np.float32(0), vmax=np.float64(5))
    assert res["im"] is not None


def test_plot_map_all_nan_raises_valueerror(latlon_da):
    import geoutils.plotting.maps.fields as mp
    with pytest.raises(ValueError, match="all-NaN"):
        mp.plot_map(latlon_da * np.nan)


def test_plot_map_constant_array_draws(latlon_da):
    import geoutils.plotting.maps.fields as mp
    res = mp.plot_map(latlon_da * 0 + 3)
    assert res["im"] is not None


# --- figure helpers -----------------------------------------------------

def test_save_fig_bare_filename(tmp_path, monkeypatch):
    import geoutils.plotting as cplt
    monkeypatch.chdir(tmp_path)
    cplt.create_plot()
    cplt.save_fig("plain_name")
    assert (tmp_path / "plain_name.png").stat().st_size > 0


@pytest.mark.parametrize("name, expected", [
    ("png_plots/out", "png_plots/out.png"),
    ("figs/out.pdf", "figs/out.pdf"),
])
def test_save_fig_adds_extension_only_when_missing(tmp_path, name, expected):
    import geoutils.plotting as cplt
    res = cplt.create_plot()
    path = cplt.save_fig(tmp_path / name, fig=res["fig"])
    assert path == tmp_path / expected
    assert path.stat().st_size > 0


def test_create_plot_single_returns_axes():
    import geoutils.plotting as cplt
    res = cplt.create_plot()
    assert isinstance(res["ax"], mpl.axes.Axes)


def test_create_plot_multi_returns_flat_array():
    import geoutils.plotting as cplt
    res = cplt.create_plot(2, 2)
    assert np.shape(res["ax"]) == (4,)


# --- axes and colormaps -------------------------------------------------

def test_yscale_symlog_applied(xs):
    res = xyp.plot_2d(y=xs - 5, yscale="symlog")
    assert res["ax"].get_yscale() == "symlog"


def test_ysymlog_flag_applied(xs):
    res = xyp.plot_2d(y=xs - 5, ysymlog=True)
    assert res["ax"].get_yscale() == "symlog"


def test_ylog_and_ysymlog_conflict_raises(xs):
    with pytest.raises(ValueError, match="ylog and ysymlog"):
        xyp.plot_2d(y=xs + 1, ylog=True, ysymlog=True)


def test_create_cmap_accepts_colormap_object():
    import geoutils.plotting.colors as put
    cmap, norm = put.create_cmap(mpl.colormaps["viridis"], levels=[0, 1, 2])
    assert isinstance(norm, mpl.colors.BoundaryNorm)


def test_create_cmap_rejects_other_types():
    import geoutils.plotting.colors as put
    with pytest.raises(ValueError, match="cmap has to be"):
        put.create_cmap(42, levels=[0, 1, 2])


def test_plot_rectangle_with_text():
    import geoutils.plotting as cplt
    ax = cplt.create_map()["ax"]
    cplt.plot_rectangle(ax, lon_range=(0, 10), lat_range=(0, 10), text="box")
    assert "box" in [t.get_text() for t in ax.texts]
