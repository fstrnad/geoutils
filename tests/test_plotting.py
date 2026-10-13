"""Smoke + characterization tests for the map plotting entry points.

These cover each plot_type branch of plot_array so the dispatch refactor can be
verified to preserve behavior.
"""
import inspect

import numpy as np
import pytest

import geoutils.plotting.maps.fields as mp


def test_plot_map_returns_expected_artifacts(latlon_da):
    res = mp.plot_map(latlon_da)
    # PlotResult keeps dict-style access for back-compat.
    for key in ("ax", "fig", "im"):
        assert key in res
        assert res[key] is not None


@pytest.mark.parametrize("plot_type", ["colormesh", "contourf", "contour"])
def test_plot_map_plot_types(latlon_da, plot_type):
    res = mp.plot_map(latlon_da, plot_type=plot_type)
    assert res["im"] is not None


# --- plot_array directly: one test per draw branch -----------------------

@pytest.mark.parametrize("plot_type", ["colormesh", "contourf", "discrete"])
def test_plot_array_field_types(latlon_da, plot_type):
    res = mp.plot_array(z=latlon_da, plot_type=plot_type)
    assert res["im"] is not None


def test_plot_array_scatter():
    x = np.array([0.0, 1, 2, 3])
    y = np.array([0.0, 1, 0, 1])
    z = np.array([1.0, 2, 3, 4])
    res = mp.plot_array(x=x, y=y, z=z, plot_type="scatter")
    assert res["im"] is not None


def test_plot_array_points():
    res = mp.plot_array(x=np.array([0.0, 1, 2]), y=np.array([0.0, 1, 2]),
                        plot_type="points")
    assert res["ax"] is not None


def test_plot_array_hatch(latlon_da):
    res = mp.plot_array(z=(latlon_da > 20).astype(float), plot_type="hatch")
    assert res["im"] is not None


def test_color_bounds_is_explicit_parameter():
    sig = inspect.signature(mp.plot_array)
    assert "color_bounds" in sig.parameters
    assert sig.parameters["color_bounds"].default == "robust"
    assert inspect.signature(mp.plot_map).parameters["color_bounds"].default == "robust"


# --- significance overlays ---------------------------------------------

def _hatched(ax):
    return [c for c in ax.collections if c.get_hatch()]


def test_significance_mask_hatches_map(latlon_da):
    mask = latlon_da > latlon_da.median()
    res = mp.plot_map(latlon_da, significance_mask=mask)
    assert len(_hatched(res["ax"])) == 1


def test_significance_contour_draws_lines(latlon_da):
    mask = latlon_da > latlon_da.median()
    plain = mp.plot_map(latlon_da)
    res = mp.plot_map(latlon_da, significance_mask=mask, sig_plot_type="contour")
    assert len(res["ax"].collections) > len(plain["ax"].collections)


def test_significance_mask_false_rejected(latlon_da):
    with pytest.raises(ValueError, match="not False"):
        mp.plot_map(latlon_da, significance_mask=False)


def test_significance_mask_shape_mismatch_rejected(latlon_da):
    with pytest.raises(ValueError, match="same dimension"):
        mp.plot_map(latlon_da, significance_mask=latlon_da[:2])


# --- explicit options and unknown-keyword warning ----------------------

def test_plot_array_returns_colorbar_when_labelled(latlon_da):
    res = mp.plot_map(latlon_da, label="K")
    assert res["colorbar"] is not None
    assert res["colorbar"].ax.get_xlabel() == "K"


def test_plot_array_no_colorbar_without_label(latlon_da):
    assert mp.plot_map(latlon_da)["colorbar"] is None


def test_unknown_keyword_warns_and_names_it(latlon_da):
    with pytest.warns(FutureWarning, match="not_an_option"):
        mp.plot_map(latlon_da, not_an_option=1)


def test_documented_keywords_do_not_warn(latlon_da, recwarn):
    mp.plot_map(latlon_da, label="K", orientation="vertical", tick_step=1,
                levels=6, extend="both", centercolor="white", round_dec=1,
                title="t", set_map=True, land_ocean=True, hatch_type="//",
                y_title=1.1, fontsize=10)
    assert not [w for w in recwarn if w.category is FutureWarning]


def test_axis_options_documented_for_plain_arrays(recwarn):
    mp.plot_array(np.arange(12.0).reshape(3, 4), xlabel="x", ylabel="y",
                  xlim=(0, 3), levels=4, plot_type="contourf")
    assert not [w for w in recwarn if w.category is FutureWarning]


@pytest.mark.parametrize("module, names", [
    ("axes", ("TITLE_OPTIONS",)),
    ("maps.basemap", ("BASEMAP_OPTIONS",)),
])
def test_option_registries_match_source(module, names):
    import importlib
    import inspect
    mod = importlib.import_module(f"geoutils.plotting.{module}")
    src = inspect.getsource(mod)
    for name in names:
        for key in getattr(mod, name):
            assert f"'{key}'" in src or f'"{key}"' in src, (name, key)
