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
    assert sig.parameters["color_bounds"].default == "quantile"
