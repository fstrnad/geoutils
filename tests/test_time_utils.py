"""Characterization tests for time_utils.get_mean_time_series."""
import numpy as np


import geoutils.utils.time_utils as tu


def test_get_mean_time_series_shape_and_values(spacetime_da):
    ts_mean, ts_std = tu.get_mean_time_series(
        spacetime_da, lon_range=[-10, 10], lat_range=[-5, 5])
    assert ts_mean.dims == ("time",)
    assert ts_mean.sizes["time"] == spacetime_da.sizes["time"]
    expected = spacetime_da.sel(
        lon=slice(-10, 10), lat=slice(-5, 5)).mean(dim=("lon", "lat"))
    np.testing.assert_allclose(ts_mean.values, expected.values)


def test_get_mean_time_series_quantile(spacetime_da):
    ts_q, _ = tu.get_mean_time_series(
        spacetime_da, lon_range=[-10, 10], lat_range=[-5, 5], q=0.5)
    assert ts_q.sizes["time"] == spacetime_da.sizes["time"]


def test_get_mean_time_series_time_roll(spacetime_da):
    ts_mean, _ = tu.get_mean_time_series(
        spacetime_da, lon_range=[-10, 10], lat_range=[-5, 5], time_roll=3)
    assert ts_mean.sizes["time"] == spacetime_da.sizes["time"]
