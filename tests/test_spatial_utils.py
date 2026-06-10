"""Characterization tests for spatial_utils: cut_map and check_dimensions.

These pin the *current* behavior so the step-4 de-duplication (collapsing the
three cut_map / check_dimensions copies) can be verified to preserve behavior.
"""
import numpy as np
import xarray as xr

import geoutils.indices.enso_utils as enso
import geoutils.utils.spatial_utils as sput


def test_cut_map_subregion(latlon_da):
    out = sput.cut_map(latlon_da, lon_range=[-10, 10], lat_range=[-5, 5])
    assert list(out.lon.values) == [-10.0, -5.0, 0.0, 5.0, 10.0]
    assert list(out.lat.values) == [-5.0, 0.0, 5.0]
    assert out.shape == (3, 5)


def test_cut_map_none_ranges_returns_full(latlon_da):
    out = sput.cut_map(latlon_da, lon_range=None, lat_range=None)
    xr.testing.assert_equal(out, latlon_da)


def test_cut_map_handles_descending_lat(latlon_da):
    da = latlon_da.sortby("lat", ascending=False)
    out = sput.cut_map(da, lon_range=[-10, 10], lat_range=[-5, 5])
    assert set(out.lat.values) == {-5.0, 0.0, 5.0}
    assert out.sizes["lon"] == 5


def test_enso_and_sput_cut_map_agree_on_small_region(latlon_da):
    # For ranges <= 180 the two implementations select the same points;
    # this equivalence is what justifies merging enso.cut_map into sput.cut_map.
    region = dict(lon_range=[-10, 10], lat_range=[-5, 5])
    xr.testing.assert_equal(
        sput.cut_map(latlon_da, **region),
        enso.cut_map(latlon_da, **region),
    )


def test_check_dimensions_renames_sorts_and_shifts(unnormalized_ds):
    out = sput.check_dimensions(unnormalized_ds)
    # latitude/longitude renamed to lat/lon
    assert "lon" in out.dims and "lat" in out.dims
    assert "longitude" not in out.dims and "latitude" not in out.dims
    # longitudes shifted into [-180, 180]
    assert float(out.lon.min()) >= -180.0
    assert float(out.lon.max()) <= 180.0
    # both axes sorted ascending
    assert np.all(np.diff(out.lon.values) > 0)
    assert np.all(np.diff(out.lat.values) > 0)
