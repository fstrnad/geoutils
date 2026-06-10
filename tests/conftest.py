"""Shared fixtures for the geoutils test suite."""
import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

# Headless backend so plotting tests never try to open a window.
matplotlib.use("Agg")


@pytest.fixture
def latlon_da():
    """2D lat-lon DataArray with ascending coords in [-180, 180]."""
    lat = np.arange(-10.0, 11.0, 5.0)        # -10..10  (5 pts)
    lon = np.arange(-20.0, 21.0, 5.0)        # -20..20  (9 pts)
    data = np.arange(lat.size * lon.size, dtype=float).reshape(lat.size, lon.size)
    return xr.DataArray(data, coords={"lat": lat, "lon": lon},
                        dims=["lat", "lon"], name="var")


@pytest.fixture
def spacetime_da():
    """3D time-lat-lon DataArray with deterministic random data."""
    time = pd.date_range("2000-01-01", periods=12, freq="D")
    lat = np.arange(-10.0, 11.0, 5.0)
    lon = np.arange(-20.0, 21.0, 5.0)
    rng = np.random.default_rng(0)
    data = rng.standard_normal((time.size, lat.size, lon.size))
    return xr.DataArray(data, coords={"time": time, "lat": lat, "lon": lon},
                        dims=["time", "lat", "lon"], name="var")


@pytest.fixture
def unnormalized_ds():
    """Dataset with non-standard dim names, descending lat, lon in [0, 360)."""
    lon = np.arange(0.0, 360.0, 30.0)         # 0..330 (12 pts, some > 180)
    lat = np.arange(80.0, -81.0, -40.0)       # 80,40,0,-40,-80 (descending)
    data = np.arange(lat.size * lon.size, dtype=float).reshape(lat.size, lon.size)
    return xr.Dataset(
        {"var": (["latitude", "longitude"], data)},
        coords={"latitude": lat, "longitude": lon},
    )
