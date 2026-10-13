"""Shared helpers for the example scripts."""
import sys
from pathlib import Path

import xarray as xr

DEFAULT_DATA = Path("/Users/fstrnad/projects/era5/data/era5_test.nc")
FIGURES = Path(__file__).parent / "figures"


def load_t2m(path=None):
    """2 m temperature in degC as a lat/lon DataArray (geoutils' coordinate names).

    The data path can be given as the first command line argument.
    """
    path = Path(path or (sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA))
    da = xr.open_dataset(path)["t2m"].squeeze(drop=True)
    da = da.rename(latitude="lat", longitude="lon") - 273.15
    da.attrs["units"] = "°C"
    da.name = "2 m temperature"
    return da.sortby("lat")
