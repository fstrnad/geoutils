"""Shared helpers for the example scripts: locate and load the ERA5 files.

The data directory can be overridden with the ``ERA5_DATA`` environment
variable; it must contain ``<variable>/<resolution>/europe/hourly`` folders.
"""
import os
from pathlib import Path

import xarray as xr

DATA = Path(os.environ.get(
    "ERA5_DATA", "/Users/fstrnad/projects/era5/data/era5-single-levels"))
FIGURES = Path(__file__).parent / "figures"

# short name -> (folder, unit conversion, new unit)
VARIABLES = {
    "t2m": ("2m_temperature", lambda da: da - 273.15, "°C"),
    "tp": ("total_precipitation", lambda da: da * 1000, "mm"),
}
LONG_NAMES = {"t2m": "2 m temperature", "tp": "Hourly precipitation"}


def months(var, resolution="0.25deg"):
    """Months (1-12) of 2024 that are completely downloaded.

    A file counts once its ``.sig.json`` marker exists, so a download that is
    still running never hands us a half-written file.
    """
    folder = DATA / VARIABLES[var][0] / resolution / "europe" / "hourly"
    return sorted(int(p.stem[-2:]) for p in folder.glob("*_2024-??.nc")
                  if p.with_name(p.name + ".sig.json").exists())


def load(var, month=None, resolution="0.25deg"):
    """Hourly data as a lazy ``time, lat, lon`` DataArray in °C or mm.

    ``month`` is an int, a list of ints, or None for every month available.
    """
    folder, convert, unit = VARIABLES[var]
    month = months(var, resolution) if month is None else month
    month = [month] if isinstance(month, int) else month
    files = [DATA / folder / resolution / "europe" / "hourly"
             / f"{folder}_2024-{m:02d}.nc" for m in month]
    ds = xr.open_mfdataset(files, chunks={})
    da = ds[var].rename(valid_time="time", latitude="lat", longitude="lon")
    da = convert(da.sortby("lat"))
    da.attrs["units"] = unit
    da.name = LONG_NAMES[var]
    return da


def label(da):
    return f"{da.name} [{da.attrs['units']}]"


CITIES = {  # name: (lat, lon)
    "Oslo": (59.9, 10.75), "London": (51.5, -0.1), "Berlin": (52.5, 13.4),
    "Madrid": (40.4, -3.7), "Rome": (41.9, 12.5), "Athens": (38.0, 23.7),
}


def at_city(da, name):
    """Nearest grid point of a city, as a time series."""
    lat, lon = CITIES[name]
    return da.sel(lat=lat, lon=lon, method="nearest")
