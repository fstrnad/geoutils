import geoutils.geodata.multilevel_pressure as mp
import metpy.calc as metcalc
from metpy.units import units
from importlib import reload
import numpy as np
import xarray as xr
import geoutils.utils.general_utils as gut
import geoutils.utils.file_utils as fut
from metpy.interpolate import cross_section
reload(gut)

def parse_cf(ds):
    """Parse the coordinates of a dataset.

    Args:
    ----------
    ds: xarray.Dataset
        dataset to parse

    Returns:
    ----------
    xarray.Dataset
        parsed dataset
    """
    if not isinstance(ds, xr.Dataset):
        raise ValueError('ds must be an xarray.Dataset')
    return ds.metpy.parse_cf().squeeze()


def kelvin_to_degC(temperature):
    """Convert temperature to degC.

    Args:
    ----------
    temperature: xarray.DataArray
        temperature

    Returns:
    ----------
    xarray.DataArray
        temperature in degC
    """
    return temperature - 273.15


def degC_to_kelvin(temperature):
    """Convert temperature to Kelvin.

    Args:
    ----------
    temperature: xarray.DataArray
        temperature

    Returns:
    ----------
    xarray.DataArray
        temperature in Kelvin
    """
    return temperature + 273.15


def specific_humidity_to_relative_humidity(specific_humidity,
                                           temperature,
                                           pressure,
                                           percentage=True):
    """Convert specific humidity to relative humidity.

    Args:
    ----------
    q: xarray.DataArray
        specific humidity in kg/kg
    t: xarray.DataArray
        temperature in Kelvin
    p: xarray.DataArray
        pressure in hPa

    Returns:
    ----------
    xarray.DataArray
        relative humidity
    """
    rh = metcalc.relative_humidity_from_specific_humidity(
        pressure=pressure * units.hPa,
        temperature=temperature*units.K,
        specific_humidity=specific_humidity)
    if percentage:
        return rh.metpy.convert_units('percent')
    else:
        return rh


def potential_temperature(temperature, pressure):
    """Calculate potential temperature.

    Args:
    ----------
    temperature: xarray.DataArray
        temperature in Kelvin
    pressure: xarray.DataArray
        pressure in hPa

    Returns:
    ----------
    xarray.DataArray
        potential temperature in Kelvin
    """
    return metcalc.potential_temperature(
        pressure=pressure * units.hPa,
        temperature=temperature * units.K)


def vertical_cross_section(data, lon_range, lat_range,
                           interp_steps=100,
                           interp_type='linear',
                           set_coords=True):
    # rename level to isobaric labelling for metpy
    # data = gut.rename_dim(data, dim='lev', name='isobaric')
    # data = gut.rename_dim(data, dim='lat', name='latitude')
    # data = gut.rename_dim(data, dim='lon', name='longitude')

    cross = cross_section(data,
                          (lon_range[0], lat_range[0]),
                          (lon_range[1], lat_range[1]),
                          interp_type=interp_type,
                          steps=interp_steps)
    return cross
