import geoutils.utils.spatial_utils as sput
import metpy.calc as metcalc
import numpy as np
from metpy.units import units
from importlib import reload
import xarray as xr
import geoutils.utils.general_utils as gut
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


def compute_windspeed(u, v, ws_name='windspeed'):
    windspeed = np.sqrt(u ** 2 + v ** 2)
    windspeed = windspeed.rename(ws_name)
    gut.myprint(
        f"Computed windspeed of wind dataset as {ws_name}.")
    return windspeed


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
                                           percentage=True,
                                           dequantify=True):
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
    # check for units:
    if temperature.metpy.units != units.K:
        temperature = temperature * units.K
        gut.myprint('Temperature converted to Kelvin')
    if pressure.metpy.units != units.hPa:
        pressure = pressure * units.hPa
        gut.myprint(f'Pressure converted to hPa')
    if specific_humidity.metpy.units != units('kg/kg'):
        specific_humidity = specific_humidity * units('kg/kg')
        gut.myprint(f'Specific humidity converted to kg/kg')

    rh = metcalc.relative_humidity_from_specific_humidity(
        pressure=pressure,
        temperature=temperature,
        specific_humidity=specific_humidity)
    if percentage:
        rh = rh.metpy.convert_units('percent')

    rh = rh.to_dataset(name='rh')

    if dequantify:
        rh = remove_units(rh)

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
    # check for units:
    if temperature.metpy.units != units.K:
        temperature = temperature * units.K
    if pressure.metpy.units != units.hPa:
        pressure = pressure * units.hPa

    pt = metcalc.potential_temperature(
        pressure=pressure,
        temperature=temperature)
    pt = pt.to_dataset(name='pt')

    return pt


def vertical_cross_section_average(data, lon_range, lat_range, av_dim='lon',
                                   av_type='mean'):
    cut_data = sput.cut_map(data, lon_range, lat_range)

    mean_data = sput.horizontal_average(cut_data, av_dim,
                                        average_type=av_type)
    cross_horizonatal_dim = 'lat' if av_dim == 'lon' else 'lon'
    if 'time' in gut.get_dims(mean_data):
        mean_data = mean_data.transpose(
            'time', 'lev', cross_horizonatal_dim).compute()

    return mean_data


def vertical_cross_section(data, lon_range, lat_range,
                           interp_steps=100,
                           interp_type='linear',
                           ):
    """This is a vertical cross section function along the line defined by the
    lon_range and lat_range. The function uses metpy's cross_section function.
    """
    # rename level to isobaric labelling for metpy
    data = gut.rename_dim(data, dim='lev', name='isobaric')
    # data = gut.rename_dim(data, dim='lat', name='latitude')
    # data = gut.rename_dim(data, dim='lon', name='longitude')

    # ATTENTION: metpy expects start as LAT, LON Pairs! (not LON, LAT)
    cross = cross_section(data,
                          (lat_range[0], lon_range[0]),
                          (lat_range[1], lon_range[1]),
                          interp_type=interp_type,
                          steps=interp_steps)
    if isinstance(data, xr.Dataset):
        cross = cross.set_coords(('lat', 'lon'))

    # Bring in the order to plot as (lon/lat) - isobaric plot
    if 'time' in gut.get_dims(cross):
        cross = cross.transpose('time', 'isobaric', 'index').compute()
    else:
        cross = cross.transpose('isobaric', 'index').compute()

    cross = gut.rename_dim(cross, dim='isobaric', name='lev')

    return cross


def geopotential_to_heigth(geopotential):
    """Convert geopotential to heigth.

    Args:
    ----------
    geopotential: xarray.Dataset
        dataset with geopotential

    Returns:
    ----------
    xarray.Dataset
        dataset with heigth
    """
    if not isinstance(geopotential, xr.DataArray):
        raise ValueError('geopotential must be an xarray.Dataset')

    if geopotential.metpy.units != units('m^2/s^2'):
        geopotential = geopotential * units('m^2/s^2')
    height = metcalc.geopotential_to_height(geopotential)
    return height


def remove_units(data):
    """Remove units from an array.

    Args:
    ----------
    data: xarray.DataArray
        data array with units

    Returns:
    ----------
    xarray.DataArray
        data array without units
    """
    return data.metpy.dequantify()
