from tracemalloc import start
import geoutils.utils.time_utils as tu
import pandas as pd
import geoutils.utils.spatial_utils as sput
import numpy as np
from importlib import reload
import xarray as xr
import geoutils.utils.general_utils as gut
reload(gut)

JULIAN_YEAR_LENGTH_IN_DAYS = 365.25

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


def compute_windspeed(u, v, ws_name='windspeed', verbose=False):
    windspeed = np.sqrt(u ** 2 + v ** 2)
    windspeed = windspeed.rename(ws_name)
    gut.myprint(
        f"Computed windspeed of wind dataset as {ws_name}.", verbose=verbose)
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
    import metpy.calc as metcalc
    from metpy.units import units

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
    import metpy.calc as metcalc
    from metpy.units import units

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
    from metpy.interpolate import cross_section

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
    import metpy.calc as metcalc
    from metpy.units import units

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


def compute_toa_solar_radiation(grid_step=1, start_date="2023-01-01", end_date="2023-12-31",
                                times=None, longitudes=None, latitudes=None,
                                unit="Watts",
                                xarray_dataset=None):
    """
    Computes the top-of-atmosphere (TOA) incident solar radiation for a global grid
    with a user-defined resolution, a custom date range, and hourly time resolution.

    Args:
        grid_step (float): Grid resolution in degrees (e.g., 2, 1, 0.5).
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        xarray Dataset containing TOA solar radiation [W/m²] for each latitude, longitude, and hour.
    """
    # Constants
    if xarray_dataset is not None:
        latitudes = xarray_dataset.lat
        longitudes = xarray_dataset.lon
        times = xarray_dataset.time

    S0 = 1361  # Solar constant (W/m²)

    # Create latitude and longitude arrays with the given grid step
    latitudes = np.arange(-90, 90 + grid_step,
                          grid_step) if latitudes is None else latitudes
    # No +grid_step to avoid duplicate 180° point
    longitudes = np.arange(-180, 180,
                           grid_step) if longitudes is None else longitudes

    # Generate time range (hourly timestamps)
    times = tu.get_dates_in_range(start_date=start_date, end_date=end_date,
                                  freq="h") if times is None else times

    # Extract day of the year and hour for each timestamp
    days_of_year = times.dt.dayofyear.data
    hours_of_day = times.dt.hour.data

    # # Generate time range (hourly timestamps)
    # times = pd.date_range(start=start_date, end=end_date, freq="h")

    # # Extract day of the year and hour for each timestamp
    # days_of_year = times.dayofyear
    # hours_of_day = times.hour

    # Convert latitude and longitude to radians
    lat_radians = np.radians(latitudes)

    # Compute solar declination angle δ (radians) for each day of the year
    declination = np.radians(
        23.44 * np.sin((2 * np.pi * (days_of_year + 10)) / 365))

    # Compute Earth-Sun distance correction factor (r0/r)^2
    distance_factor = (1 + 0.033 * np.cos(2 * np.pi * days_of_year / 365)) ** 2

    # Initialize TOA radiation array
    toa_radiation = np.zeros((len(times), len(latitudes), len(longitudes)))

    # Compute TOA solar radiation for each grid point
    for i, lat in enumerate(lat_radians):
        for t, (day, hour) in enumerate(zip(days_of_year, hours_of_day)):
            # Hour angle (ω) in radians: ω = (hour - 12) * 15° (converted to radians)
            hour_angle = np.radians((hour - 12) * 15)

            # Compute cosine of the solar zenith angle (θz)
            cos_theta_z = (np.sin(lat) * np.sin(declination[t])) + (
                np.cos(lat) * np.cos(declination[t]) * np.cos(hour_angle))

            # TOA radiation: If the Sun is below the horizon (cos_theta_z < 0), set to 0
            toa_radiation[t, i, :] = np.maximum(
                S0 * distance_factor[t] * cos_theta_z, 0)

    if unit == 'Joule':
        # Convert to J/m²
        toa_radiation = toa_radiation * 3600
        units = 'J/m²'
    else:
        units = 'W/m²'

    # Convert to xarray Dataset
    ds = xr.DataArray(
        data=toa_radiation,
        dims=["time", "lat", "lon"],
        coords={
            "time": times,
            "lat": latitudes,
            "lon": longitudes
        },
        attrs={
            "units": f"{units}",
            "description": f"TOA solar radiation with {grid_step}° resolution from {start_date} to {end_date}, hourly"
        }
    )

    return ds
