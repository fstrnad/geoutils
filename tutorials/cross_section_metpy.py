# %%
# Based on https://unidata.github.io/MetPy/latest/examples/cross_section.html#sphx-glr-examples-cross-section-py
"""
======================
Cross Section Analysis
======================

The MetPy function `metpy.interpolate.cross_section` can obtain a cross-sectional slice through
gridded data.
"""

import geoutils.utils.general_utils as gut
from importlib import reload
import geoutils.plotting.plots as cplt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import geoutils.utils.met_utils as mut
from metpy.units import units
import geoutils.utils.time_utils as tu
import metpy.calc as mpcalc
from metpy.cbook import get_test_data
from metpy.interpolate import cross_section

# %%
# This example uses [NARR reanalysis data](
# https://www.ncei.noaa.gov/products/weather-climate-models/north-american-regional)
# for 18 UTC 04 April 1987 from NCEI.
date = tu.str2datetime('1987-04-04')

# We use MetPy's CF parsing to get the data ready for use, and squeeze down the size-one time
# dimension.

data = xr.open_dataset(get_test_data('narr_example.nc', False))

data = data.metpy.parse_cf().squeeze()
##############################
# %% Define start and end points:

start = (37.0, -105.0)
end = (35.5, -65.0)

cross = cross_section(data, start, end).set_coords(('lat', 'lon'))
cross['Potential_temperature'] = mut.potential_temperature(
    pressure=cross['isobaric'],
    temperature=cross['Temperature']
)['pt']
cross['Relative_humidity'] = mut.specific_humidity_to_relative_humidity(
    pressure=cross['isobaric'],
    temperature=cross['Temperature'],
    specific_humidity=cross['Specific_humidity'],
    percentage=False
)['rh']

cross['t_wind'], cross['n_wind'] = mpcalc.cross_section_components(
    cross['u_wind'],
    cross['v_wind']
)


##############################
# %% For this example, we will be plotting potential temperature, relative humidity, and tangential/normal winds.
reload(cplt)

yticks = np.arange(1000, 50, -100)
im = cplt.plot_2D(x=cross['lon'],
                  y=cross['isobaric'],
                  z=cross['Relative_humidity'],
                  levels=20,
                  cmap='YlGnBu',
                  label='Relative Humidity (dimensionless)',
                  orientation='vertical',
                  vmin=0, vmax=1,
                  xlabel='Longitude [°]',
                  ylabel='Pressure (hPa)',
                  flip_y=True,
                  ysymlog=True,
                  yticks=yticks,
                  title=f'NARR Cross-Section \u2013 {start} to {end} \u2013 '
                  )

im = cplt.plot_2D(x=cross['lon'],
                  y=cross['isobaric'],
                  z=cross['Potential_temperature'],
                  ax=im['ax'],
                  plot_type='contour',
                  color='k',
                  levels=20,
                  vmin=250, vmax=450,
                  lw=1,
                  clabel=True,
                  clabel_fmt='%i',
                  )
im = cplt.plot_wind_field(
    ax=im['ax'],
    transform=False,
    steps=2,
    u=cross['t_wind'],
    v=cross['n_wind'],
    x_vals=cross['lon'],
    y_vals=cross['isobaric'],
    key_loc=(1, 1.05),
    key_length=20,
)

# %%
