# %%
# Based on https://unidata.github.io/MetPy/latest/examples/cross_section.html#sphx-glr-examples-cross-section-py
# Copyright (c) 2018 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
======================
Cross Section Analysis
======================

The MetPy function `metpy.interpolate.cross_section` can obtain a cross-sectional slice through
gridded data.
"""

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
import geoutils.utils.file_utils as fut
import metpy.calc as mpcalc
from metpy.cbook import get_test_data
from metpy.interpolate import cross_section
import geoutils.geodata.multilevel_pressure as mp
output_dir = "/home/strnad/data/climnet/outputs/summer_monsoon/"
plot_dir = "/home/strnad/data/plots/summer_monsoon/"
data_dir = "/home/strnad/data/"
lat_range_cut = [-30, 70]
lon_range_cut = [-180, 180]

# %% #############################
# **Getting the data**
# ERA 5 data
reload(mp)
plevels = [200, 600, 1000]

nc_files_q = []
for plevel in plevels:
    dataset_file_q = data_dir + \
        f"/climate_data/2.5/era5_q_{2.5}_{plevel}_ds.nc"
    nc_files_q.append(dataset_file_q)
time_range = ['1980-01-01', '2000-12-31']
ds_q = mp.MultiPressureLevelDataset(data_nc=nc_files_q,
                                    can=True,
                                    an_types=['month', 'JJAS'],
                                    lon_range=lon_range_cut,
                                    lat_range=lat_range_cut,
                                    plevels=plevels,
                                    time_range=time_range,
                                    )


# This example uses [NARR reanalysis data](
# https://www.ncei.noaa.gov/products/weather-climate-models/north-american-regional)
# for 18 UTC 04 April 1987 from NCEI.
date = tu.str2datetime('1987-04-04')

# We use MetPy's CF parsing to get the data ready for use, and squeeze down the size-one time
# dimension.

data = xr.open_dataset(get_test_data('narr_example.nc', False))
filepath = './test_files/narr_example.nc'
fut.save_ds(data, filepath=filepath)

data = data.metpy.parse_cf().squeeze()
# data = data.set_coords(('lat', 'lon'))
##############################
# %% Define start and end points:

start = (37.0, -105.0)
end = (35.5, -65.0)

##############################
# %% Get the cross section, and convert lat/lon to supplementary coordinates:
reload(mut)
cross_q = cross_section(ds_q.ds['q'].sel(time=date,),
                        start, end).set_coords(('lat', 'lon'))
# %%
cross = cross_section(data, start, end).set_coords(('lat', 'lon'))
cross['Potential_temperature'] = mut.potential_temperature(
    pressure=cross['isobaric'],
    temperature=cross['Temperature']
)
cross['Relative_humidity'] = mut.specific_humidity_to_relative_humidity(
    pressure=cross['isobaric'],
    temperature=cross['Temperature'],
    specific_humidity=cross['Specific_humidity'],
    percentage=False
)

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
fig = plt.figure(1, figsize=(16., 9.))
data_crs = cplt.get_projection(projection='LambertConformal')
data_crs = data['Geopotential_height'].metpy.cartopy_crs()
ax_inset = plt.axes(projection=data_crs)


# Plot geopotential height at 500 hPa using xarray's contour wrapper
ax_inset.contour(data['x'], data['y'],
                 data['Geopotential_height'].sel(isobaric=500.),
                 levels=np.arange(5100, 6000, 60), cmap='inferno')
# Plot the path of the cross section
endpoints = data_crs.transform_points(ccrs.Geodetic(),
                                      *np.vstack([start, end]).transpose()[::-1])
ax_inset.scatter(endpoints[:, 0], endpoints[:, 1], c='k', zorder=2)
ax_inset.plot(cross['x'], cross['y'], c='k', zorder=2)

# Add geographic features
ax_inset.coastlines()
