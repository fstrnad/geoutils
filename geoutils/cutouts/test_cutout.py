# %%
from matplotlib.gridspec import GridSpec
from cartopy.crs import PlateCarree as plate
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
import geoutils.geodata.wind_dataset as wds
import geoutils.preprocessing.open_nc_file as of
import geoutils.plotting.plots as gplt
import geoutils.utils.time_utils as tu
import geoutils.geodata.base_dataset as bds
import geoutils.utils.spatial_utils as sput
import geoutils.utils.general_utils as gut
import geoutils.utils.file_utils as fut
import geoutils.plotting.plots as gplt
import geoutils.preprocessing.open_nc_file as of
import atlite
from importlib import reload
import geoutils.countries.countries as cnt
import geoutils.countries.capacities as cap


plot_dir = "/home/strnad/plots/dunkelflauten/"
data_dir = "/home/strnad/data/"

# %%
# Downloading the Cutout
country_name = "Germany"
year = 2013
dataset_name = f"{country_name}-{year}"
dataset_file = f"{dataset_name}.nc"
reload(cnt)
germany = cnt.get_country('Germany')
germany_shape = cnt.get_country_shape('Germany')
x1, y1, x2, y2 = germany_shape.total_bounds

# %%
if fut.exist_file(dataset_file):
    cutout = atlite.Cutout(dataset_file,
                           x=slice(x1 - 0.2, x2 + 0.2),
                           y=slice(y1 - 0.2, y2 + 0.2),).prepare()
else:
    cutout = atlite.Cutout(
        dataset_name,
        module="era5",
        x=slice(x1 - 0.2, x2 + 0.2),
        y=slice(y1 - 0.2, y2 + 0.2),
        chunks={"time": 100},
        time=f'{year}',
    )
    cutout.prepare(tmpdir='./download_era5/')
# %%
cells = cutout.grid

# %%
# Plt raw data time series
reload(gplt)
fig = plt.figure(figsize=(12, 7))
gs = GridSpec(3, 3, figure=fig)


projection = ccrs.Orthographic(-10, 35)
# projection = ccrs.PlateCarree()
ax = fig.add_subplot(gs[:, 0:2], projection=projection)
plot_grid_dict = dict(
    alpha=0.1,
    edgecolor="k",
    zorder=4,
    aspect="equal",
    facecolor="None",
    transform=plate(),
)
germany_shape.plot(ax=ax, zorder=1, transform=plate())
cells.plot(ax=ax, **plot_grid_dict)
ax.set_frame_on(False)
ax1 = fig.add_subplot(gs[0, 2])
cutout.data.wnd100m.mean(["x", "y"]).plot(ax=ax1)
ax1.set_frame_on(False)
ax1.xaxis.set_visible(False)

ax2 = fig.add_subplot(gs[1, 2], sharex=ax1)
cutout.data.influx_direct.mean(["x", "y"]).plot(ax=ax2)
ax2.set_frame_on(False)
ax2.xaxis.set_visible(False)

ax3 = fig.add_subplot(gs[2, 2], sharex=ax1)
cutout.data.runoff.mean(["x", "y"]).plot(ax=ax3)
ax3.set_frame_on(False)
ax3.set_xlabel(None)

savepath = f"{plot_dir}/{country_name}_raw_data_time_series.png"
gplt.save_fig(savepath)

# %%
# Power Generation Country shape
turbine = "Vestas_V112_3MW"
cap_factors = cutout.wind(turbine=turbine, capacity_factor=True)
power_generation_wind = cutout.wind(turbine=turbine, shapes=germany_shape).to_pandas(
).rename_axis(index="", columns="shapes")
# %%
power_generation_pv = cutout.pv(
    shapes=germany_shape,
    panel="CSi",
    orientation={'slope': 35.0, 'azimuth': 180.0}).to_pandas(
).rename_axis(index="", columns="shapes")
# %%
reload(gplt)
im = gplt.plot_map(cap_factors, title="Cap. Factor Wind", set_borders=True)
# cells.plot(ax=im['ax'], **plot_grid_dict)
savepath = f"{plot_dir}/{country_name}_cap_fact_wind.png"
gplt.save_fig(savepath, fig=im['fig'])
# %%

im = gplt.create_multi_plot(nrows=1, ncols=2, figsize=(13, 4),
                            projection_arr=['PlateCarree', None],
                            lon_range=[x1, x2], lat_range=[y1, y2],)
ax0 = im['ax'][0]
ax1 = im['ax'][1]
df = gpd.GeoDataFrame(germany_shape, columns=["geometry"]).assign(color=["1"])
df.plot(column="color", ax=im['ax'][0], zorder=1, transform=plate(), alpha=0.6)
ax0.set_frame_on(False)
power_generation_wind["Germany"].plot(ax=im['ax'][1],
                                      title="Wind Power Generation", color="indianred")

ax1.set_ylabel("Generation [MW]")
savepath = f"{plot_dir}/{country_name}_windpower_ts.png"
gplt.save_fig(savepath, fig=im['fig'])
# %%
reload(gplt)
im = gplt.create_multi_plot(nrows=1, ncols=2, figsize=(13, 4),
                            projection_arr=['PlateCarree', None],
                            lon_range=[x1, x2], lat_range=[y1, y2],)
ax0 = im['ax'][0]
ax1 = im['ax'][1]
df = gpd.GeoDataFrame(germany_shape, columns=["geometry"]).assign(color=["1"])
df.plot(column="color", ax=im['ax'][0], zorder=1, transform=plate(), alpha=0.6)
ax0.set_frame_on(False)
gplt.plot_2d(x=power_generation_pv["Germany"].index.values,
             y=power_generation_pv["Germany"].values,
             ax=ax1,
             title="PV Power Generation",
             color="indianred")

# power_generation_pv["Germany"].plot(ax=im['ax'][1],
#                                     title="PV Power Generation",
#                                     color="indianred")

ax1.set_ylabel("Generation [MW]")
savepath = f"{plot_dir}/{country_name}_pv_ts.png"
gplt.save_fig(savepath, fig=im['fig'])


# %%
# Indicator Matrix
reload(cnt)
country = "Germany"
indicator_matrix = cnt.cutout_country_cells(cutout, country)
im = gplt.plot_map(indicator_matrix,
                   vmin=0, vmax=1, set_borders=True,
                   cmap="Greens", label='Weight')
savepath = f"{plot_dir}/{country_name}_indicator_matrix.png"
gplt.save_fig(savepath, fig=im['fig'])

# %%
