# %%
# Project for testing geoutils on paleo data.
import xarray as xr
import numpy as np
import geoutils.geodata.base_dataset as bds
import geoutils.plotting.plots as gplt
import geoutils.utils.time_utils as tu
from importlib import reload

# %%
# Read files
reload(bds)
data_folder = '/home/strnad/data/era5/'
orograph_file = f'{data_folder}/orography_2019.nc'

grid_step = None
ds_era5_orography = bds.BaseDataset(data_nc=orograph_file,
                                    var_name=None,
                                    grid_step=grid_step,
                                    decode_times=False,
                                    )
ds_era5_orography.average_time()
# %%
# %%
reload(gplt)
mean_t = ds_era5_orography.get_da()/9.81
im_comp = gplt.plot_map(dmap=mean_t,
                        plot_type='contourf',
                        cmap='cividis',
                        levels=12,
                        vmin=0,
                        # vmax=310,
                        title=f"Global Orography ERA5 ",
                        label=f'Elevation [m]',
                        orientation='horizontal',
                        tick_step=3,
                        # round_dec=2,
                        set_map=False,
                        sci=3,
                        )

# %%
savepath = f'{data_folder}/orography_era5.nc'
ds_era5_orography.save(filepath=savepath)