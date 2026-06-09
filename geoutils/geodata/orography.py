# %%
# Create era5 orography.
import xarray as xr
import numpy as np
import geoutils.geodata.base_dataset as bds
import geoutils.plotting.plots as gplt
import geoutils.utils.time_utils as tu
import geoutils.geodata.downloader.download_era5 as d5
import geoutils.utils.met_utils as mut
from importlib import reload

# %%
# Download the files
# reload(d5)
# dw_data = d5.download_era5(variable='Geopotential',
#                            starty=2023,
#                            endy=2023,
#                            start_month='Jan',
#                            end_month='Jan',
#                            start_day=1,
#                            end_day=1,
#                            folder='/home/strnad/data/era5/',  # full path
                        #    run=True,
                        #    )
# %%
# Read files
reload(bds)
data_folder = '/home/strnad/data/climate_data/0.25/'
orograph_file = f'{data_folder}/orography_2019.nc'

grid_step = None
ds_era5_orography = bds.BaseDataset(data_nc=orograph_file,
                                    var_name=None,
                                    grid_step=1,
                                    decode_times=False,
                                    )
ds_era5_orography.average_time()

# %%
reload(gplt)
reload(mut)
mean_t = mut.geopotential_to_heigth(ds_era5_orography.get_da())-1024*mut.units('m')
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
