# %%
# Project for testing geoutils on paleo data.
import xarray as xr
import numpy as np
import geoutils.geodata.base_dataset as bds
import geoutils.plotting.plots as gplt
import geoutils.utils.time_utils as tu
import geoutils.utils.general_utils as gut
from importlib import reload

# %%
# Read files
reload(bds)
data_folder = '/home/strnad/data/lpjguess-input/temp/'
cesm_file = f'{data_folder}/CESM1.2_CAM5-deepmip_stand_3xCO2-tas-v1.0.time_series.nc'
lpj_file = f'{data_folder}/CESM1.2_CAM5-deepmip_stand_3xCO2-temp-v1.0.time_series.nc'
cesm_file_new = f'{data_folder}/CESM1.2_CAM5-deepmip_stand_3xCO2-temp-v1.0.time_series_new.nc'

# lpj_mask_file_pi = f'{data_folder}/mask_1_Preindustrial_.nc'
# mask_file_eocene = f'{data_folder}/mask_1_Eocene_.nc'

grid_step = None

ds_cesm = bds.BaseDataset(data_nc=cesm_file,
                          var_name=None,
                          grid_step=grid_step,
                          lsm_file=None,
                          sort=False,
                          lon360=True,
                          )
ds_cesm.rename_var(new_var_name='temp')
ds_cesm.ds['temp'].attrs['standard_name'] = 'air_temperature'
ds_cesm.save(filepath=cesm_file_new)
# %%

im_comp = gplt.plot_map(dmap=ds_cesm.get_da().mean(dim='time'),
                        plot_type='contourf',
                        cmap='cividis',
                        # levels=12,
                        # vmin=0,
                        # vmax=1,
                        title=f"Mask lpj PI",
                        bar=True,
                        plt_grid=True,
                        label=f'Global Mean Temperature [K]',
                        orientation='horizontal',
                        tick_step=2,
                        round_dec=2,
                        set_map=False,
                        # central_longitude=180
                        )
# %%
# for precipitation
# %%
# Read files
reload(bds)
data_folder = '/home/strnad/data/lpjguess-input/prec/'
cesm_file = f'{data_folder}/CESM1.2_CAM5-deepmip_stand_3xCO2-prec-v1.0.time_series.nc'
lpj_file = f'{data_folder}/CESM1.2_CAM5-deepmip_stand_3xCO2-pr-v1.0.time_series.nc'
cesm_file_new = f'{data_folder}/CESM1.2_CAM5-deepmip_stand_3xCO2-pr-v1.0.time_series_new.nc'

# lpj_mask_file_pi = f'{data_folder}/mask_1_Preindustrial_.nc'
# mask_file_eocene = f'{data_folder}/mask_1_Eocene_.nc'

grid_step = None

ds_cesm = bds.BaseDataset(data_nc=cesm_file,
                          var_name=None,
                          grid_step=grid_step,
                          lsm_file=None,
                          sort=False,
                          lon360=True,
                          )
ds_cesm.rename_var(new_var_name='pr')
# ds_cesm.save(filepath=cesm_file_new)