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
                          init_mask=True,
                          decode_times=False,
                          freq='M'
                          )

ds_cesm.rename_var(new_var_name='temp')
ds_cesm.add_var_attribute({'standard_name': 'air_temperature'})
lpj_ds = xr.open_dataset(lpj_file)
full_ds = xr.open_dataset(cesm_file)
ds_cesm.save(filepath=cesm_file_new, unlimited_dim='lat',
             classic_nc=True)
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
lpj_file = f'{data_folder}/CESM1.2_CAM5-deepmip_stand_3xCO2-prec-v1.0.time_series.nc'
cesm_file = f'{data_folder}/CESM1.2_CAM5-deepmip_stand_3xCO2-pr-v1.0.time_series.nc'
cesm_file_new = f'{data_folder}/CESM1.2_CAM5-deepmip_stand_3xCO2-pr-v1.0.time_series_new.nc'

# lpj_mask_file_pi = f'{data_folder}/mask_1_Preindustrial_.nc'
# mask_file_eocene = f'{data_folder}/mask_1_Eocene_.nc'

grid_step = None

ds_cesm_pr = bds.BaseDataset(data_nc=cesm_file,
                             var_name=None,
                             grid_step=grid_step,
                             lsm_file=None,
                             sort=False,
                             lon360=True,
                             init_mask=True,
                             )
ds_cesm_pr.rename_var(new_var_name='prec')
prec = ds_cesm_pr.ds['prec']
ds_cesm_pr.ds['prec'] = xr.where(prec < 0, 0, prec)
ds_cesm_pr.set_source_attrs()
# ds_cesm.save(filepath=cesm_file_new)

# %%
# COSMOS
reload(bds)
data_folder = '/home/strnad/data/paleo'
cosmos_file = f'{data_folder}/COSMOS-landveg_r2413-deepmip_stand_3xCO2-pr-v1.0.time_series.nc'
lpj_file = f'{data_folder}/CESM1.2_CAM5-deepmip_stand_3xCO2-prec-v1.0.time_series.nc'
cesm_file = f'{data_folder}/CESM1.2_CAM5-deepmip_stand_3xCO2-pr-v1.0.time_series.nc'
cosmos_file_new = f'{data_folder}/COSMOS-landveg_r2413-deepmip_stand_3xCO2-pr-v1.0.time_series_new.nc'

# lpj_mask_file_pi = f'{data_folder}/mask_1_Preindustrial_.nc'
# mask_file_eocene = f'{data_folder}/mask_1_Eocene_.nc'

grid_step = None

ds_cosmos_pr = bds.BaseDataset(data_nc=cosmos_file,
                               var_name=None,
                               decode_times=False,
                               grid_step_lon=1.25,
                               grid_step_lat=1.5,
                               lsm_file=None,
                               sort=True,
                               lon360=True,
                               init_mask=True,
                               )
ds_cosmos_pr.rename_var(new_var_name='prec')
prec = ds_cosmos_pr.ds['prec']
ds_cosmos_pr.ds['prec'] = xr.where(prec < 0, 0, prec)
ds_cosmos_pr.set_source_attrs()
