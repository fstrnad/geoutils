# %%
import geoutils.geodata.wind_dataset as wds
import geoutils.geodata.multilevel_pressure as mp
import geoutils.utils.statistic_utils as sut
from importlib import reload
import xarray as xr
import numpy as np
import geoutils.geodata.base_dataset as bds
import geoutils.utils.time_utils as tu
import geoutils.utils.file_utils as fut
import geoutils.utils.general_utils as gut

output_dir = "/home/strnad/data/"
data_dir = "/home/strnad/data/"

# %% OLR
grid_step = 1
dataset_file = data_dir + \
    f"climate_data/{grid_step}/era5_ttr_{grid_step}_ds.nc"

ds_olr = bds.BaseDataset(data_nc=dataset_file,
                         grid_step=grid_step,
                         )
# %%
nc_file = ds_olr.ds['olr']
nc_file = gut.delete_all_non_dimension_attributes(nc_file)
new_file_name = data_dir + \
    f"climate_data/{grid_step}/era5_olr_{grid_step}_ds.nc"
fut.save_ds(nc_file, new_file_name, backup=True)

# %%
# Q

nc_files_q = []
nc_files_t = []
nc_files_z = []

levs = [400]

grid_step = 1
for plevel in levs:
    dataset_file_q = data_dir + \
        f"/climate_data/{grid_step}/era5_q_{grid_step}_{plevel}_ds.nc"
    nc_files_q.append(dataset_file_q)
    dataset_file_t = data_dir + \
        f"/climate_data/{grid_step}/era5_t_{grid_step}_{plevel}_ds.nc"
    nc_files_t.append(dataset_file_t)
    dataset_file_z = data_dir + \
        f"/climate_data/{grid_step}/era5_z_{grid_step}_{plevel}_ds.nc"
    nc_files_z.append(dataset_file_z)

# %%
reload(mp)
ds_q = mp.MultiPressureLevelDataset(data_nc=nc_files_q,
                                    plevels=levs,
                                    grid_step=grid_step,
                                    )
# %%
# Save the data as new files
for lev in levs:
    for var_name in ['q']:
        new_file_name = data_dir + \
            f"/climate_data/{grid_step}/era5_{var_name}_{grid_step}_{lev}_ds.nc"
        if len(levs) > 1:
            nc_file = ds_q.ds[var_name].sel(lev=lev)
        else:
            nc_file = ds_q.ds[var_name]
        nc_file = gut.delete_all_non_dimension_attributes(nc_file)
        fut.save_ds(nc_file, new_file_name, backup=True)
# %%
ds_t = mp.MultiPressureLevelDataset(data_nc=nc_files_t,
                                    plevels=levs,
                                    # grid_step=grid_step,
                                    )
# %%
# Save the data as new files
for lev in levs:
    for var_name in ['t']:
        new_file_name = data_dir + \
            f"/climate_data/{grid_step}/era5_{var_name}_{grid_step}_{lev}_ds.nc"
        if len(levs) > 1:
            nc_file = ds_t.ds[var_name].sel(lev=lev)
        else:
            nc_file = ds_t.ds[var_name]
        nc_file = gut.delete_all_non_dimension_attributes(nc_file)
        fut.save_ds(nc_file, new_file_name, backup=True)


# %%
# Load wind fields
reload(wds)
nc_files_u = []
nc_files_v = []
nc_files_w = []
levs = [800, 500]
grid_step = 1
for lev in levs:
    dataset_file_u = data_dir + \
        f"/climate_data/{grid_step}/era5_u_{grid_step}_{lev}_ds.nc"
    nc_files_u.append(dataset_file_u)
    dataset_file_v = data_dir + \
        f"/climate_data/{grid_step}/era5_v_{grid_step}_{lev}_ds.nc"
    nc_files_v.append(dataset_file_v)
    dataset_file_w = data_dir + \
        f"/climate_data/{grid_step}/era5_w_{grid_step}_{lev}_ds.nc"
    nc_files_w.append(dataset_file_w)

reload(wds)
ds_wind = wds.Wind_Dataset(data_nc_u=nc_files_u,
                           data_nc_v=nc_files_v,
                           can=False,
                           plevels=levs,
                           init_mask=False,
                           convention_labelling=False,
                           )
# %%
reload(fut)
for lev in levs:
    for var_name in ['u', 'v']:
        new_file_name = data_dir + \
            f"/climate_data/{grid_step}/era5_{var_name}_{grid_step}_{lev}_ds.nc"
        nc_file = ds_wind.ds[var_name].sel(lev=lev)
        nc_file = gut.delete_all_non_dimension_attributes(nc_file)
        fut.save_ds(nc_file, new_file_name, backup=True)
