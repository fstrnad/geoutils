# %%
import geoutils.geodata.wind_dataset as wds
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
plot_dir = "/home/strnad/data/plots/summer_monsoon/"
output_folder = "summer_monsoon"

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
