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
import geoutils.preprocessing.open_nc_file as of

reload(of)

output_dir = "/home/strnad/data/"
data_dir = "/home/strnad/data/"
plot_dir = "/home/strnad/data/plots/summer_monsoon/"
output_folder = "summer_monsoon"

# %%
# Load wind fields
reload(wds)
levs = np.arange(100, 1050, 100)
grid_step = 1
# %%
reload(fut)
reload(of)
for lev in levs:
    for var_name in ['u', 'v']:
        dataset_file = data_dir + \
            f"/climate_data/{grid_step}/era5_{var_name}_{grid_step}_{lev}_ds.nc"
        wind_ds = xr.open_dataset(dataset_file)
        if not tu.check_hour_occurrence(wind_ds.time):
            # This automatically removes hourly data in time index
            wind_ds = of.open_nc_file(dataset_file,
                                  verbose=False)
            new_file_name = data_dir + \
                f"/climate_data/{grid_step}/era5_{var_name}_{grid_step}_{lev}_ds.nc"
            nc_file = gut.delete_all_non_dimension_attributes(wind_ds)
            fut.save_ds(nc_file, new_file_name, backup=False)
