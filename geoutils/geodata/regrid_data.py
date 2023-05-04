import geoutils.plotting.plots as cplt
import os
import geoutils.utils.general_utils as gut
import geoutils.geodata.base_dataset as bds
import geoutils.utils.general_utils as gut
import geoutils.utils.file_utils as fut
import os
from importlib import reload

output_folder = 'climate_data'
time_range = ['1979-01-01', '2019-12-31']

if os.getenv("HOME") == '/home/goswami/fstrnad80':
    output_dir = "/mnt/qb/goswami/processed_data/"
else:
    dirname_ttr = "/mnt/qb/goswami/data/era5/single_pressure_level/top_net_thermal_radiation/"
    dirname_sp = "/home/strnad/data/era5/single_pressure_level/surface_pressure/"
    dirname_t2m = "/mnt/qb/goswami/data/era5/single_pressure_level/2m_temperature/"
    output_dir = '/home/strnad/data/climnet/outputs/'
    plot_dir = '/home/strnad/data/climnet/plots/'


# %%
# TRMM
reload(bds)
if os.getenv("HOME") == '/home/goswami/fstrnad80':
    dirname_trmm = "/mnt/qb/goswami/data/trmm/trmm_pr_daily_1998_2019.nc4"
else:
    dirname_trmm = "/home/strnad/data/trmm/trmm_pr_daily_1998_2019.nc4"
var_name = 'pr'
name = 'trmm'
grid_step = 1
dataset_file = output_dir + \
    f"/{output_folder}/{name}_{var_name}_{grid_step}_1998_2020_ds.nc"

if os.path.exists(dataset_file) is False:
    gut.myprint(f'Create Dataset {dataset_file}')
    ds = bds.BaseDataset(data_nc=dirname_trmm,
                         var_name=var_name,
                         grid_step=grid_step,
                         large_ds=True,
                         time_range=['1998-01-01', '2020-01-01'],
                         )
    ds.save(dataset_file)
else:
    gut.myprint(f'File {dataset_file} already exists!')


# %%
# Test how the regridded data looks like
# reload(cplt)
# mean_pr = ds.ds.pr.mean(dim='time')

# cplt.plot_map(
#     mean_pr,
#     plot_type="contourf",
#     label=f"Mean Precipitation [mm/day]",
#     orientation="horizontal",
#     vmin=0,
#     vmax=10,
#     levels=12,
#     cmap="coolwarm_r",
#     tick_step=2,
#     round_dec=1,
# )


# # %%
# # MSWEP
# reload(bds)
# if os.getenv("HOME") == '/home/goswami/fstrnad80':
#     dirname_mswep = "/mnt/qb/goswami/data/mswep/daymean/precipitation_1979_2021.nc"
# else:
#     dirname_mswep = "/mnt/qb/goswami/data/mswep/daymean/precipitation_1979_2021.nc"
# var_name = 'pr'
# name = 'mswep'
# grid_step = 1
# dataset_file = output_dir + \
#     f"/{output_folder}/{name}_{var_name}_{grid_step}_1979_2021_ds.nc"

# if os.path.exists(dataset_file) is False:
#     gut.myprint(f'Create Dataset {dataset_file}')
#     ds = bds.BaseDataset(data_nc=dirname_mswep,
#                          var_name=var_name,
#                          grid_step=grid_step,
#                          large_ds=True,
#                          time_range=['1979-01-02', '2020-12-30'],
#                          )
#     ds.save(dataset_file)
# else:
#     gut.myprint(f'File {dataset_file} already exists!')
