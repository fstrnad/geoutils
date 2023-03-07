# %%
from geoutils.geodata.base_dataset import BaseDataset
import geoutils.utils.general_utils as gut
import os
output_folder = 'climate_data'
time_range = ['1979-01-01', '2019-12-31']

if os.getenv("HOME") == '/home/goswami/fstrnad80' or os.getenv("HOME") == '/home/goswami/jschloer46':
    dirname_sp = "/mnt/qb/goswami/data/era5/single_pressure_level/surface_pressure/"
    dirname_t2m = "/mnt/qb/goswami/data/era5/single_pressure_level/2m_temperature/"
    dirname_sst = "/mnt/qb/goswami/data/era5/single_pressure_level/sea_surface_temperature/"
    output_dir = "/mnt/qb/goswami/processed_data/"
    dirname_ttr = "/mnt/qb/goswami/data/era5/single_pressure_level/top_net_thermal_radiation/"
    dirname_tcrw = "/mnt/qb/goswami/data/era5/single_pressure_level/total_column_rain_water/"
    dirname_tcrw = "/mnt/qb/goswami/data/era5/single_pressure_level/total_column_rain_water/"
    dirname_ewvf = "/mnt/qb/goswami/data/era5/single_pressure_level/vertical_integral_of_eastward_water_vapour_flux/"
    dirname_nwvf = "/mnt/qb/goswami/data/era5/single_pressure_level/vertical_integral_of_northward_water_vapour_flux/"
    dirname_vimd = "/mnt/qb/goswami/data/era5/single_pressure_level/vertically_integrated_moisture_divergence/"
else:
    dirname_ttr = "/mnt/qb/goswami/data/era5/single_pressure_level/top_net_thermal_radiation/"
    dirname_sp = "/home/strnad/data/era5/single_pressure_level/surface_pressure/"
    dirname_t2m = "/mnt/qb/goswami/data/era5/single_pressure_level/2m_temperature/"
    output_dir = '/home/strnad/data/climnet/outputs/'
    plot_dir = '/home/strnad/data/climnet/plots/'

# %%
# # MSWEP
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
#     ds = BaseDataset(data_nc=dirname_mswep,
#                      var_name=var_name,
#                      grid_step=grid_step,
#                      large_ds=True,
#                      time_range=['1979-01-02', '2020-12-30'],
#                      )
#     ds.save(dataset_file)
# else:
#     print(f'File {dataset_file} already exists!')


# %%
# ERA 5 single pressure levels
fname_sp = dirname_sp + \
    'surface_pressure_sfc_1979_2020.nc'
fname_t2m = dirname_t2m + \
    '2m_temperature_sfc_1979_2020.nc'
fname_sst = dirname_sst + \
    'sea_surface_temperature_sfc_1979_2021.nc'
fname_ttr = dirname_ttr + \
    'top_net_thermal_radiation_sfc_1979_2020.nc'
fname_tcrw = dirname_tcrw + \
    'total_column_rain_water_1990_2020.nc'
fname_ewvf = dirname_ewvf + \
    'vertical_integral_of_eastward_water_vapour_flux_sfc_1979_2020.nc'
fname_nwvf = dirname_nwvf + \
    'vertical_integral_of_northward_water_vapour_flux_sfc_1979_2020.nc'
fname_vimd = dirname_vimd + \
    'vertically_integrated_moisture_divergence_sfc_1979_2020.nc'

# %%
# print('Loading Data', flush=True)

# names = ['era5', 'era5', 'era5', 'era5', 'era5', 'era5']
# fnames = [fname_t2m, fname_t2m, fname_ttr, fname_tcrw]
# fnames_dict = dict(
#         vimd=fname_vimd,
#         t2m=fname_t2m,
#         sst=fname_sst,
#         sp=fname_t2m,
#         ttr=fname_ttr,
#         tcrw=fname_tcrw,
#         ewvf=fname_ewvf,
#         nwvf=fname_nwvf
#     )

# name = 'era5'
# var_names = ['vimd', 'ewvf', 'nwvf', 't2m', 'sp', 'ttr', 'tcrw']
# var_names = ['t2m', 'ttr']
# var_names = ['sst']

# for idx, var_name in enumerate(var_names):
#     fname = fnames_dict[var_name]
#     dataset_file = output_dir + \
#         f"/{output_folder}/{name}_{var_name}_{grid_step}_ds.nc"

#     if os.path.exists(dataset_file) is False:
#         print(f'Create Dataset {dataset_file}', flush=True)
#         ds = BaseDataset(data_nc=fname,
#                              var_name=var_name,
#                              grid_step=grid_step,
#                              large_ds=True,
#                              )
#         ds.save(dataset_file)
#     else:
#         print(f'File {fname} already exists!', flush=True)

# %%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Wind and geopot fields on different pressure levels
plevels = [5, 50, 100, 150, 200, 300, 400, 450, 500, 550,
           600, 650, 700, 750, 800, 850, 1000]
plevels = [150, 250, 300,
           600, 850, 950, 1000]
# plevels = [250, 600, 850, 1000]
plevels = [800, 550, 650]
plevels = [200, 850, 500]
name = 'era5'
grid_step = 1


for plevel in plevels:

    if os.getenv("HOME") == '/home/goswami/fstrnad80' or os.getenv("HOME") == '/home/goswami/jschloer46':
        dirname_uwind = f"/mnt/qb/goswami/data/era5/multi_pressure_level/u_component_of_wind/{plevel}/"
        dirname_vwind = f"/mnt/qb/goswami/data/era5/multi_pressure_level/v_component_of_wind/{plevel}/"
        dirname_w = f"/mnt/qb/goswami/data/era5/multi_pressure_level/vertical_velocity/{plevel}/"
        dirname_z = f"/mnt/qb/goswami/data/era5/multi_pressure_level/geopotential/{plevel}/"
        dirname_pv = f"/mnt/qb/goswami/data/era5/multi_pressure_level/potential_vorticity/{plevel}/"
        dirname_sh = f"/mnt/qb/goswami/data/era5/multi_pressure_level/specific_humidity/{plevel}/"
        dirname_temp = f"/mnt/qb/goswami/data/era5/multi_pressure_level/temperature/{plevel}/"
    else:
        dirname_uwind = f"/home/strnad/data/era5/multi_pressure_level/u_component_of_wind/{plevel}/"
        dirname_vwind = f"/home/strnad/data/era5/multi_pressure_level/v_component_of_wind/{plevel}/"
        dirname_w = f"/home/strnad/data/era5/multi_pressure_level/vertical_velocity/{plevel}/"
        dirname_z = f"/home/strnad/data/era5/multi_pressure_level/geopotential/{plevel}/"

    fname_u = dirname_uwind + f'u_component_of_wind_{plevel}_1979_2020.nc'
    fname_v = dirname_vwind + f'v_component_of_wind_{plevel}_1979_2020.nc'
    fname_w = dirname_w + f'vertical_velocity_{plevel}_1979_2020.nc'
    fname_z = dirname_z + f'geopotential_{plevel}_1979_2020.nc'

    fname_pv = dirname_pv + f'potential_vorticity_{plevel}_1979_2020.nc'
    fname_sh = dirname_sh + f'specific_humidity_{plevel}_1979_2020.nc'
    fname_temp = dirname_temp + f'temperature_{plevel}_1979_2020.nc'

    fnames_dict = dict(
        u=fname_u,
        v=fname_v,
        w=fname_w,
        pv=fname_pv,
        q=fname_sh,
        t=fname_temp,
        z=fname_z
    )

    var_names = ['u', 'v', 'w', 'pv', 'z', 'q']
    var_names = ['u', 'v', 'z', 'pv']
    var_names = ['u', 'v']
    var_names = ['t', 'q']
    var_names = ['u', 'v', 'w']

    for idx, var_name in enumerate(var_names):
        fname = fnames_dict[var_name]
        dataset_file = output_dir + \
            f"/{output_folder}/{name}_{var_name}_{grid_step}_{plevel}_ds.nc"

        if os.path.exists(dataset_file) is False:
            gut.myprint(f'Create Dataset {dataset_file}')
            ds = BaseDataset(data_nc=fname,
                             var_name=var_name,
                             grid_step=grid_step,
                             large_ds=True,
                             time_range=time_range,
                             )
            ds.save(dataset_file)
        else:
            gut.myprint(f'File {dataset_file} already exists!')


# %%
# NOAA
# if os.getenv("HOME") == '/home/goswami/fstrnad80':
#     dirname_sppw = '/mnt/qb/goswami/data/NCEP-NCAR/single_pressure_level/pr_wtr/'
# else:
#     dirname_sppw = '/home/strnad/data/NCEP-NCAR/pr_wtr/eatm/'
# var_name = 'pr_wtr'
# fname_sppw = dirname_sppw + 'pr_wtr_eatm_1979_2020.nc'
# name = 'noaa'
# dataset_file = output_dir + \
#         f"/{output_folder}/{name}_{var_name}_{grid_step}_ds.nc"

# if os.path.exists(dataset_file) is False:
#     print(f'Create Dataset {dataset_file}', flush=True)
#     ds = BaseDataset(data_nc=fname_sppw,
#                          var_name=var_name,
#                          grid_step=grid_step,
#                          large_ds=True,
#                         #  time_range=time_range,
#                          )
#     ds.save(dataset_file)
# else:
#     print(f'File {dataset_file} already exists!')
