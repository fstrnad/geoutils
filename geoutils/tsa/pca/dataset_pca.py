# %%
# Apply PCA on dataset
import numpy as np
import xarray as xr
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from climnet.pca.pca import SpatioTemporalPCA


# %%
from climnet.datasets.dataset import AnomalyDataset, BaseDataset
import sys
import os

grid_type = 'fibonacci'
grid_step = 1

trmm = False
mswep = False

name = 'era5'

full_year = True
output_folder = 'pca'

num_cpus = 64
num_jobs = 30
time_range = ['1979-01-01', '2020-01-01']

lon_range = [-180, 180]
lat_range = [-90, 90]

var_names = ['t2m', 'psurf']

if os.getenv("HOME") == '/home/goswami/fstrnad80':
    dirname_t2m = f"/mnt/qb/goswami/data/era5/single_pressure_level/2m_temperature/"
    dirname_psurf = "/mnt/qb/goswami/data/era5/single_pressure_level/surface_pressure/"
    output_dir = "/home/goswami/fstrnad80/data/climnet/outputs/"
else:
    dirname_t2m = f"/home/strnad/data/era5/2m_temperature/"
    dirname_psurf = "/mnt/qb/goswami/data/era5/surface_pressure/psurf/"
    output_dir = '/home/strnad/data/climnet/outputs/'
    plot_dir = '/home/strnad/data/climnet/plots/'

fname_t2m = dirname_t2m + '2m_temperature_sfc_1979_2020_mon_mean.nc'
fname_psurf = dirname_psurf + \
    'vertical_integral_of_divergence_of_total_energy_flux_sfc_1979_2020.nc'

# %%

print('Loading Data')

for idx, fname in enumerate([fname_t2m, ]):
    var_name = var_names[idx]
    dataset_file = output_dir + \
        f"/{output_folder}/{name}_{grid_type}_{grid_step}_{var_name}_ds.nc"

    if os.path.exists(dataset_file) is False:
        print('Create Dataset')
        ds = AnomalyDataset(data_nc=fname,
                            var_name=var_name,
                            lon_range=lon_range,
                            lat_range=lat_range,
                            time_range=time_range,
                            grid_step=grid_step,
                            grid_type=grid_type,
                            large_ds=False,
                            )
        ds.save(dataset_file)
