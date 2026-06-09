# %%
# packages required
import geoutils.plotting.plots as cplt
import geoutils.tsa.fft_analysis as ffta
import geoutils.utils.time_utils as tu
import numpy as np
import xarray as xr
import geoutils.geodata.base_dataset as bds
import geoutils.geodata.wind_dataset as wds
from importlib import reload

plot_dir = "/home/strnad/data/rmm/"
data_dir = "/home/strnad/data/"

grid_step = 2.5
# %%
dataset_file = data_dir + \
    f"climate_data/2.5/era5_ttr_{2.5}_ds.nc"

ds_olr = bds.BaseDataset(data_nc=dataset_file,
                         can=True,
                         an_types=['dayofyear'],
                         detrend=True,
                         )

# wind data
reload(wds)
nc_files_u = []
nc_files_v = []
nc_files_w = []
levs = [200, 850]
for lev in levs:
    dataset_file_u = data_dir + \
        f"/climate_data/2.5/era5_u_{2.5}_{lev}_ds.nc"
    nc_files_u.append(dataset_file_u)
    dataset_file_v = data_dir + \
        f"/climate_data/2.5/era5_v_{2.5}_{lev}_ds.nc"
    nc_files_v.append(dataset_file_v)


ds_wind = wds.Wind_Dataset(data_nc_u=nc_files_u,
                           data_nc_v=nc_files_v,
                           plevels=levs,
                           can=True,
                           an_types=['dayofyear'],
                           init_mask=False,
                           )
# %%
reload(tu)
da = ds_olr.ds['olr']
da_slideing = tu.sliding_window_mean(da=da, length=120)
# %%
# substract harmonics from time series
ts = da.mean(dim=['lat', 'lon'])
# %%
reload(ffta)
reload(cplt)
fft_ts = ffta.compute_fft(ts=ts)
reco = ffta.revmove_harmonics(ts=ts, num_harmonics=3)
fft_reco = ffta.compute_fft(ts=reco)
cplt.plot_xy(x_arr=[fft_ts['period'],
                    fft_reco['period']],
             y_arr=[fft_ts['power'],
                    fft_reco['power']],
             xlim=[0, 400],
             )
# %%
reload(ffta)
ts_reconstruct = ffta.inverse_fft(ts_fft = fft_ts['fft'])
# %%
