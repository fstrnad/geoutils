# %%
# Project for testing geoutils on paleo data.
import xarray as xr
import numpy as np
import geoutils.geodata.base_dataset as bds
import geoutils.plotting.plots as gplt
import geoutils.utils.time_utils as tu
from importlib import reload

# %%
# Read files
reload(bds)
data_folder = '/home/strnad/data/paleo'
hadcm_file = f'{data_folder}/HadCM3B_M2.1aN-deepmip_sens_1xCO2-mlotst-v1.0.mean_r360x180.nc'
hadcm_mask_file_pi = f'{data_folder}/mask_1_Preindustrial_.nc'
mask_file_eocene = f'{data_folder}/mask_1_Eocene_.nc'
cesm_file = f'{data_folder}/CESM1.2_CAM5-deepmip_sens_1xCO2-mlotst-v1.0.mean_r360x180.nc'

grid_step = 1

ds_hadcm = bds.BaseDataset(data_nc=hadcm_file,
                           var_name=None,
                           grid_step=grid_step,
                           decode_times=False,
                           lsm_file=mask_file_eocene
                           )
ds_cesm = bds.BaseDataset(data_nc=cesm_file,
                          var_name=None,
                          grid_step=grid_step,
                          decode_times=False,
                          lsm_file=mask_file_eocene
                          )
# %%
ds = ds_hadcm.get_da()
ds_month = {}
max_mixedlayer = 0.6*float(ds.max())
months = tu.months
im = gplt.create_multi_plot(nrows=3, ncols=4,
                            projection='PlateCarree',
                            central_longitude=180,
                            wspace=0.1, hspace=0.8)


for j in range(0, 12):
    ds_month[j] = ds.sel(time=j)

    im_comp = gplt.plot_map(dmap=ds_month[j],
                            ax=im['ax'][j],
                            fig=im['fig'],
                            plot_type='contourf',
                            cmap='cividis',
                            # levels=12,
                            vmin=0,
                            vmax=200,
                            title=f"HadCM Mixed-layer depth, "+months[j],
                            bar=True,
                            plt_grid=True,
                            label=f'mixed-layer depth [m]',
                            orientation='horizontal',
                            tick_step=2,
                            round_dec=2,
                            set_map=False,
                            )


# %%
ds = ds_cesm.get_da()
ds_month = {}
max_mixedlayer = 0.6*float(ds.max())
months = tu.months
im = gplt.create_multi_plot(nrows=3, ncols=4,
                            projection='PlateCarree',
                            central_longitude=180,
                            wspace=0.1, hspace=0.8)


for j in range(0, 12):
    ds_month[j] = ds.sel(time=j)

    im_comp = gplt.plot_map(dmap=ds_month[j],
                            ax=im['ax'][j],
                            fig=im['fig'],
                            plot_type='contourf',
                            cmap='cividis',
                            # levels=12,
                            vmin=0,
                            vmax=700,
                            title=f"CESM Mixed-layer depth, {months[j]}",
                            bar=True,
                            plt_grid=True,
                            label=f'mixed-layer depth [m]',
                            orientation='horizontal',
                            tick_step=2,
                            round_dec=2,
                            set_map=False,
                            )

# %%

im_comp = gplt.plot_map(dmap=ds_hadcm.mask,
                        plot_type='contourf',
                        cmap='cividis',
                        # levels=12,
                        vmin=0,
                        vmax=1,
                        title=f"Mask HadCM PI",
                        bar=True,
                        plt_grid=True,
                        label=f'mixed-layer depth [m]',
                        orientation='horizontal',
                        tick_step=2,
                        round_dec=2,
                        set_map=False,
                        central_longitude=180
                        )
