# %%
import geoutils.geodata.base_dataset as bds
import xarray as xr
import geoutils.utils.time_utils as tu
import geoutils.utils.general_utils as gut
import geoutils.utils.file_utils as fut
import geoutils.plotting.plots as cplt
from importlib import reload

# %%
plot_dir = "/home/strnad/data/plots/2023/"
data_dir = "/home/strnad/data/"
pr_folder = f'{data_dir}/mswep/NRT/'
grid_step = 1
name = 'mswep'
var_name = 'pr'

files = fut.find_files_with_string(pr_folder, "2023")
# %%
ds_pr_2023 = bds.BaseDataset(data_nc=files,
                             grid_step=grid_step,
                             large_ds=False,)

# %%

output_dir = '/home/strnad/data/'
output_folder = 'climate_data'
dataset_file = output_dir + \
    f"/{output_folder}/{grid_step}/2023_{name}_{var_name}_{grid_step}_ds.nc"

ds_pr_2023.save(dataset_file)

# %%
# precipitation all
reload(bds)
grid_step = 1
dataset_file = data_dir + \
    f"climate_data/{grid_step}/mswep_pr_{grid_step}_ds.nc"

ds_pr = bds.BaseDataset(data_nc=dataset_file,
                        can=True,
                        an_types=['month', 'JJAS'],
                        month_range=['Jun', 'Sep']
                        )
# %%
# Mean Pr 2023
label = f'Pr [mm/day]'
vmin = 0
vmax = 10
lon_range = [-30, 170]
lat_range = [-30, 60]

climatology = ds_pr.ds['pr'].mean(dim='time')
mean_2023 = ds_pr_2023.ds['pr'].mean(dim='time')
im = cplt.create_multi_plot(nrows=1,
                            ncols=2,
                            title=rf'Comparison Mean JJAS Pr 2023',
                            orientation='horizontal',
                            hspace=0.5,
                            wspace=0.23,
                            projection='PlateCarree',
                            lat_range=lat_range,
                            lon_range=lon_range,
                            )
titles = ['Climatology', '2023']
for d, mean_data in enumerate([climatology, mean_2023]):
    im_olr = cplt.plot_map(mean_data,
                           ax=im['ax'][d],
                           title=f'Mean {titles[d]}',
                           plot_type='contourf',
                           cmap='viridis_r',
                           levels=20,
                           label=label,
                           vmin=vmin, vmax=vmax,
                           tick_step=2,
                           )

savepath = plot_dir +\
    f"pr_mean_2023.png"
cplt.save_fig(savepath)
