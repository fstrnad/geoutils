# %%
import geoutils.utils.file_utils as fut
import geoutils.geodata.base_dataset as bds
import geoutils.utils.general_utils as gut
import geoutils.plotting.plots as cplt
from importlib import reload


# %%
# MSWEP precipitation
reload(bds)
output_dir = "/home/strnad/data/"
plot_dir = "/home/strnad/data/climnet/plots/"

dataset_file = output_dir + \
    f"/climate_data/mswep_pr_{1}_1979_2021_ds.nc"
grid_step = 1
ds_pr_mswep = bds.BaseDataset(data_nc=dataset_file,
                              grid_step=grid_step,
                              lon_range=[40, 150],
                              lat_range=[-25, 50]
                              )

# %%
# Create Event time series
ds_pr_mswep.create_evs_ds(var_name='pr',
                          q=0.9)
# %%
# Save results as nc file
savepath =  output_dir + \
    f"/evs_data/mswep_pr_{grid_step}_evs.nc"
ds_pr_mswep.save(filepath=savepath,
                 zlib=False)  # zlib for data compression

# %%
# Test if file is stored correctly and read file
ds_pr_evs = bds.BaseDataset(data_nc=savepath,
                              )

# %%
# Plot together EEs and Q-values
reload(cplt)
num_eres = ds_pr_evs.ds.evs.sum(dim='time')
mean_pr = ds_pr_evs.ds.pr.mean(dim='time')
fdic = cplt.create_multi_plot(
    1, 2,
    figsize=(12, 5),
    wspace=0.15,
    projection="PlateCarree")
cplt.plot_map(
    mean_pr,
    ax=fdic["ax"][0],
    plot_type="contourf",
    label=rf"Mean Precipitation [mm/day]",
    orientation="horizontal",
    vmin=0,
    vmax=10,
    levels=12,
    cmap="coolwarm_r",
    tick_step=2,
    round_dec=1,
)

cplt.plot_map(
    num_eres,
    ax=fdic["ax"][1],
    plot_type="contourf",
    label=rf"Number of EREs, $Q_{{pr}}({{{ds_pr_mswep.ds.evs.q}}})$",
    orientation="horizontal",
    vmin=0,
    vmax=1050,
    levels=12,
    tick_step=2,
    round_dec=1,
    cmap="coolwarm_r",
)

output_folder = 'geoutils_tutorial'
savepath = f"{plot_dir}/{output_folder}/ee_and_mean.png"
cplt.save_fig(savepath)
