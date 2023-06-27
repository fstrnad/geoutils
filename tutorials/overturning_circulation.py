# %%
import geoutils.utils.time_utils as tu
import geoutils.indices.enso_utils as eut
import geoutils.geodata.wind_dataset as wds
import geoutils.geodata.base_dataset as bds
import geoutils.utils.spatial_utils as sput
import geoutils.utils.statistic_utils as sut
import geoutils.plotting.plots as cplt

from importlib import reload

output_folder = "circulation"


output_dir = "/home/strnad/data/climnet/outputs/"
plot_dir = "/home/strnad/data/plots/tutorials/"
data_dir = "/home/strnad/data/"


# %%
# Load Wind-Field data
reload(wds)
nc_files_u = []
nc_files_v = []
nc_files_w = []

plevels = [100, 200, 300,
           400, 500, 600,
           700, 800, 900,
           1000]

# plevels = [100, 1000]

for plevel in plevels:
    dataset_file_u = data_dir + \
        f"/climate_data/2.5/era5_u_{2.5}_{plevel}_ds.nc"
    nc_files_u.append(dataset_file_u)
    dataset_file_v = data_dir + \
        f"/climate_data/2.5/era5_v_{2.5}_{plevel}_ds.nc"
    nc_files_v.append(dataset_file_v)
    dataset_file_w = data_dir + \
        f"/climate_data/2.5/era5_w_{2.5}_{plevel}_ds.nc"
    nc_files_w.append(dataset_file_w)

ds_wind = wds.Wind_Dataset(data_nc_u=nc_files_u,
                           data_nc_v=nc_files_v,
                           data_nc_w=nc_files_w,
                           plevels=plevels,
                           can=True,
                           an_types=['month'],
                           month_range=['Dec', 'Feb'],
                           timemean='month',
                           init_mask=False,
                           )
# %%
ds_wind.compute_massstreamfunction()
# %%
# SST data
reload(bds)
dataset_file = data_dir + \
    f"/climate_data/2.5/era5_sst_{2.5}_ds.nc"

ds_sst = bds.BaseDataset(data_nc=dataset_file,
                         can=True,
                         detrend=True,
                         an_types=['JJAS', 'month', 'dayofyear'],
                         )

# %%
# Define ENSO time points
# El Nino La Nina overturning circulations
reload(eut)
reload(tu)
enso_def = 'N3N4'
offset = 0

month_range = ['Dec', 'Mar']

# Get classification of years
enso_classes = eut.get_enso_flavors_obs(
    definition=enso_def,
    ssta=ds_sst.ds['an_month'],
    vname='sst',
    climatology='month',
    month_range=month_range,
    time_range=ds_sst.time_range,
    offset=offset
)

group = 'Nina'
enso_years = eut.get_enso_years(enso_classes=enso_classes,
                                season_types=[group],
                                class_type='type'
                                )
sel_tps = tu.get_dates_of_time_ranges(enso_years, freq='M')


# %%
# Plot for paper multiple pressure levels wind fields
# Vertical Cuts Walker Circulation
reload(cplt)
reload(cplt)
reload(sut)

an_type = 'month'
lon_range_c = [-70, 35]
lat_range_c = [-75, 75]

var_type_wind_u = f'msf_u_an_{an_type}'
var_type_wind_v = f'msf_v_an_{an_type}'
var_type_map_u = f'u_chi_an_{an_type}'
var_type_map_v = f'v_chi_an_{an_type}'
var_type_map_u = f'msf_u_an_{an_type}'
var_type_map_v = f'msf_v_an_{an_type}'

# For MSF Field
plevel_range = [500]
data_cut_hd = sput.cut_map(ds=ds_wind.ds.sel(lev=plevel_range)[
    [var_type_map_u, var_type_map_v]],
    lon_range=lon_range_c,
    lat_range=lat_range_c,
    dateline=True)
data_cut_hd = data_cut_hd.mean(dim='lev')
# %%
# Range data for wind fields
an_type = 'month'
c_lat_range = [-5, 5]
c_lon_range = [120, -80]

u_var_type = f'U_an_{an_type}'
v_var_type = f'V_an_{an_type}'
w_var_type = f'OMEGA_an_{an_type}'
wind_data_lon = sput.cut_map(ds_wind.ds[[u_var_type,
                                        v_var_type,
                                        w_var_type]],
                             lon_range=c_lon_range,
                             lat_range=c_lat_range, dateline=True)

range_data_lon = sput.cut_map(ds_wind.ds[[var_type_wind_u,
                                          var_type_wind_v]],
                              lon_range=c_lon_range,
                              lat_range=c_lat_range, dateline=True)


# %%
# Stream function background plot
reload(cplt)
var_type = f'an_{an_type}'
label_msf_u = r'$\Psi_u$ monthly-anomalies [$\frac{kg}{m^2s}$]'
label_vert_u = r'$\bar{\Psi}_u$  monthly-anomalies [$\frac{kg}{m^2s}$]'

vmax_vert = 1e11
vmin_vert = -vmax_vert
vmax_map = 1e11
vmin_map = -vmax_map

lon_ticks = [60, 120, 180, 240]
lon_ticklabels = ['60°E', '120°E', '180°', '120°W']

sci = 11
sci_map = 11
ncols = 4

# Wind fields
k_data_u = tu.get_sel_tps_ds(
    ds=wind_data_lon[u_var_type], tps=sel_tps).mean(dim=['lat', 'time'])
k_data_uw = tu.get_sel_tps_ds(
    ds=wind_data_lon[w_var_type], tps=sel_tps).mean(dim=['lat', 'time'])
vert_data_lon = tu.get_sel_tps_ds(
    ds=range_data_lon[var_type_wind_u], tps=sel_tps).mean(dim=['lat', 'time'])


lons = vert_data_lon.lon
plevels = vert_data_lon.lev

h_im_u = cplt.plot_2D(x=lons, y=plevels,
                      z=vert_data_lon.T,
                      levels=9,
                      title=f'Walker Circulation ({group})',
                      cmap='RdBu',
                      tick_step=3,
                      plot_type='contourf',
                      extend='both',
                      vmin=vmin_vert, vmax=vmax_vert,
                      xlabel='Longitude [degree]',
                      ylabel='Pressure Level [hPa]',
                      flip_y=True,
                      label=label_vert_u
                    #   xticks=lon_ticks,
                    #   xticklabels=lon_ticklabels
                      )

dict_w = cplt.plot_wind_field(
    ax=h_im_u['ax'],
    u=k_data_u.T,
    v=k_data_uw.T*-100,
    x_vals=k_data_uw.lon,
    y_vals=k_data_uw.lev,
    steps=1,
    x_steps=4,
    transform=False,
    scale=50,
    key_length=2,
    wind_unit='m/s | 0.02 hPa/s',
    key_loc=(0.95, 1.05)
)

savepath = plot_dir +\
    f"{output_folder}/msf_vertical_cut_lon.png"
cplt.save_fig(savepath, fig=h_im_u['fig'])

# %%
reload(cplt)
this_comp_ts = tu.get_sel_tps_ds(
    data_cut_hd, tps=sel_tps)

mean, pvalues_ttest = sut.ttest_field(
    this_comp_ts[var_type_map_u], data_cut_hd[var_type_map_u])
mask = sut.field_significance_mask(
    pvalues_ttest, alpha=0.05, corr_type=None)

im_wind_u = cplt.plot_map(mean,
                          title=r'400hPa-600hPa $\bar{\Psi}_u$',
                          y_title=1.32,
                          cmap='RdBu',
                          plot_type='contourf',
                          levels=9,
                          tick_step=3,
                          vmin=vmin_map, vmax=vmax_map,
                          extend='both',
                          orientation='horizontal',
                          significance_mask=mask,
                          hatch_type='..',
                          plt_grid=True,
                          label=label_msf_u,
                          )

savepath = plot_dir +\
    f"{output_folder}/msf_map.png"
cplt.save_fig(savepath, fig=im_wind_u['fig'])

# %%
