# %%
import atlite.datasets.era5 as ald
import atlite as at
import xarray as xr
import numpy as np
import os
import geoutils.utils.file_utils as fut
import geoutils.utils.general_utils as gut
import geoutils.utils.spatial_utils as sput
import geoutils.utils.time_utils as tu
import geoutils.preprocessing.open_nc_file as of
import geoutils.plotting.plots as gplt

from importlib import reload

if os.getenv("HOME") == '/home/goswami/fstrnad80':
    data_dir = "/mnt/qb/goswami/data/era5/weatherbench2"
    output_dir = "/mnt/qb/datasets/STAGING/goswami/weatherbench2/Europe"
else:
    data_dir = "/home/strnad/data/dunkelflauten/"
    output_dir = "/home/strnad/data/climate_data/Europe"

# %%
requested_era5_flux = [
    "surface_net_solar_radiation",
    "surface_solar_radiation_downwards",
    "toa_incident_solar_radiation",
    "total_sky_direct_solar_radiation_at_surface",
]

requested_era5_temp = [
    "2m_temperature",
    # "soil_temperature_level_4",
    # "2m_dewpoint_temperature",
]

requested_era5_wind = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "forecast_surface_roughness",
]

requested_static_era5 = [
    'geopotential',

]

feature_dict = {
    "wind": requested_era5_wind,
    "temperature": requested_era5_temp,
    "influx": requested_era5_flux,
    "static": requested_static_era5,
    'runoff': ['runoff']
}


requested_vars = requested_era5_flux + requested_era5_temp + \
    requested_era5_wind + requested_static_era5

grid_step = 1
country_name = "Germany"

features = ['runoff', 'wind', 'influx', 'temperature']

variables = []
for feature in features:
    variables += feature_dict[feature]
files = []
for variable in variables:
    new_file = f'{output_dir}/{country_name}/{grid_step}/{variable}_{grid_step}.nc'
    if fut.exist_file(new_file):
        gut.myprint(f'File {new_file} exists!')
        files.append(new_file)
    else:
        gut.myprint(f'File {new_file} does not exist!')

# %%
reload(sput)
reload(gut)
reload(of)

wind_features = ['wnd100m', 'wnd_azimuth', 'wnd_shear_exp',
                 'roughness']
influx_features = ['influx_direct', 'influx_diffuse',
                   'influx_toa', 'solar_altitude', 'solar_azimuth', 'albedo']
temp_features = ['temperature']

output_feature_dict = {
    "wind": wind_features,
    "influx": influx_features,
    "runoff": ['runoff'],
    "temperature": temp_features,
}


def prepare_input_cutout(features, files):
    ds = of.open_nc_file(files)
    ds = sput.long_dimension_names(ds)

    ds = gut.rename_var_era5(ds=ds, verbose=True)

    reload(ald)
    ds = ald.get_data_wind(ds=ds, retrieval_params=None)
    ds = sput.long_dimension_names(ds)
    ds = ald.get_data_influx(ds=ds, retrieval_params=None)

    # Add attributes according to naming convention of atlite
    ds = gut.add_attribute(ds=ds, attribute_name='prepared_features',
                           attribute_value=features,
                           )
    ds = gut.add_attribute(ds=ds, attribute_name='module',
                           attribute_value='era5',
                           var_names='all')
    for feature in features:
        var_names = output_feature_dict[feature]
        ds = gut.add_attribute(ds=ds, attribute_name='feature',
                               attribute_value=feature,
                               var_names=var_names)

    return ds


ds = prepare_input_cutout(features, files)

ds
# %%
reload(fut)
savepath = f'{data_dir}/{country_name}/pv_wind_{grid_step}.nc'
fut.save_ds(ds, savepath)


# %%
# Test ds as input for cutout
cutout_wind = at.Cutout(savepath)
cutout_wind
# %%
turbine = "Vestas_V112_3MW"
cap_factors = cutout_wind.wind(turbine=turbine, capacity_factor=True)

# %%
im = gplt.plot_map(cap_factors, title="Cap. Factor Wind", set_borders=True)
