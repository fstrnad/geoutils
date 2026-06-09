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

# %%
reload(sput)
reload(gut)
reload(of)
reload(ald)


def prepare_cutout(features, ds):

    wind_features = [
        # 'wnd100m',
        'wnd10m',
        #  'wnd_azimuth',
        #  'wnd_shear_exp',
        'roughness'
    ]
    influx_features = ['influx_direct', 'influx_diffuse', 'influx',
                       'influx_toa', 'solar_altitude', 'solar_azimuth', 'albedo']
    temp_features = ['temperature']
    features += temp_features  # Temperature is always needed as exra feature
    output_feature_dict = {
        "wind": wind_features,
        "influx": influx_features,
        # "runoff": ['runoff'],
        "temperature": temp_features,
    }

    ds = gut.rename_var_era5(ds=ds, verbose=True)

    if 'wind' in features:
        ds = sput.long_dimension_names(ds)
        ds = ald.get_data_wind(ds=ds,
                               wnd_shear=False,
                               wnd_azimuth=False,
                               retrieval_params=None)
    if 'influx' in features:
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
