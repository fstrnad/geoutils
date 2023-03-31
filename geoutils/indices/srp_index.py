import numpy as np
import xarray as xr
import geoutils.utils.statistic_utils as sut
import geoutils.utils.spatial_utils as sput
import geoutils.utils.general_utils as gut
import geoutils.utils.time_utils as tut
import geoutils.tsa.pca.pca as pca

from importlib import reload

reload(tut)

# ======================================================================================
# Compute the  Silk Road Pattern index (SRP) as the
# 1st EOF of v200 over the region 20-60°N, 30-130°E
# ======================================================================================


def get_srp_index(v200, var_name='an_dayofyear', timemean=None, idx=0):
    """Returns the cgti index based on the 50hPa zonal winds dataset.

    Args:
        v200 (baseDataset): BaseDataset of meridional wind at 200hPa.
        monthly (boolean): Averages time dimensions to monthly.
            Default to False.
        time_range(list, optional): Select Nino indices only in a given time-range.
            Defauts to None

    Returns:
        cgti_index (xr.Dataset): Nino indices.
    """
    v200.cut_map(lon_range=[30, 130],  # Kosaka et al. 2009
                 lat_range=[20, 60],
                 dateline=False,
                 set_ds=True)

    rot = 'None'
    pca_ = pca.SpatioTemporalPCA(v200,
                                 var_name=var_name,
                                 n_components=10,
                                 rotation=rot)

    pca_dict = pca_.get_pca_loc_dict(q=None)

    # Take first principal component (idx = 0)
    srp_index = pca_dict[idx]['ts']
    srp_pattern = pca_dict[idx]['map']

    return {'index': srp_index,
            'map': srp_pattern, }


# ======================================================================================
# Compute the  Silk Road Pattern index (SRP) as the
# 1st EOF of v200 over the region 20-60°N, 30-180°E
# ======================================================================================


def get_srp_index_gph(z200, var_name='an_dayofyear', timemean=None, idx=0):
    """Returns the cgti index based on the 50hPa zonal winds dataset.

    Args:
        z200 (baseDataset): BaseDataset of meridional wind at 200hPa.
        monthly (boolean): Averages time dimensions to monthly.
            Default to False.
        time_range(list, optional): Select Nino indices only in a given time-range.
            Defauts to None

    Returns:
        cgti_index (xr.Dataset): Nino indices.
    """
    z200.cut_map(lon_range=[30, 180],  # Kosaka et al. 2009
                 lat_range=[20, 60],
                 dateline=False,
                 set_ds=True)

    rot = 'None'
    pca_ = pca.SpatioTemporalPCA(z200,
                                 var_name=var_name,
                                 n_components=10,
                                 rotation=rot)

    pca_dict = pca_.get_pca_loc_dict(q=None)

    # Take first principal component (idx = 0)
    srp_index = pca_dict[idx]['ts']
    srp_pattern = pca_dict[idx]['map']

    return {'index': srp_index,
            'map': srp_pattern, }