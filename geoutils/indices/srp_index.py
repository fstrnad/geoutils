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
# 1st EOF of Z200 over the region 35-40°N, 60-70°E
# ======================================================================================


def get_srp_index(z200, timemean=None, idx=0):
    """Returns the cgti index based on the 50hPa zonal winds dataset.

    Args:
        z200 (baseDataset): BaseDataset of GP at 200hPa.
        monthly (boolean): Averages time dimensions to monthly.
            Default to False.
        time_range(list, optional): Select Nino indices only in a given time-range.
            Defauts to None

    Returns:
        cgti_index (xr.Dataset): Nino indices.
    """
    z200.cut_map(lon_range=[30, 130],  # Kosaka et al. 2019
                 lat_range=[20, 60],
                 dateline=False,
                 set_ds=True)

    if timemean is not None:
        cgti_mm = z200.compute_timemean(timemean=timemean)

    rot = 'None'
    pca_ = pca.SpatioTemporalPCA(z200,
                                 var_name='an_dayofyear',
                                 n_components=10,
                                 rotation=rot)

    pca_dict = pca_.get_pca_loc_dict(q=None)

    # Take first principal component (idx = 0)
    srp_index = pca_dict[idx]['ts']
    srp_pattern = pca_dict[idx]['map']

    return {'index': srp_index,
            'map': srp_pattern, }
