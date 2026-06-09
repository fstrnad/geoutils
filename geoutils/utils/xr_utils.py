import xarray as xr
import geoutils.utils.general_utils as gut
import geoutils.utils.time_utils as tu
import geoutils.utils.file_utils as fut
import numpy as np
import geoutils.utils.statistic_utils as sut
from importlib import reload
reload(gut)
reload(tu)
reload(fut)
reload(sut)


def unify_datasets(list_da, delete_non_dim_coords=True):
    common_tr = tu.find_common_time_range(list_da)

    for i, da in enumerate(list_da):
        da = tu.get_time_range_data(ds=da, time_range=common_tr)
        da = gut.delete_all_non_dimension_attributes(dataarray=da)
        new_time = da.time
        if i == 0:
            old_time = new_time
            ds = da.to_dataset()
        else:
            if tu.are_same_time_points(old_time, new_time) is False:
                raise ValueError('Time coordinates are not equal!')
            ds = gut.merge_datasets(ds1=ds, ds2=da.to_dataset())
    return ds


