# %%
import geoutils.plotting.plots as cplt
import geoutils.utils.general_utils as gut
import geoutils.geodata.base_dataset as bds
import geoutils.utils.general_utils as gut
import geoutils.utils.file_utils as fut
import geoutils.preprocessing.open_nc_file as of
import xarray as xr

from importlib import reload

output_dir = "/home/strnad/data/"
data_dir = "/home/strnad/data/"
# %%
reload(fut)
reload(of)
files = fut.get_files_in_folder(
    data_dir + "era5/single_pressure_level/100m_u_component_of_wind"
)
open_files = files[:4]
# %%
ds = of.open_ds(nc_files=open_files, decode_times=True, transpose_dims=False)
