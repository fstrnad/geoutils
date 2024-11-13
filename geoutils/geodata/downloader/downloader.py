# %%
import geoutils.geodata.downloader.download_era5 as d5
import geoutils.utils.general_utils as gut
import geoutils.utils.time_utils as tu
import geoutils.preprocessing.open_nc_file as onf
from importlib import reload

# %%
reload(d5)
variable = 'surface_net_solar_radiation'
starty = 2023
endy = 2024
start_month = 'Jun'
end_month = 'Jun'
times = '6h'
dir = '/home/strnad/data/era5/'

d5.download_era5(variable=variable,
                 starty=starty,
                 endy=endy,
                 start_month=start_month,
                 end_month=end_month,
                 times=times,
                 folder=dir,  # full path
                 run=True,
                 plevel=None,
                 daymean=True,
                 )
# %%
reload(onf)
year = 2023
tstr = times
filename = f'{dir}/single_pressure_level/{variable}/{variable}_{year}_{start_month}_{end_month}_{tstr}.nc'
fname_daymean = f'{dir}/single_pressure_level/{variable}/{variable}_{year}_{start_month}_{end_month}_{tstr}_daymean.nc'

file = onf.open_nc_file(filename)
# fdaymean = onf.open_nc_file(fname_daymean)