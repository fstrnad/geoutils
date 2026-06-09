#%%
import cdsapi
import geoutils.preprocessing.open_nc_file as onf
client = cdsapi.Client()
# %%
dataset = 'reanalysis-era5-pressure-levels'
request = {
    'product_type': ['reanalysis'],
    'variable': ['geopotential'],
    'year': ['2024'],
    'month': ['03'],
    'day': ['01'],
    'time': ['13:00'],
    'pressure_level': ['1000'],
    'data_format': 'netcdf',
}
target = 'download_example.nc'

client.retrieve(dataset, request, target)
# %%
# Check the file
onf.open_nc_file(target)