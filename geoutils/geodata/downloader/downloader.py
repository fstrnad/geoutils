# %%
import geoutils.geodata.downloader.download_era5 as d5
import geoutils.utils.general_utils as gut
import geoutils.utils.time_utils as tu
from importlib import reload

# %%
reload(d5)

d5.download_era5(variable='pr',
                 starty=2023,
                 endy=2023,
                 start_month='Jun',
                 end_month='Sep',
                 folder='/home/strnad/data/era5/',  # full path
                 run=True,
                 )

