# %%
import geoutils.plotting.plots as cplt
import geoutils.utils.statistic_utils as sut
import geoutils.indices.indices_utils as iut
import numpy as np
import xarray as xr
import geoutils.utils.time_utils as tu
from importlib import reload
import pandas as pd
import requests


def get_current_nao_data():
    """
    Get the current NAO data from NOAA
    :return: NAO data as xarray DataArray
    """
    url = 'https://downloads.psl.noaa.gov/Public/map/teleconnections/nao.reanalysis.t10trunc.1948-present.txt'
    # Download the file
    response = requests.get(url).text.splitlines()
    # Read the data into a pandas DataFrame
    data = [line.split() for line in response if line.strip()]
    # Create DataFrame
    df = pd.DataFrame(data, columns=["Year", "Month", "Day", "NAO"])

    # Convert columns to appropriate data types
    df["Year"] = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(int)
    df["Day"] = df["Day"].astype(int)
    df["NAO"] = df["NAO"].astype(float)

    # Create a datetime index
    # Create datetime column
    df["time"] = pd.to_datetime(df[["Year", "Month", "Day"]])

    # Optionally, set Date as index
    df = df[["time", "NAO"]]  # Keep only Date and NAO if preferred
    df.set_index("time", inplace=True)

    ds = xr.Dataset(
        {
            "NAO": (["time"], df["NAO"].values)
        },
        coords={
            "time": df.index
        }
    )

    return ds


if __name__ == '__main__':
    # Get the current NAO data
    nao_index = get_current_nao_data()

    time_range = ['1979-01-01', '1985-12-31']
    nao_index = nao_index.sel(time=slice(*time_range))
    nao_index = tu.compute_timemean(nao_index, timemean='month')
    cplt.plot_2d(x=nao_index.time,
                 y=nao_index.NAO,
                 label_arr=['NAO Index'])
