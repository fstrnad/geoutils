[![DOI](https://zenodo.org/badge/574990518.svg)](https://zenodo.org/badge/latestdoi/574990518)

# Geoutils

Geoutils is a lightweight package designed for the efficient processing, analysis, and visualization of geospatial data stored in NetCDF (nc) files. It provides tools for both pre- and post-processing workflows, making it a versatile solution for geoscientific applications. The package leverages the powerful capabilities of the [xarray](https://xarray.pydata.org/) library to handle multi-dimensional arrays and datasets with ease.

### 1. Clone repository:
To clone the repository to your local machine, execute the following command in your terminal:

```bash
git clone git@github.com:fstrnad/geoutils.git
```

This will create a local copy of the Geoutils repository in the current directory. Ensure you have the necessary permissions and SSH keys configured for accessing the repository.


### 2. Installing packages
The code runs with python *3.12.8.* and higher. <br>
We recommend updating your python version to the latest version otherwise some line breaks might cause problems. <br>

The following packages are required for running the package: <br>
- xarray
- zarr
- netcdf4
- cftime
- pandas
- numpy
- matplotlib
- seaborn
- cartopy
- scipy
- scikit-learn
- statsmodels

These packages are not necessary, but needed for some further further functionalities:
- metpy
- windspharm
- palettable (for more colorbars)
- tqdm

To install the packages, we recommend to use a new environment, e.g. by using conda:
```
conda create -n geoutils ipykernel netcdf4 xarray zarr cftime scikit-learn scipy statsmodels matplotlib seaborn -c conda-forge
conda activate geoutils
```
Then you can install the geoutils packages on your own python environment by:
```
pip install -e .
```

### 3. Download climate data
Download climate data, e.g. from [ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview) server and store the merged files in the data folder.
To download the data, you will need to register for an account on the Copernicus Climate Data Store (CDS) and install the `cdsapi` package. Follow these steps:

1. Install the `cdsapi` package:
    ```bash
    pip install cdsapi
    ```

2. Configure the CDS API key:
    After registering on the CDS website, you will receive an API key. Save this key in a file named `.cdsapirc` in your home directory with the following format:
    ```
    url: https://cds.climate.copernicus.eu/api/v2
    key: <your-uid>:<your-api-key>
    ```

3. Use the following Python script to download ERA5 data:
    ```python
    import cdsapi

    c = cdsapi.Client()

    c.retrieve(
         'reanalysis-era5-pressure-levels',
         {
              'product_type': 'reanalysis',
              'format': 'netcdf',
              'variable': 'temperature',
              'pressure_level': '500',
              'year': '2022',
              'month': '01',
              'day': '01',
              'time': '12:00',
         },
         'data/era5_sample.nc'
    )
    ```

Replace the parameters in the script as needed to customize the data you want to download. Ensure the downloaded files are stored in the `data` folder for easy access by the package.

## Using the package

A tutorial for reading, processing and plotting data can be found at this ![tutorial](tutorials/plotting_tutorial.ipynb).

A tutorial for analyzing and processing data can be found ![here](tutorials/analysis_tutorial.ipynb).

An example how to use climate indices is presented in this ![tutorial](tutorials/climate_indices.ipynb).



## TODOs
So far the package is still in development.<br>
The following features are planned to be implemented:
- Add further functionalities for processing data
- Better documentation
- Add further tutorials
- Examples for statistical analysis.
- Add tests for the package

## Contributing
We highly encourage contributions to the Geoutils project. Whether you have identified a bug, have suggestions for new features, or want to improve the documentation, your input is invaluable.

To contribute, please start by opening an issue on the GitHub repository to discuss your proposed changes or enhancements. This ensures alignment with the project's goals and avoids duplication of effort.

If you wish to submit code changes, fork the repository, create a new branch for your feature or fix, and submit a pull request. Please ensure your code adheres to the project's coding standards and includes appropriate tests and documentation updates.

We also welcome feedback on usability, feature requests, and ideas for extending the package's functionality. Your contributions help make Geoutils a better tool for the community.





