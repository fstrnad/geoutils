[![DOI](https://zenodo.org/badge/574990518.svg)](https://zenodo.org/badge/latestdoi/574990518)

# Geoutils

Geoutils is a lightweight package designed for the efficient processing, analysis, and visualization of geospatial data stored in NetCDF (nc) files. It provides tools for both pre- and post-processing workflows, making it a versatile solution for geoscientific applications. The package leverages the powerful capabilities of the [xarray](https://xarray.pydata.org/) library to handle multi-dimensional arrays and datasets with ease.

### 1. Clone repository:
To clone the repository to your local machine, execute the following command in your terminal:

```bash
git clone https://github.com/fstrnad/geoutils.git
```

This will create a local copy of the Geoutils repository in the current directory. Ensure you have the necessary permissions and SSH keys configured for accessing the repository.


### 2. Installation

Geoutils requires **Python 3.12 or higher**. Create an isolated virtual environment inside the repository and install the package in editable mode:

```bash
cd geoutils
python3.12 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```

Or, if you use [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

All dependencies are declared in `pyproject.toml`. The core install pulls in `numpy`, `scipy`, `pandas`, `xarray`, `netcdf4`, `cftime`, `zarr`, `dask`, `matplotlib`, `cartopy`, `seaborn`, `scikit-learn`, `statsmodels`, `joblib`, `tqdm` and `palettable`.

#### Optional dependencies

Install extras as needed, e.g. `pip install -e ".[met,download]"`:

| Extra | Adds | For |
|---|---|---|
| `met` | `metpy`, `windspharm` | meteorological calculations, spherical-harmonic wind analysis |
| `download` | `cdsapi`, `requests` | downloading ERA5 and other data (`geoutils.geodata.downloader`) |
| `energy` | `atlite`, `geopandas`, `pgeocode` | countries, energy and cutouts subpackages |
| `viz` | `cmocean`, `cmweather` | extra colormaps |
| `video` | `imageio`, `opencv-python` | animations (`geoutils.plotting.videos`) |
| `notebook` | `jupyter`, `ipython`, `ipywidgets` | running the tutorials |
| `dev` | `pytest`, `ruff` | tests and linting |

To use the environment in Jupyter, install the `notebook` extra and register the kernel:

```bash
pip install -e ".[notebook]"
python -m ipykernel install --user --name geoutils
```

#### Verify the installation

```bash
pip install -e ".[dev]"
pytest
```


### 3. Download climate data
Download climate data, e.g. from [ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview) server and store the merged files in the data folder.
To download the data, you will need to register for an account on the Copernicus Climate Data Store (CDS) and install the `cdsapi` package. Follow these steps:

1. Install the `cdsapi` package (included in the `download` extra):
    ```bash
    pip install -e ".[download]"
    ```

2. Configure the CDS API key:
    After registering on the CDS website, you will find a personal access token on your CDS profile page. Save this key in a file named `.cdsapirc` in your home directory with the following format:
    ```
    url: https://cds.climate.copernicus.eu/api
    key: <your-personal-access-token>
    ```

3. Use the following Python script to download ERA5 data:
    ```python
    import cdsapi

    c = cdsapi.Client()

    c.retrieve(
         'reanalysis-era5-pressure-levels',
         {
              'product_type': 'reanalysis',
              'data_format': 'netcdf',
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

A tutorial for reading, processing and plotting data can be found in this [tutorial](tutorials/plotting_tutorial.ipynb).

Scripts for the plotting API using ERA5 data are in [examples/](examples/README.md).

A tutorial for analyzing and processing data can be found [here](tutorials/analysis_tutorial.ipynb).

An example how to use climate indices is presented in this [tutorial](tutorials/climate_indices.ipynb).



## TODOs
So far the package is still in development.<br>
The following features are planned to be implemented:
- Add further functionalities for processing data
- Better documentation
- Add further tutorials
- Examples for statistical analysis.
- Extend the test suite

## Contributing
We highly encourage contributions to the Geoutils project. Whether you have identified a bug, have suggestions for new features, or want to improve the documentation, your input is invaluable.

To contribute, please start by opening an issue on the GitHub repository to discuss your proposed changes or enhancements. This ensures alignment with the project's goals and avoids duplication of effort.

If you wish to submit code changes, fork the repository, create a new branch for your feature or fix, and submit a pull request. Please ensure your code adheres to the project's coding standards and includes appropriate tests and documentation updates. Install the `dev` extra, then run `pytest` and `ruff check` on the files you changed before opening the pull request.

We also welcome feedback on usability, feature requests, and ideas for extending the package's functionality. Your contributions help make Geoutils a better tool for the community.





