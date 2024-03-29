[![DOI](https://zenodo.org/badge/574990518.svg)](https://zenodo.org/badge/latestdoi/574990518)

# Geoutils

This is a small package for processing and plotting geodata provided as nc-files and pre- and postprocessing them accordingly.
It is mostly build upon the xarray package.

### 1. Clone repository:
```
git clone git@github.com:fstrnad/geoutils.git
```

### 2. Installing packages
The following packages are required for running the package:
- xarray
- netcdf4
- pandas
- numpy
- metpy
- windspharm
- matplotlib
- seaborn
- cartopy
- scipy
- scikit-learn
- statsmodels
- tqdm
- palettable (for more colorbars)

The required packages are provided by the condaEnv.yml file.

To install the package and all its dependencies, we recommend the following steps:

Create a new environment and install all required packages by running:
```
conda env create -f condaEnv.yml
conda activate geoutils
```

Otherwise you can install the required packages on your own python environment and install the geoutils package by:
```
pip install -e .
```

### 3. Download data
Download climate data, e.g. from [ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview) server and store the merged files in the data folder.
## Using the package

A tutorial for reading, processing and plotting data can be found at this ![tutorial](tutorials/plotting_tutorial.ipynb).

A tutorial for analyzing and processing data can be found ![here](tutorials/analysis_tutorial.ipynb).

An example how to use climate indices is presented in this ![tutorial](tutorials/climate_indices.ipynb).



## TODO
- Better documentation
- Add further tutorials
- Examples for statistical analysis.

## Contributing

Any feedback is very welcome. Useful extension proposals are very much appreciated as well.




