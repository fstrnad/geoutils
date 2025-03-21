[![DOI](https://zenodo.org/badge/574990518.svg)](https://zenodo.org/badge/latestdoi/574990518)

# Geoutils

This is a small package for processing and plotting geodata provided as nc-files and pre- and postprocessing them accordingly.
It is mostly built upon the xarray package.

### 1. Clone repository:
```
git clone git@github.com:fstrnad/geoutils.git
```

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
-

## Contributing
We highly encourage contributions to the Geoutils project. Whether you have identified a bug, have suggestions for new features, or want to improve the documentation, your input is invaluable.

To contribute, please start by opening an issue on the GitHub repository to discuss your proposed changes or enhancements. This ensures alignment with the project's goals and avoids duplication of effort.

If you wish to submit code changes, fork the repository, create a new branch for your feature or fix, and submit a pull request. Please ensure your code adheres to the project's coding standards and includes appropriate tests and documentation updates.

We also welcome feedback on usability, feature requests, and ideas for extending the package's functionality. Your contributions help make Geoutils a better tool for the community.





