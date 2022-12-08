# Geoutils

This is a useful package for processing and plotting geodata provided as nc-files and pre- and postprocessing them accordingly.

### 1. Clone repository:
```
git clone git@github.com:fstrnad/geoutils.git
```

### 2. Installing packages
The following packages are required for running the package:
- xarray 
- pandas
- numpy
- metpy
- windspharm
- cartopy
- scipy
- scikit-learn
The required packages are provided by the condaEnv.yml file.	

To install the package and all its dependencies, we recommend the following steps:

Create a new environment and install all required packages by running: 
```
conda env create -f condaEnv.yml 
conda activate geoutils 
```

### 3. Download data
Download climate data, e.g. from [ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview) server and store the merged files in the data folder.
## Using the package

A tutorial for reading, processing and plotting data can be found at this![tutorial](tutorials/plotting_tutorial.ipynb) .




## TODO
- Better documentation
- Open multiple files into one BaseDataset object

## Contributing 

Any feedback is very welcome. Useful extension proposals are very much appreciated as well. 




