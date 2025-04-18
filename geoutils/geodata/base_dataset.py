# %%
"""
Base class for the geodata datasets with lon-lat resolution.
"""

import os
import numpy as np
import copy
import geoutils.preprocessing.open_nc_file as of
import geoutils.utils.general_utils as gut
import geoutils.utils.file_utils as fut
import geoutils.utils.time_utils as tu
import geoutils.utils.spatial_utils as sput
from importlib import reload
import xarray as xr
reload(gut)
reload(fut)
reload(sput)
reload(tu)
reload(of)
PATH = os.path.dirname(os.path.abspath(__file__))


class BaseDataset():
    """Class that defines a classical rectangular dataset, that is stored as as classical
    nc file. It has however the same functions of BaseDataset but is not defined of the
    grid of BaseDataset, but on the standard lon, lat grid that is by default used in nc-files.
    i.e. for different longitude and latitude boxes
    """

    def __init__(
        self,
        data_nc=None,
        var_name=None,
        time_range=None,
        lon_range=[-180, 180],
        lat_range=[-90, 90],
        grid_step=None,
        large_ds=False,
        can=False,
        detrend=False,
        month_range=None,
        lsm_file=None,
        decode_times=True,
        verbose=True,
        metpy_labels=False,  # labelling according to metpy convention
        metpy_unit=None,
        parse_cf=False,
        **kwargs,
    ):
        """Initializes a BaseDataset object with an nc file provided.
        The file can be a blank nc file (data_nc) or as an already processed file (load_nc).

        Args:
            var_name (str, optional): Name of input variables. Defaults to None.
            data_nc (str, optional): Nc file to be processed. Defaults to None.
            load_nc (str, optional): Nc file already processed. Defaults to None.
            time_range (tuple(str, str), optional): Tuple of start and end time YY-MM-DD formatt. Defaults to None.
            lon_range (tuple(float, Float), optional): Input range of longitude. Defaults to [-180, 180].
            lat_range (tuple(float, float), optional): Input range of latitude. Defaults to [-90, 90].
            grid_step (int, optional): Grid step distance of points on lon-lat grid. Defaults to 1.
            large_ds (bool, optional): Is the input file very large? Defaults to False.
            can (bool, optional): Compute Anomalies. Defaults to False.
            detrend (bool, optional): Detrend the data. Defaults to False.
            month_range (tuple(str, str), optional): Month range of data.  Defaults to None.
            lsm_file (str, optional): additional land sea mask file. Defaults to None.
            decode_times (bool, optional): decode times if in nc file np.datetime64 format is provided. Defaults to True.
        """
        self.verbose = verbose
        if isinstance(data_nc, list) or isinstance(data_nc, np.ndarray):
            data_nc_arr = data_nc
        elif isinstance(data_nc, str):
            data_nc_arr = [data_nc]
        else:
            raise ValueError(
                f'Provide single or multiple files as strings but data_nc = {data_nc}!')

        read_into_memory = kwargs.pop('read_into_memory', True)
        self.grid_step = grid_step
        self.set_dim_names()
        ds = self.open_ds(
            nc_files=data_nc_arr,
            var_name=var_name,  # if var_name given only this variable is read
            lat_range=lat_range,
            lon_range=lon_range,
            time_range=time_range,
            month_range=month_range,
            grid_step=grid_step,
            large_ds=large_ds,
            decode_times=decode_times,
            verbose=verbose,
            parse_cf=parse_cf,
            **kwargs,
        )
        (
            self.time_range,
            self.lon_range,
            self.lat_range,
        ) = self.get_spatio_temp_range(ds)

        self.set_var(var_name=var_name, ds=ds, verbose=verbose)

        self.ds = ds

        # detrending
        if detrend is True:
            detrend_from = kwargs.pop('detrend_from', None)
            self.detrend(dim="time", startyear=detrend_from)

        if metpy_unit is not None:
            self.ds = self.set_metpy_units(metpy_unit)

        # Compute Anomalies if needed
        self.can = can
        if self.can is True:
            self.compute_anomalies_ds(**kwargs)

        # Filter nans
        nan_filter = kwargs.pop('nan_filter', None)
        if nan_filter is not None:
            self.ds = self.filter_nans(th=nan_filter)
        # Init Mask
        self.init_mask(da=self.ds[self.var_name], lsm_file=lsm_file,
                       verbose=verbose, **kwargs)
        self.set_source_attrs(verbose=verbose)

        self.set_ds_objects()

        if read_into_memory:
            self.ds = self.read_data_into_memory()

        if metpy_labels:
            self.ds = self.set_metpy_labels()

    def set_ds_objects(self):
        if 'time' in self.get_dims():
            self.time = self.ds.time
        self.coords = self.ds.coords
        self.loc_dict = dict()
        self.grid_type = 'rectangular'

    def open_ds(
        self,
        nc_files,
        plevels=None,
        time_range=None,
        month_range=None,
        grid_step=None,
        lon_range=[-180, 180],
        lat_range=[-90, 90],
        use_ds_grid=False,
        decode_times=True,
        verbose=True,
        parse_cf=False,
        var_name=None,
        **kwargs,
    ):
        reload(gut)
        reload(of)
        self.plevel_name = kwargs.pop('plevel_name', 'lev')

        ds = of.open_nc_file(nc_files=nc_files,
                             plevels=plevels,
                             decode_times=decode_times,
                             verbose=verbose,
                             var_name=var_name,
                             plevel_name=self.plevel_name,
                             **kwargs)#.compute()
        self.dims = self.get_dims(ds=ds)

        if 'time' in self.dims:
            ds = self.get_data_timerange(ds, time_range, verbose=verbose)

        if month_range is not None:
            gut.myprint(f'Get month range data {month_range}!')
            ds = tu.get_month_range_data(dataset=ds,
                                         start_month=month_range[0],
                                         end_month=month_range[1],
                                         verbose=verbose)

        if 'lon' in self.dims and 'lat' in self.dims:
            min_lon = kwargs.pop('min_lon', None)
            max_lon = kwargs.pop('max_lon', None)
            min_lat = kwargs.pop('min_lat', None)
            max_lat = kwargs.pop('max_lat', None)
            grid_step_lon = kwargs.pop('grid_step_lon', None)
            grid_step_lat = kwargs.pop('grid_step_lat', None)

            if grid_step_lat is not None and grid_step_lon is None:
                gut.myprint(
                    f'Grid_step_lon not specified, but grid_step_lat is!',
                    verbose=verbose)
                grid_step_lon = grid_step_lat
            if grid_step_lon is not None and grid_step_lat is None:
                gut.myprint(
                    f'Grid_step_lat not specified, but grid_step_lon is!',
                    verbose=verbose)
                grid_step_lat = grid_step_lon

            if grid_step_lat is not None or grid_step_lon is not None:
                grid_step = 1  # Not to be None

            if grid_step is not None:
                ds = self.common_grid(dataarray=ds, grid_step=grid_step,
                                      min_lon=min_lon, max_lon=max_lon,
                                      min_lat=min_lat, max_lat=max_lat,
                                      grid_step_lon=grid_step_lon,
                                      grid_step_lat=grid_step_lat,
                                      use_ds_grid=use_ds_grid)

            if lon_range != [-180, 180] or lat_range != [-90, 90]:
                gut.myprint(
                    f'Cut the dataset {lon_range}, {lat_range}!', verbose=verbose)
                ds = self.cut_map(ds, lon_range, lat_range)

            self.grid_step, self.grid_step_lon, self.grid_step_lat = sput.get_grid_step(
                ds=ds, verbose=verbose)

        # ds.unify_chunks()
        ds.compute()
        self.info_dict = copy.deepcopy(ds.attrs)

        timemean = kwargs.pop('timemean', None)
        if timemean is not None:
            ds = tu.compute_timemean(ds=ds, timemean=timemean, verbose=verbose)
        timemax = kwargs.pop('timemax', None)
        if timemax is not None:
            ds = tu.compute_timemax(ds=ds, timemean=timemax, verbose=verbose)

        if parse_cf:
            import geoutils.utils.met_utils as mut
            ds = mut.parse_cf(ds=ds)

        delete_hist = kwargs.pop('delete_hist', False)
        if delete_hist:
            ds = gut.delete_ds_attr(ds=ds, attr='history')
        gut.myprint("Finished processing data", verbose=verbose)

        return ds

    def read_data_into_memory(self):
        gut.myprint(f'Attention! All data is read into memory!',
                    color='red', bold=True)
        self.ds = self.ds.compute()
        gut.myprint(f'Finished reading data into memory!',
                    verbose=self.verbose)
        return self.ds

    def load(self, load_nc, lon_range=[-180, 180], lat_range=[-90, 90],
             verbose=True):
        """Load dataset object from file.

        Parameters:
        ----------
        ds: xr.Dataset
            Dataset object containing the preprocessed dataset

        """
        # check if file exists
        if not os.path.exists(load_nc):
            PATH = os.path.dirname(os.path.abspath(__file__))
            gut.myprint(f"You are here: {PATH}!")
            raise ValueError(f"File does not exist {load_nc}!")

        gut.myprint(f"Load Dataset: {load_nc}")
        ds = xr.open_dataset(load_nc)
        ds = self.check_dimensions(ds)
        ds = self.cut_map(ds=ds, lon_range=lon_range, lat_range=lat_range)
        self.time_range, self.lon_range, self.lat_range = self.get_spatio_temp_range(
            ds)
        ds_attrs = list(ds.attrs.keys())
        if "grid_step" in ds_attrs:
            self.grid_step = ds.attrs["grid_step"]
        self.info_dict = copy.deepcopy(ds.attrs)  # TODO

        # Read and create grid class
        ds = gut.rename_var_era5(ds=ds, verbose=verbose)
        self.vars = self.get_vars(ds=ds, verbose=True)
        self.dims = self.get_dims(ds=ds)

        # mask = np.ones_like(ds[name][0].data, dtype=bool)
        # for idx, t in enumerate(ds.time):
        #     mask *= np.isnan(ds[name].sel(time=t).data)

        # self.mask = xr.DataArray(
        #     data=xr.where(mask == 0, 1, np.NaN),
        #     dims=da.sel(time=da.time[0]).dims,
        #     coords=da.sel(time=da.time[0]).coords,
        #     name="mask",
        # )
        self.ds = self.check_time(ds)

        return None

    def save(self, filepath, save_params=True,
             unlimited_dim=None,
             var_list=None,
             classic_nc=False,
             zlib=False):
        """Save the dataset class object to file.
        Args:
        ----
        filepath: str
        """
        if var_list is None:
            ds_temp = self.ds
        else:
            gut.myprint(f'Save variables {var_list}!')
            ds_temp = self.ds[list(var_list)]
        if save_params and self.grid_step is not None:
            if len(ds_temp.attrs) == 0:
                gut.myprint(f'Attention, attribute list is empty!')
                param_class = {
                    "grid_step": self.grid_step,
                    "grid_type": self.grid_type,
                }
                ds_temp.attrs = param_class

        fut.save_ds(ds=ds_temp, filepath=filepath,
                    unlimited_dim=unlimited_dim,
                    classic_nc=classic_nc,
                    zlib=zlib)

        return None

    def save_mask(self, filepath, set_nan=False):
        """Save the dataset class object to file.
        Args:
        ----
        filepath: str
        set_nan: bool  Set if mask should contain 1 and nans.
        """
        param_class = {
            "grid_step": self.grid_step,
        }
        ds_temp = self.mask
        if set_nan:
            gut.myprint('0 in Mask are set to nan for storing in array!')
            ds_temp = xr.where(self.mask, 1, np.nan)
        ds_temp.attrs = param_class

        fut.save_ds(ds=ds_temp, filepath=filepath)

        return None

    # ############# functions for opening and processing input data ############
    def check_dimensions(self, ds, verbose=True, **kwargs):
        """
        Checks whether the dimensions are the correct ones for xarray!
        """
        reload(sput)
        sort = kwargs.pop('sort', True)
        self.lon360 = kwargs.pop('lon360', False)
        ts_days = kwargs.pop('ts_days', True)
        keep_time = kwargs.pop('keep_time', False)

        freq = kwargs.pop('freq', 'D')
        ds = sput.check_dimensions(ds=ds,
                                   ts_days=ts_days,
                                   lon360=self.lon360,
                                   sort=sort,
                                   keep_time=keep_time,
                                   freq=freq,
                                   verbose=verbose)
        # Set time series to days
        if len(list(ds.dims)) > 2:
            ds = self.check_time(ds, **kwargs)

        return ds

    def set_dim_names(self):
        self.lon_name = 'lon'
        self.lat_name = 'lat'
        self.time_name = 'time'
        self.lev_name = 'lev'
        self.info_dict = {}

    def check_time(self, ds, **kwargs):
        """Sets the respective calender!

        Args:
            ds (xr.dataset): dataset

        Returns:
            xr.dataset: dataset
        """
        ts_days = kwargs.pop("ts_days", True)
        if ts_days:
            if not gut.is_datetime360(time=ds.time.data[0]):
                self.calender360 = False
            else:
                gut.myprint('WARNING: 360 day calender is used!',
                            color='yellow')
                self.calender360 = True
        else:
            gut.myprint('WARNING! No standard calender is used!',
                        color='yellow')
            self.calender360 = False
        return ds

    def rename_var(self, new_var_name, old_var_name=None, ds=None):
        """Renames the dataset's variable name in self.ds and self.var_name

        Args:
            var_name (str): string of new var name
        """
        if ds is None:
            ds = self.ds
            set_ds = True
        else:
            set_ds = False
        if old_var_name is None:
            old_var_name = self.var_name
        if old_var_name in self.get_dims():
            if old_var_name == 'lon' or old_var_name == 'lat':
                if old_var_name == 'lon':
                    self.lon_name = new_var_name
                else:
                    self.lat_name = new_var_name
        elif old_var_name not in self.get_vars(ds=ds):
            raise ValueError(
                f'This variable {old_var_name} does not exist in dataset!')
        ds = ds.rename({old_var_name: new_var_name})
        if set_ds:
            self.ds = ds
        gut.myprint(f"Rename {old_var_name} to {new_var_name}!")
        self.vars = self.get_vars()
        self.set_var(verbose=False)
        return ds

    def delete_var(self, var_name):
        vars = self.get_vars()
        if var_name not in vars:
            raise ValueError(
                f'Variable {var_name} not in dataset. Available variables are {vars}')
        else:
            self.ds = self.ds.drop_vars(names=var_name)
            gut.myprint(f'Deleted variable: {var_name}!')
        self.set_var()

    def get_vars(self, ds=None, verbose=False):
        # vars = []
        # for name, da in self.ds.data_vars.items():
        #     vars.append(name)
        if ds is None:
            ds = self.ds
        vars = gut.get_vars(ds=ds)
        if verbose:
            gut.myprint(f'Variables in dataset: {vars}!')
        return vars

    def set_var(self, ds=None, var_name=None, verbose=True):
        # select a main var name
        self.vars = self.get_vars(ds=ds)
        var_name = var_name if var_name is not None else self.vars[0]
        self.var_name = 'evs' if 'evs' in self.vars else var_name
        if self.var_name not in self.vars:
            raise ValueError(
                f'{var_name} not in variables available {self.vars}!')
        gut.myprint(f'Set variable name to {self.var_name}!', verbose=verbose)
        self.get_source_attrs(ds=ds)

    def get_source_attrs(self, ds=None):
        """
        Get the attributes of the source dataset
        """
        if ds is None:
            ds = self.ds
        self.source_attrs = ds.attrs
        self.var_attrs = {}
        for var in self.vars:
            self.var_attrs[var] = ds[var].attrs
        self.lon_attrs = ds[self.lon_name].attrs
        self.lat_attrs = ds[self.lat_name].attrs
        self.time_attrs = None
        self.lon_attrs['standard_name'] = self.lon_attrs['long_name'] = 'longitude'
        self.lon_attrs['units'] = 'degree_east'
        self.lon_attrs['axis'] = 'X'
        self.lat_attrs['standard_name'] = self.lat_attrs['long_name'] = 'latitude'
        self.lat_attrs['units'] = 'degree_north'
        self.lat_attrs['axis'] = 'Y'

        dims = self.get_dims(ds=ds)
        if 'time' in dims:
            self.time_attrs = ds.time.attrs
            self.time_attrs['standard_name'] = self.time_attrs['long_name'] = 'time'
            self.time_attrs['axis'] = 'T'

    def set_source_attrs(self, verbose=True):
        gut.myprint(f'Set dataset source attributes!', verbose=verbose)
        if self.source_attrs is None:
            raise ValueError('Source attributes is not set yet!')
        self.ds.attrs.update(self.source_attrs)
        for var in list(self.var_attrs.keys()):
            self.ds[var].attrs.update(self.var_attrs[var])
        self.ds[self.lon_name].attrs.update(self.lon_attrs)
        self.ds[self.lat_name].attrs.update(self.lat_attrs)
        if 'time' in self.get_dims():
            self.ds.time.attrs.update(self.time_attrs)

    def add_var_attribute(self, var_dict):
        for key, val in var_dict.items():
            gut.myprint(f'Added {key}: {val} to {self.var_name} attributes!')
            self.ds[self.var_name].attrs[key] = val

    def init_mask(self, da=None, lsm_file=None, mask_ds=None, verbose=True, **kwargs):

        init_mask = kwargs.pop('init_mask', False)
        if lsm_file is not None:
            init_mask = True
        if init_mask:
            if da is None:
                da = self.get_da()
            dims = self.get_dims(ds=da)
            if 'lev' in dims:
                levs = da.lev
                da = da.sel(lev=levs[0])

            # if len(dims) > 2 or dims == ['time', 'points'] or dims == ['points', 'time']:
            if 'time' in dims:
                gut.myprint(
                    f'Init spatial mask for shape: {da.shape}', verbose=verbose)
                num_non_nans = xr.where(~np.isnan(da), 1, 0).sum(dim='time')
                mask = xr.where(num_non_nans == len(da.time), 1, 0)
                mask_dims = da.sel(time=da.time[0]).dims
                mask_coords = da.sel(time=da.time[0]).coords
                gut.myprint(
                    f'... Finished Initialization spatial mask', verbose=verbose)
            else:
                mask = xr.where(~np.isnan(da), 1, 0)
                mask_dims = da.dims
                mask_coords = da.coords

            # Optional Land Sea Mask File
            if lsm_file is not None or mask_ds is not None:
                if mask_ds is None:
                    gut.myprint(f'Read file {lsm_file} for land sea mask!')
                    mask_ds = self.open_ds(nc_files=lsm_file,
                                           time_range=None,
                                           grid_step=self.grid_step,
                                           lon_range=self.lon_range,
                                           lat_range=self.lat_range,
                                           use_ds_grid=True,
                                           large_ds=False,
                                           lon360=self.lon360,  # bring to same lon range
                                           )
                    flip_mask = kwargs.pop('flip_mask', False)
                    if flip_mask:
                        gut.myprint(f'Flip land-sea mask!')
                        mask_ds = xr.where(mask_ds == 1, 0, 1)
                if isinstance(mask_ds, xr.Dataset):
                    mask_name = self.get_vars(ds=mask_ds)[0]
                    mask_ds = mask_ds[mask_name]

                if not mask.shape == mask_ds.shape:
                    raise ValueError(
                        f'LSM Mask {mask_ds.shape} and input data {mask.shape} not of same shape')
                mask = mask.data * mask_ds.data

            self.mask = xr.DataArray(
                data=mask,
                dims=mask_dims,
                coords=mask_coords,
                name="mask",
            )
            if np.count_nonzero(self.mask.data) == 0:
                raise ValueError('ERROR! Mask is the whole dataset!')
            self.ds = xr.where(self.mask, self.ds, np.nan)
            self.ds = self.ds.assign_attrs(self.info_dict)
            init_indices = kwargs.pop('init_indices', True)
            if init_indices:
                self.indices_flat, self.idx_map = self.init_map_indices(
                    verbose=verbose)
            else:
                gut.myprint('WARNING! Index dictionaries not initialized!',
                            color='yellow')
                self.indices_flat = self.idx_map = None
        else:
            gut.myprint('No mask initialized!', verbose=verbose)
            self.mask = None
        return self.mask

    def flatten_array(self, time=True, var_name=None, check=False):
        """Flatten and remove NaNs.
        """
        var_name = self.var_name if var_name is None else var_name
        dataarray = self.ds[var_name]

        data = gut.flatten_array(dataarray=dataarray, mask=self.mask,
                                 time=time, check=check)

        return data

    def lon_2_360(self):
        reload(sput)
        self.ds = sput.da_lon2_360(da=self.ds)

        return self.ds

    def lon_2_180(self):
        reload(sput)
        self.ds = sput.da_lon2_180(da=self.ds)

        return self.ds

    def get_da(self, ds=None):

        da = self.ds[self.var_name]
        return da

    def get_da_vars(self, var_list):
        vars = self.get_vars()
        for var in var_list:
            if var not in vars:
                raise ValueError(f'{var} not in dataset variables {vars}!')
        gut.myprint(f'Get dataarray of variables {var_list}!')

        return self.ds[var_list]

    # ###################### Find out indices, map, locations etc. ####################

    def get_map(self, data, name=None):
        """Restore dataarray map from flattened array.

        This also includes adding NaNs which have been removed.
        Args:
        -----
        data: np.ndarray (n,0)
            flatten datapoints without NaNs
        mask_nan: xr.dataarray
            Mask of original dataarray containing True for position of NaNs
        name: str
            naming of xr.DataArray

        Return:
        -------
        dmap: xr.dataArray
            Map of data
        """
        if name is None:
            name = self.var_name
        mask_arr = np.where(self.mask.data.flatten() == 1, True, False)
        non_zero_ds = np.count_nonzero(mask_arr)
        # Number of non-NaNs should be equal to length of data
        if np.count_nonzero(mask_arr) != len(data):
            raise ValueError(
                f"Number of defined ds points {
                    non_zero_ds} != # datapoints {len(data)}"
            )

        # create array with NaNs
        data_map = np.empty(len(mask_arr))
        data_map[:] = np.nan

        # fill array with sample
        data_map[mask_arr] = data

        dmap = xr.DataArray(
            data=np.reshape(data_map, self.mask.data.shape),
            dims=self.mask.dims,
            coords=self.mask.coords,
            name=name,
        )

        return dmap

    def flat_idx_array(self, idx_list):
        """
        Returns a flattened list of indices where the idx_list is at the correct position.
        """
        mask_arr = np.where(self.mask.data.flatten() == 1, True, False)
        len_index = np.count_nonzero(mask_arr)
        if len(idx_list) == 0:
            raise ValueError('Error no indices in idx_lst')
        max_idx = np.max(idx_list)
        if max_idx > len_index:
            raise ValueError(
                f"Error: index {max_idx} higher than #nodes {len_index}!")
        full_idx_lst = np.zeros(len_index)
        full_idx_lst[idx_list] = 1

        return full_idx_lst

    def get_map_for_idx(self, idx_lst, fill_val=0):
        flat_idx_arr = self.flat_idx_array(idx_list=idx_lst)
        idx_lst_map = self.get_map(flat_idx_arr)
        if fill_val != 0:
            idx_lst_map = xr.where(idx_lst_map == 0, fill_val, idx_lst_map)
        return idx_lst_map

    def init_map_indices(self, verbose=True):
        """
        Initializes the flat indices of the map.
        Usefule if get_map_index is called multiple times.
        Also defined spatial lon, lat locations are initialized.
        """
        reload(gut)
        gut.myprint('Init the point-idx dictionaries', verbose=verbose)
        mask_arr = np.where(self.mask.data.flatten() == 1, True, False)
        if np.count_nonzero(mask_arr) == 0:
            raise ValueError('ERROR! Mask is the whole dataset!')
        self.indices_flat = np.arange(
            0, np.count_nonzero(mask_arr), 1, dtype=int)
        self.idx_map = self.get_map(self.indices_flat, name="idx_flat")
        idx_lst_map = self.get_map_for_idx(idx_lst=self.indices_flat)
        if 'points' in self.get_dims():
            point_lst = idx_lst_map.where(
                idx_lst_map == 1, drop=True).points.data
            lons = self.idx_map.lon
            lats = self.idx_map.lat
            self.def_locs = gut.zip_2_lists(list1=lons, list2=lats)[point_lst]
        else:
            point_lst = np.where(idx_lst_map.data.flatten() == 1)[0]
            lons = self.ds.lon
            lats = self.ds.lat
            lons, lats = np.meshgrid(lons, lats)
            self.def_locs = gut.zip_2_lists(
                lons.flatten(), lats.flatten())[point_lst]

        self.key_val_idx_point_dict = gut.mk_dict_2_lists(
            self.indices_flat, point_lst)

        # This takes longer
        # def_locs = []
        # for idx in self.indices_flat:
        #     slon, slat = self.get_coord_for_idx(idx)
        #     def_locs.append([slon, slat])
        # self.def_locs = np.array(def_locs)

        return self.indices_flat, self.idx_map

    def get_coordinates_flatten(self):
        """Get coordinates of flatten array with removed NaNs.

        Return:
        -------
        coord_deg:
        coord_rad:
        map_idx:
        """
        # length of the flatten array with NaNs removed
        # length = self.flatten_array().shape[1]
        length = len(self.indices_flat)
        coord_deg = []
        map_idx = []
        for i in range(length):
            buff = self.get_map_index(i)
            coord_deg.append([buff["lon"], buff["lat"]])  # x, y
            map_idx.append(buff["point"])

        coord_rad = np.radians(coord_deg)  # transforms to np.array

        return np.array(coord_deg), coord_rad, np.array(map_idx)

    def get_dims(self, ds=None):
        if ds is None:
            ds = self.ds
        if isinstance(ds, xr.Dataset):
            # check if xarray version is new
            if not gut.check_xarray_version():
                dims = list(ds.dims.keys())
            else:
                dims = list(ds.dims)  # new in xarray 2023.06.

        elif isinstance(ds, xr.DataArray):
            dims = ds.dims
        else:
            dtype = type(ds)
            raise ValueError(
                f'ds needs to be of type xr.DataArray but is of type {dtype}!')

        return dims

    def get_idx_for_locs(self, locs):
        """This is just a wrapper for self.get_index_for_coord.

        Args:
            locs (tuples): tuples of (lon, lat) pairs

        Returns:
            int: idx
        """
        locs = np.array(locs)
        if not isinstance(locs, np.ndarray):
            locs = [locs]

        if len(locs) == 0:
            raise ValueError('No locations given!')

        idx_lst = []
        for loc in locs:
            lon, lat = loc
            idx = self.get_index_for_coord(lon=lon, lat=lat)
            idx_lst.append(idx)
        if len(locs) > 1:
            idx_lst = np.sort(np.unique(np.array(idx_lst)))
            return idx_lst
        else:
            return idx

    def get_index_for_coord(self, lon, lat):
        """Get index of flatten array for specific lat, lon."""
        lons = self.def_locs[:, 0]
        lats = self.def_locs[:, 1]

        # Here we reduce the range in which the location can be
        if lon < 180 - 2*self.grid_step and lon > -180 + 2*self.grid_step:
            idx_true = ((lats > lat-2*self.grid_step) &
                        (lats < lat+2*self.grid_step) &
                        (lons > lon-2*self.grid_step) &
                        (lons < lon+2*self.grid_step)
                        )
        else:  # Close to dateline you only use lats not to run into problems
            idx_true = ((lats > lat-2*self.grid_step) &
                        (lats < lat+2*self.grid_step)
                        )
        # This is important to find the old index again!
        idx_red = np.where(idx_true)[0]

        lon, lat, idx_all_red = sput.find_nearest_lat_lon(
            lon=lon, lat=lat,
            lon_arr=lons[idx_true],
            lat_arr=lats[idx_true]
        )
        idx = idx_red[idx_all_red]
        # idx = self.idx_map.sel(points=idx_all)   # wrong

        if np.isnan(idx) is True:
            raise ValueError(f"Error the lon {lon} lat {lat} is not defined!")

        return int(idx)

    def get_map_for_locs(self, locations):
        """Gives a map for a list of [lon, lat] locations.

        Args:
            locations (list): 2d list of locations

        Returns:
            xr.DataArray: Dataarray with the locations as 1 in the map
        """
        index_locs_lst = []

        index_locs_lst = self.get_idx_for_locs(locs=locations)
        loc_map = self.get_map_for_idx(idx_lst=index_locs_lst)

        return loc_map

    def get_points_for_idx(self, idx_lst):
        """Returns the point number of the map for a given index list.
        Important eg. to transform node ids to points of the network
        Args:
            idx_lst (list): list of indices of the network.

        Returns:
            np.array: array of the points of the index list
        """
        point_lst = []

        for idx in idx_lst:
            # map_idx = self.get_map_index(idx)
            # point = int(map_idx["point"])
            # point_lst.append(point)
            point_lst.append(self.key_val_idx_point_dict[idx])

        return np.array(point_lst, dtype=int)

    def add_loc_dict(
        self, name, lon_range, lat_range,
        slon=None, slat=None,
        n_rep_ids=1, color=None,
        sname=None,
        lname=None,
        reset_loc_dict=False,
    ):
        if reset_loc_dict:
            self.loc_dict = dict()
        this_loc_dict = dict(
            name=name,
            lon_range=lon_range,
            lat_range=lat_range,
            color=color,
            sname=sname,  # short name
            lname=lname   # long name
        )
        if (gut.check_range(lon_range, self.lon_range)) and (
            gut.check_range(lat_range, self.lat_range)
        ):
            # Choose first all ids in lon-lat range!
            ids_lst, _ = self.get_idx_region(
                this_loc_dict, def_map=None)  # these are ids, not points!
            locs = self.get_locs_for_indices(ids_lst)

            if len(ids_lst) > 0:
                # Representative Ids for every location
                mean_loc = self.get_mean_loc(ids_lst)
                gut.myprint(f"Name: {name}, loc{mean_loc}")
                if slon is None or slat is None:
                    slon, slat = mean_loc
                    loc = mean_loc
                if gut.check_range_val(slon, lon_range) and gut.check_range_val(slat, lat_range):
                    slon = slon
                    slat = slat
                else:
                    raise ValueError(f'ERROR {slon} or {slat} not in range!')
                idx_loc = self.get_index_for_coord(lon=slon, lat=slat)
                loc = self.get_loc_for_idx(idx_loc)
                if np.isnan(idx_loc):
                    gut.myprint(
                        f"WARNING! Rep IDs {idx_loc} for {name} not defined!")
                else:

                    rep_ids = self.get_n_ids(loc=mean_loc, num_nn=n_rep_ids)
                    rep_locs = self.get_locs_for_indices(rep_ids)
                    pids = self.get_points_for_idx(ids_lst)
                    if 'points' in self.get_dims():
                        data = self.ds.sel(points=pids)
                    else:
                        data = self.get_data_for_lon_lat_range(lon_range=lon_range,
                                                               lat_range=lat_range)

                    this_loc_dict["rep_ids"] = np.array(rep_ids)
                    this_loc_dict["loc"] = loc
                    this_loc_dict["locs"] = locs
                    this_loc_dict["rep_locs"] = rep_locs
                    this_loc_dict['ids'] = ids_lst
                    this_loc_dict['pids'] = pids
                    this_loc_dict['data'] = data
                    this_loc_dict["map"] = self.get_map(
                        self.flat_idx_array(ids_lst))
            else:
                raise ValueError(
                    f"ERROR! This region {
                        name} does not contain any data points!"
                )
        else:
            raise ValueError(
                f"ERROR! This region {name} does not fit into {
                    lon_range}, {lat_range}!"
            )

        self.loc_dict[name] = this_loc_dict

    def get_map_index(self, idx_flat):
        """Get lat, lon and index of map from index of flatten array
           without Nans.

        # Attention: Mask has to be initialised

        Args:
        -----
        idx_flat: int, list
            index or list of indices of the flatten array with removed NaNs

        Return:
        idx_map: dict
            Corresponding indices of the map as well as lat and lon coordinates
        """

        indices_flat = self.indices_flat

        idx_map = self.idx_map

        buff = idx_map.where(idx_map == idx_flat, drop=True)
        if idx_flat > len(indices_flat):
            raise ValueError("Index doesn't exist.")

        point = int(self.get_points_for_idx([idx_flat]))
        map_idx = {
            "lat": float(buff.lat.data),
            "lon": float(buff.lon.data),
            # "point": int(np.argwhere(idx_map.data == idx_flat)),
            "point": point
        }
        return map_idx

    # ################## spatial functions #################

    def cut_map(
        self, ds=None, lon_range=[-180, 180], lat_range=[-90, 90], dateline=False,
        set_ds=False, verbose=True, **kwargs,
    ):
        """Cut an area in the map. Use always smallest range as default.
        It lon ranges accounts for regions (eg. Pacific) that are around the -180/180 region.

        Args:
        ----------
        lon_range: list [min, max]
            range of longitudes
        lat_range: list [min, max]
            range of latitudes
        dateline: boolean
            use dateline range in longitude (eg. -170, 170 range) contains all points from
            170-180, -180- -170, not all between -170 and 170. Default is True.
        Return:
        -------
        ds_area: xr.dataset
            Dataset cut to range
        """
        if ds is None:
            ds = self.ds
        ds_cut = sput.cut_map(
            ds=ds, lon_range=lon_range, lat_range=lat_range, dateline=dateline
        )
        if set_ds:
            self.ds = ds_cut
            self.init_mask(da=self.ds[self.var_name],
                           verbose=verbose,
                           **kwargs)
        return ds_cut

    def get_spatio_temp_range(self, ds):
        dims = self.get_dims(ds=ds)
        time_range = [ds.time.data[0], ds.time.data[-1]
                      ] if len(dims) > 2 else None
        if len(dims) >= 2:
            lon_range = [float(ds[self.lon_name].min()),
                         float(ds[self.lon_name].max())]
            lat_range = [float(ds[self.lat_name].min()),
                         float(ds[self.lat_name].max())]
        else:
            lon_range = lat_range = None
        return time_range, lon_range, lat_range

    def common_grid(self, dataarray, grid_step=1,
                    min_lon=None, max_lon=None,
                    min_lat=None, max_lat=None,
                    grid_step_lon=None,
                    grid_step_lat=None,
                    use_ds_grid=False):
        """Common grid for all datasets.
        """
        if use_ds_grid:
            init_lat = self.ds.lat
            init_lon = self.ds.lon
            min_lat = float(np.min(init_lat)) if min_lat is None else min_lat
            min_lon = float(np.min(init_lon)) if min_lon is None else min_lon

            max_lat = float(np.max(init_lat)) if max_lat is None else min_lat
            max_lon = float(np.max(init_lon)) if max_lon is None else max_lon
        else:
            correct_max_lon = True if max_lon is None else False
            correct_min_lon = True if min_lon is None else False
            correct_min_lat = True if min_lat is None else False
            correct_max_lat = True if max_lat is None else False

            min_lat = float(
                np.min(dataarray["lat"])) if min_lat is None else min_lat
            min_lon = float(
                np.min(dataarray["lon"])) if min_lon is None else min_lon

            max_lat = float(
                np.max(dataarray["lat"])) if max_lat is None else max_lat
            max_lon = float(
                np.max(dataarray["lon"])) if max_lon is None else max_lon
            diff_lon = np.abs(max_lon - min_lon)
            if diff_lon-0.01 < 352:  # To avoid scenarios with big gap
                gut.myprint(f'WARNING: Max lon smaller than {180}!')
            if max_lon < 179 and max_lon > 175:  # To avoid scenarios with big gap
                gut.myprint(f'WARNING! Set max lon from {max_lon} to 179.75!')
                max_lon = 179.75 if correct_max_lon else max_lon
            if diff_lon > 352 and diff_lon < 360 and min_lon >= 0:
                gut.myprint(
                    f'WARNING! Set max lon from {max_lon} to 359.75 and {min_lon} to 0!')
                min_lon = 0 if correct_min_lon else min_lon
                max_lon = 359.75 if correct_max_lon else max_lon

            if min_lon == -180 and max_lon == 180:  # To avoid scenarios with big gap
                gut.myprint(f'WARNING! Set min lon from {min_lon} to -179.75')
                min_lon = 179.75 if correct_min_lon else min_lon

            if max_lat < 89 and max_lat > 85:  # To avoid scenarios with big gap
                max_lat = 89.5 if correct_max_lat else max_lat
                if max_lat == 89.5:
                    gut.myprint(
                        f'WARNING! Set max lat from {max_lat} to 89.5!')

            if min_lat > -89 and min_lat < -85:  # To avoid scenarios with big gap
                gut.myprint(f'WARNING! Set min lat from {min_lat} to -89.5!')
                min_lat = -89.5 if correct_min_lat else min_lat

            grid_step_lon = grid_step if grid_step_lon is None else grid_step_lon
            grid_step_lat = grid_step if grid_step_lat is None else grid_step_lat

            init_lat = gut.custom_arange(start=min_lat,
                                         end=max_lat,
                                         step=grid_step_lat)
            init_lon = gut.custom_arange(start=min_lon,
                                         end=max_lon,
                                         step=grid_step_lon)

            nlat = len(init_lat)
            if nlat % 2:
                # Odd number of latitudes includes the poles.
                gut.myprint(
                    f"WARNING: Poles might be included: {
                        min_lat} and {min_lat}!",
                    color='red', bold=True
                )

        gut.myprint(
            f"Interpolte grid from {min(init_lon)} to {max(init_lon)}, {
                min(init_lat)} to {max(init_lat)}!",
        )
        grid = {"lat": init_lat, "lon": init_lon}

        da = dataarray.interp(grid, method="nearest",
                              kwargs={"fill_value": "extrapolate"}
                              )  # Extrapolate if outside of the range
        return da

    def get_locations_in_range(
        self, def_map=None, lon_range=None, lat_range=None,
        dateline=False
    ):
        """
        Returns a map with the location within certain range.

        Parameters:
        -----------
        lon_range: list
            Range of longitudes, i.e. [min_lon, max_lon]
        lat_range: list
            Range of latitudes, i.e. [min_lat, max_lat]
        def_map: xr.Dataarray
            Map of data, i.e. the mask.

        Returns:
        --------
        idx_lst: np.array
            List of indices of the flattened map.
        mmap: xr.Dataarray
            Dataarray including ones at the location and NaNs everywhere else
        """
        if lon_range is None:
            lon_range = self.lon_range
        if lat_range is None:
            lat_range = self.lat_range
        if def_map is None:
            def_map = self.mask
        mmap = sput.get_locations_in_range(
            def_map=def_map, lon_range=lon_range, lat_range=lat_range,
            dateline=dateline
        )

        # Return these indices (NOT points!!!) that are defined
        idx_lst = np.where(gut.flatten_array(dataarray=mmap,
                                             mask=self.mask,
                                             time=False,
                                             # TODO check for better solution!
                                             check=False) > 0)[0]

        return {'idx': idx_lst,
                'mmap': mmap,
                }

    def get_coord_for_idx(self, idx):
        map_dict = self.get_map_index(idx)
        slon = float(map_dict["lon"])
        slat = float(map_dict["lat"])

        return slon, slat

    def get_coords_for_range(self, lon_range, lat_range):
        locs = self.get_locations_in_range(lon_range=lon_range,
                                           lat_range=lat_range)
        idx_lst = locs['idx']
        coord_lst = []
        for idx in idx_lst:
            coord_lst.append(self.get_coord_for_idx(idx=idx))

        return np.array(coord_lst)

    # def get_index_for_coord(self, lon, lat):
    #     """Get index of flatten array for specific lat, lon."""
    #     mask_arr = np.where(self.mask.data.flatten() == 1, True, False)
    #     indices_flat = np.arange(0, np.count_nonzero(mask_arr), 1, dtype=int)

    #     idx_map = self.get_map(indices_flat, name="idx_flat")

    #     # idx = idx_map.sel(lat=lat, lon=lon, method='nearest')
    #     lon, lat, idx_all = grid.find_nearest_lat_lon(
    #         lon=lon, lat=lat, lon_arr=idx_map["lon"], lat_arr=idx_map["lat"]
    #     )

    #     idx = self.idx_map.sel(points=idx_all)  # Because idx_map['lon'] contains all values also non-defined!
    #     if np.isnan(idx.data) is True:
    #         raise ValueError(f"Error the lon {lon} lat {lat} is not defined!")

    #     return int(idx)

    def get_mean_loc(self, idx_lst):
        """
        Gets a mean location for a list of indices
        that is defined!
        """
        lon_arr = []
        lat_arr = []
        if len(idx_lst) == 0:
            raise ValueError("ERROR! List of points is empty!")

        for idx in idx_lst:
            map_idx = self.get_map_index(idx)
            lon_arr.append(map_idx["lon"])
            lat_arr.append(map_idx["lat"])
        mean_lat = np.mean(lat_arr)

        if max(lon_arr) - min(lon_arr) > 180:
            lon_arr = np.array(lon_arr)
            lon_arr[lon_arr < 0] = lon_arr[lon_arr < 0] + 360

        mean_lon = np.mean(lon_arr)
        if mean_lon > 180:
            mean_lon -= 360

        nearest_locs = sput.haversine(
            mean_lon, mean_lat, self.def_locs[:,
                                              0], self.def_locs[:, 1], radius=1
        )
        idx_min = np.argmin(nearest_locs)
        mean_lon = self.def_locs[idx_min, 0]
        mean_lat = self.def_locs[idx_min, 1]

        return (mean_lon, mean_lat)

    def get_mean_loc_idx(self, idx_lst):
        mean_loc = self.get_mean_loc(idx_lst=idx_lst)
        mean_idx = self.get_idx_for_locs(locs=mean_loc)
        return mean_idx

    # ####################### temporal functions ############################
    def set_sel_tps_ds(self, tps):
        ds_sel = tu.get_sel_tps_ds(ds=self.ds, tps=tps)
        self.ds = ds_sel
        return

    def select_time_periods(self, time_snippets):
        """Cut time snippets from dataset and concatenate them.

        Args:
            time_snippets (np.array): Array of n time snippets
                with dimension (n,2).

        Returns:
            None
        """
        self.ds = tu.select_time_snippets(self.ds, time_snippets)
        return self.ds

    def detrend(self, dim="time", deg=1, startyear=None):
        """Detrend dataarray.
        Args:
            dim (str, optional): [description]. Defaults to 'time'.
            deg (int, optional): [description]. Defaults to 1.
        """
        reload(tu)
        gut.myprint("Detrending data...")
        da_detrend = tu.detrend_dim(self.ds[self.var_name], dim=dim, deg=deg,
                                    startyear=startyear)
        self.ds[self.var_name] = da_detrend
        self.ds.attrs["detrended"] = "True"
        gut.myprint("... finished!")
        return

    def compute_anomalies(self, dataarray=None, group="dayofyear",
                          normalize_anomalies=False):
        """Calculate anomalies.

        Parameters:
        -----
        dataarray: xr.DataArray
            Dataarray to compute anomalies from.
        group: str
            time group the anomalies are calculated over, i.e. 'month', 'day', 'dayofyear'

        Return:
        -------
        anomalies: xr.dataarray
        """
        reload(tu)
        if dataarray is None:
            dataarray = self.ds[self.var_name]
        anomalies = tu.compute_anomalies(dataarray=dataarray,
                                         group=group,
                                         normalize=normalize_anomalies)

        return anomalies

    def compute_anomalies_ds(self, var_name=None, verbose=True, **kwargs):
        self.an_types = kwargs.pop('an_types', [])
        normalize_anomalies = kwargs.pop('normalize_anomalies', False)
        var_name = self.var_name if var_name is None else var_name
        gut.myprint('Compute Anomalies!', verbose=verbose)
        if var_name in self.vars:
            for an_type in self.an_types:
                self.ds[f"{var_name}_an_{an_type}"] = self.compute_anomalies(
                    self.ds[var_name], group=an_type,
                    normalize_anomalies=normalize_anomalies
                )
        return self.ds

    def compute_all_anomalies(self, **kwargs):
        self.an_types = kwargs.pop('an_types',
                                   ['dayofyear', 'month', 'JJAS'])
        self.vars = self.get_vars()
        if self.can:
            for vname in self.vars:
                for an_type in self.an_types:
                    var_type = f'{vname}_an_{an_type}'
                    if var_type not in self.vars:
                        gut.myprint(f'Compute anomalies {var_type}')
                        self.ds[var_type] = self.compute_anomalies(
                            dataarray=self.ds[vname],
                            group=an_type)

    def interp_times(self, dataset, time_range):
        """Interpolate time in time range in steps of days.
        TODO: So far only days works.
        """
        time_grid = np.arange(
            time_range[0], time_range[1], dtype="datetime64[D]")
        ds = dataset.interp(time=time_grid, method="nearest")
        return ds

    def get_data_timerange(self, data, time_range=None, verbose=True):
        """Gets data in a certain time range.
        Checks as well if time range exists in file!

        Args:
            data (xr.Dataarray): xarray dataarray
            time_range (list, optional): List dim 2 that contains the time interval. Defaults to None.

        Raises:
            ValueError: If time range is not in time range of data

        Returns:
            xr.Dataarray: xr.Dataarray in seleced time range.
        """

        td = data.time.data
        if time_range is not None:
            if (tu.is_larger_as(td[0], time_range[0])) or (
                tu.is_larger_as(time_range[1], td[-1])
            ):
                raise ValueError(
                    f"Chosen time {time_range} out of range. Please select times within {td[0]} - {td[-1]}!")
            else:
                sd = tu.tp2str(time_range[0])
                ed = tu.tp2str(time_range[-1])
                gut.myprint(f"Time steps within {sd} to {ed} selected!")
            # da = data.interp(time=t, method='nearest')
            da = data.sel(time=slice(time_range[0], time_range[1]))

            gut.myprint("Time steps selected!", verbose=verbose)
        else:
            da = data
        tr = tu.get_time_range(ds=da)
        gut.myprint(f'Load data from time range {tu.tps2str(tr)}!',
                    verbose=verbose)
        return da

    def get_month_range_data(
        self, dataarray=None, start_month="Jan", end_month="Dec", set_zero=False,
    ):
        """
        This function generates data within a given month range.
        It can be from smaller month to higher (eg. Jul-Sep) but as well from higher month
        to smaller month (eg. Dec-Feb)

        Parameters
        ----------
        start_month : string, optional
            Start month. The default is 'Jan'.
        end_month : string, optional
            End Month. The default is 'Dec'.
        set_zero : Sets all values outside the month range to zero, but remains the days.
                   Might be useful for event time series!

        Returns
        -------
        seasonal_data : xr.dataarray
            array that contains only data within month-range.

        """
        reload(tu)
        if dataarray is None:
            dataarray = self.ds[self.var_name]
        if set_zero:
            # Sets everything in the month range to zero
            seasonal_data = tu.get_month_range_zero(
                dataarray=dataarray, start_month=start_month, end_month=end_month
            )
        else:
            seasonal_data = tu.get_month_range_data(
                dataset=dataarray, start_month=start_month, end_month=end_month
            )

        return seasonal_data

    def set_month_range_data(
        self, dataset=None, start_month="Jan", end_month="Dec", set_zero=False
    ):
        if dataset is None:
            dataset = self.ds
        ds_all = []
        all_vars = self.get_vars(ds=dataset)
        for name in all_vars:
            gut.myprint(f"Month range for: {name}")
            ds_all.append(
                self.get_month_range_data(
                    dataarray=dataset[name],
                    start_month=start_month,
                    end_month=end_month,
                    set_zero=set_zero,
                )
            )
        self.ds = xr.merge(ds_all)

    # ######################### explore dataset values #######################

    def get_max(self, var_name=None):
        if var_name is None:
            var_name = self.var_name
        maxval = (
            self.ds[var_name]
            .where(self.ds[var_name] == self.ds[var_name].max(), drop=True)
            .squeeze()
        )
        lon = float(maxval.lon)
        lat = float(maxval.lat)
        tp = maxval.time

        return {"lon": lon, "lat": lat, "tp": tp}

    def get_min(self, var_name=None):
        if var_name is None:
            var_name = self.var_name
        maxval = (
            self.ds[var_name]
            .where(self.ds[var_name] == self.ds[var_name].min(), drop=True)
            .squeeze()
        )
        lon = float(maxval.lon)
        lat = float(maxval.lat)
        tp = maxval.time

        return {"lon": lon, "lat": lat, "tp": tp}

    def get_data_for_indices(self, idx_lst, var=None):
        if var is None:
            var = self.var_name

        dims = self.get_dims()
        if 'points' in dims:
            data_arr = self.ds[var].sel(
                points=self.get_points_for_idx(idx_lst))
        elif 'lon' in dims and 'lat' in dims:
            locs = self.get_locs_for_indices(idx_lst=idx_lst)
            data_arr = []
            for loc in locs:
                lon, lat = loc
                data_arr.append(self.ds.sel(lon=lon, lat=lat)[var].data)
            data_arr = np.array(data_arr)

            if 'lev' in dims:
                data_arr = gut.create_xr_ds(
                    data=data_arr,
                    dims=['ids', 'lev', 'time'],
                    coords={'time': self.ds.time,
                            'ids': idx_lst,
                            'lon': ('ids', [loc[0] for loc in locs]),
                            'lat': ('ids', [loc[1] for loc in locs]),
                            'lev': self.ds.lev},
                    name=var
                )
            else:
                data_arr = gut.create_xr_ds(
                    data=data_arr,
                    dims=['ids', 'time'],
                    coords={'time': self.ds.time,
                            'ids': idx_lst,
                            'lon': ('ids', [loc[0] for loc in locs]),
                            'lat': ('ids', [loc[1] for loc in locs]),
                            },
                    name=var
                )
        return data_arr

    def get_data_for_coord(self, lon, lat, var=None):
        if var is None:
            var = self.var_name
        idx = self.get_index_for_coord(lon=lon, lat=lat)
        ts = self.get_data_for_indices(idx_lst=[idx], var=var)
        return ts

    def get_data_for_locs(self, locs, var=None):
        if var is None:
            var = self.var_name

        idx_lst = self.get_idx_for_locs(locs=locs)
        ts = self.get_data_for_indices(idx_lst=idx_lst, var=var)
        return ts

    def get_data_for_lon_lat_range(self, lon_range, lat_range, dateline=False):
        return self.cut_map(lon_range=lon_range, lat_range=lat_range,
                            dateline=dateline)

    def get_idx_region(self, region_dict, def_map=None, dateline=False):
        """
        Gets the indices for a specific dictionary that has lon/lat_range as keys.
        E.g. can be applied to get all indices of the South American monsoon defined by Wang/EE.
        """
        if def_map is None:
            def_map = self.mask
            if def_map is None:
                raise ValueError(
                    "ERROR mask is None! Check if mask is computed properly!"
                )

        lon_range = region_dict["lon_range"]
        lat_range = region_dict["lat_range"]
        loc_dict = self.get_locations_in_range(
            lon_range=lon_range, lat_range=lat_range, def_map=def_map,
            dateline=dateline,
        )
        ids = loc_dict['idx']
        mmap = loc_dict['mmap']
        return ids, mmap

    def get_loc_for_idx(self, idx):
        """Returns a (lon,lat) location for an int index.

        Args:
            idx (int): list of indices integers

        Returns:
            tuple: tuple of (lon, lat)
        """
        val = sput.get_val_array(self.idx_map, idx)
        lon = float(val['lon'])
        lat = float(val['lat'])

        # point = self.get_points_for_idx([idx])[0]
        # lon = float(self.idx_map[point]['lon'])
        # lat = float(self.idx_map[point]['lat'])
        return lon, lat

    def get_locs_for_indices(self, idx_lst):
        """Returns a list of (lon,lat) locations for an int index.

        Args:
            idx_lst (list): list of indices integers

        Returns:
            np.array: array of tuples of (lon, lat)
        """
        # points = self.get_points_for_idx(idx_lst=idx_lst)
        # loc_array = []
        # for point in points:
        #     lon = float(self.idx_map[point]['lon'])
        #     lat = float(self.idx_map[point]['lat'])
        #     loc_array.append([lon, lat])

        loc_array = []
        for idx in idx_lst:
            lon, lat = self.get_loc_for_idx(idx=idx)
            loc_array.append([lon, lat])

        return np.array(loc_array)

    def get_n_ids(self, loc=None, nid=None, num_nn=3):
        """
        Gets for a specific location, the neighboring lats and lons ids.
        ----
        Args:
        loc: (float, float) provided as lon, lat values
        """
        if loc is None and nid is None:
            raise ValueError('provide either loc or nid!')
        if loc is not None and nid is not None:
            raise ValueError(
                f'Both loc {loc} and nid {nid} are given! Provide either loc or nid!')

        if nid is not None:
            slon, slat = self.get_loc_for_idx(nid)
        else:
            slon, slat = loc
        # lon = self.grid['lon']
        # lat = self.grid['lat']
        # sidx = self.get_index_for_coord(lon=slon, lat=slat)
        # sidx_r = self.get_index_for_coord(lon=slon + self.grid_step, lat=slat)
        # sidx_t = self.get_index_for_coord(lon=slon, lat=slat + self.grid_step)

        nearest_locs = sput.haversine(
            slon, slat, self.def_locs[:, 0], self.def_locs[:, 1], radius=1
        )
        idx_sort = np.argsort(nearest_locs)
        n_idx = []
        for idx in range(num_nn):
            sidx_t = self.get_index_for_coord(
                lon=self.def_locs[idx_sort[idx], 0],
                lat=self.def_locs[idx_sort[idx], 1]
            )
            n_idx.append(sidx_t)
        return np.array(n_idx)

    def get_n_locs(self, loc=None, nid=None, num_nn=3):
        """Returns an array of next neighbor locations for a location or an index.

        Args:
            loc (tuple, optional): tuple of (lon, lat). Defaults to None.
            nid (int, optional): index of location. Defaults to None.
            num_nn (int, optional): number of next neighbor locations. Defaults to 3.

        Returns:
            array: array of locations.
        """
        nids = self.get_n_ids(loc=loc, nid=nid, num_nn=num_nn)
        locs = self.get_locs_for_indices(idx_lst=nids)
        return locs

    def get_data_spatially_seperated_regions(self, min_num_locations=10,
                                             dist=None,
                                             var=None,
                                             verbose=True):
        """This function seperates the data spatially into regions that are seperated by a distance of dist.
        but the regions are also required to have a minimum number of locations.
        The locations are spatially connected by a distance of dist.

        Args:
            min_num_locations (int, optional): Minium number of locations. Defaults to 10.
            dist (float, optional): Minimum distance. Defaults to None.
            verbose (bool, optional): verbose results. Defaults to True.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if dist is None:
            dist = self.grid_step + 0.5
        if self.mask is None:
            raise ValueError(
                f'ERROR mask is None! Check if mask is computed properly!')
        def_locs = self.def_locs
        groups = sput.find_location_groups(def_locs,
                                           grid_step=dist,
                                           min_num_locations=min_num_locations,
                                           verbose=verbose)

        data_arr = []
        for group in groups:
            data = self.get_data_for_locs(locs=group, var=var)
            index_lst = self.get_idx_for_locs(locs=group)
            data_map = self.get_map_for_idx(idx_lst=index_lst, fill_val=np.nan)
            data_arr.append(dict(data=data, data_map=data_map,
                                 locs=group, idx_lst=index_lst))
        return data_arr

    def make_derivative(self, dx='time', var_name=None, group='JJAS'):
        if var_name is None:
            var_name = self.var_name

        da = self.ds[var_name]
        self.w_grad = da.differentiate(dx).rename(f'{var_name}_grad_{dx}')
        self.w_grad_an = tu.compute_anomalies(
            dataarray=self.w_grad, group=group)
        return self.w_grad, self.w_grad_an

    def apply_timemean(self, timemean=None, verbose=True):
        self.ds = tu.compute_timemean(
            ds=self.ds, timemean=timemean, verbose=verbose)
        return self.ds

    def apply_timemmax(self, timemean=None, verbose=True):
        self.ds = tu.compute_timemax(
            ds=self.ds, timemean=timemean, verbose=verbose)
        return self.ds

    def average_time(self, timemean='full'):
        if timemean == 'full':
            self.ds = self.ds.mean(dim='time')
        else:
            self.ds = self.apply_timemean(timemean=timemean)
        return self.ds

    def filter_nans(self, th=1, dims=['lon', 'lat']):
        da = self.ds[self.var_name]
        self.ds = tu.filter_nan_values(
            dataarray=da, dims=dims, th=th).to_dataset()
        self.set_var()
        return self.ds

    def set_metpy_labels(self):
        self.rename_var(old_var_name='lon', new_var_name='longitude')
        self.rename_var(old_var_name='lat', new_var_name='latitude')

    def set_metpy_units(self, unit):
        from metpy.units import units
        gut.myprint(f'Set units to {unit}')
        self.ds[self.var_name] = self.ds[self.var_name] * units(f'{unit}')
        return self.ds

    #  #################### EVS time series ##############

    def create_evs_ds(
        self, var_name,
        q=0.95,
        min_threshold=1,
        th_eev=15,
        min_evs=20,
        month_range=None
    ):
        """Genereates an event time series of the variable of the dataset.
        Attention, if month range is provided all values not in the month range are
        set to 0, not deleted, therefore the number of dates is retained

        Args:
            var_name (str): variable name
            q (float, optional): Quantile that defines extreme events. Defaults to 0.95.
            th (float, optional): threshold of minimum value in a time series. Defaults to 1.
            th_eev (float, optional): Threshold of minimum value of an extreme event. Defaults to 15.
            min_evs (int, optional): Minimum number of extreme events in the whole time Series. Defaults to 20.
            month_range (list, optional): list of strings as [start_month, end_month]. Defaults to None.

        Returns:
            xr.Dataset: Dataset with new values of variable and event series
        """
        self.q = q
        self.min_threshold = min_threshold
        self.th_eev = th_eev
        self.min_evs = min_evs
        gut.myprint(f'Create EVS with EE defined by q > {q}')
        if month_range is None:
            da_es, self.mask = self.compute_event_time_series(
                var_name=var_name,)
        else:
            da_es, self.mask = self.compute_event_time_series_month_range(
                start_month=month_range[0], end_month=month_range[1]
            )
        da_es.attrs = {"var_name": var_name}

        da_es = self.set_ds_attrs_evs(ds=da_es)
        self.ds["evs"] = da_es
        # set also to dataset object the attrs.
        self.ds = self.set_ds_attrs_evs(ds=self.ds)

        return self.ds

    def get_q_maps(self, var_name):

        if var_name is None:
            var_name = self.var_name
        gut.myprint(f"Apply Event Series on variable {var_name}")

        dataarray = self.ds[var_name]

        q_val_map, ee_map, data_above_quantile, rel_frac_q_map = tu.get_ee_ds(
            dataarray=dataarray, q=self.q, th=self.th, th_eev=self.th_eev
        )

        return q_val_map, ee_map, data_above_quantile

    def compute_event_time_series(
        self, var_name=None, **kwargs,
    ):
        reload(tu)
        if var_name is None:
            var_name = self.var_name
        gut.myprint(f"Apply Event Series on variable {var_name}")

        dataarray = self.ds[var_name]

        event_series, mask = tu.compute_evs(
            dataarray=dataarray,
            q=self.q,
            min_threshold=self.min_threshold,
            th_eev=self.th_eev,
            min_evs=self.min_evs,
        )

        return event_series, mask

    def compute_event_time_series_month_range(
        self, start_month="Jan", end_month="Dec",
    ):
        """
        This function generates data within a given month range.
        It can be from smaller month to higher (eg. Jul-Sep) but as well from higher month
        to smaller month (eg. Dec-Feb)

        Parameters
        ----------
        start_month : string, optional
            Start month. The default is 'Jan'.
        end_month : string, optional
            End Month. The default is 'Dec'.

        Returns
        -------
        seasonal_data : xr.dataarray
            array that contains only data within month-range.

        """
        reload(tu)
        times = self.ds["time"]
        start_year, end_year = tu.get_sy_ey_time(times, sy=None, ey=None)
        gut.myprint(
            f"Get month range data from year {start_year} to {end_year}!")

        da = self.ds[self.var_name]
        # Sets the data outside the month range to 0, but retains the dates
        da_mr = self.get_month_range_data(
            dataarray=da, start_month=start_month, end_month=end_month,
            set_zero=True
        )
        # Computes the Event Series
        evs_mr, mask = tu.compute_evs(
            dataarray=da_mr,
            q=self.q,
            min_threshold=self.min_threshold,
            th_eev=self.th_eev,
            min_evs=self.min_evs,
        )

        return evs_mr, mask

    def set_ds_attrs_evs(self, ds):
        param_class = {
            "grid_step": self.grid_step,
            "grid_type": self.grid_type,
            "q": self.q,
            "min_evs": self.min_evs,
            "min_threshold": self.min_threshold,
            "th_eev": self.th_eev,
            "an": int(self.can),
            **self.info_dict
        }
        ds.attrs = param_class
        return ds

    def horizontal_gradient(self, var, dim='lon', can=True):
        """Calculate horizontal gradient of variable.

        Args:
            var (str): variable name
            dim (str, optional): dimension to take gradient over. Defaults to 'lon'.

        Returns:
            xr.Dataarray: horizontal gradient
        """
        v_bar = self.ds[var]
        new_name = var+'_grad_'+dim
        grad_vbar = v_bar.differentiate(dim).rename(new_name)

        self.ds[new_name] = grad_vbar

        if can:
            for an_type in self.an_types:
                sf_an = tu.compute_anomalies(
                    dataarray=self.ds[new_name], group=an_type)
                self.ds[sf_an.name] = sf_an

        return grad_vbar
