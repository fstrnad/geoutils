#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue 1 March 08:53:08 2023
Class for multilevel pressure datasets
@author: Felix Strnad
"""
# %%

import geoutils.geodata.base_dataset as bds
import xarray as xr
import numpy as np
import geoutils.utils.time_utils as tu
import geoutils.utils.general_utils as gut
import geoutils.utils.file_utils as fut
from importlib import reload
reload(bds)


class MultiPressureLevelDataset(bds.BaseDataset):
    """ Dataset for surface pressure.

    Args:
    ----------
    nc_file: str
        filename
    var_name: str
        Variable name of interest
    """

    def __init__(self, data_nc=None,
                 plevels=None,
                 can=True,
                 **kwargs):

        if plevels is None:
            gut.myprint(
                'No Plevel provided! Assuming variable is vertically integrated!')
            plevels = [0]
        for file in data_nc:
            fut.print_file_location_and_size(filepath=file, verbose=False)
        gut.myprint(f'All files are available! Loading {data_nc}...',
                    lines=True)
        super().__init__(data_nc=data_nc,
                         plevels=plevels,
                         can=can,
                         **kwargs)
        gut.myprint(
            f'Loaded Pressure levels {plevels} as dimension {self.plevel_name}!')

        self.set_plevel_attrs()

    def load_dataset_attributes(self, base_ds, **kwargs):

        self.grid_step = base_ds.grid_step
        self.var_name = base_ds.var_name
        self.grid_type = base_ds.grid_type
        # Init Mask
        init_mask = kwargs.pop('init_mask', True)
        if init_mask:
            self.def_locs = base_ds.def_locs
            self.key_val_idx_point_dict = base_ds.key_val_idx_point_dict
            self.mask = base_ds.mask
            self.indices_flat = base_ds.indices_flat
            self.idx_map = base_ds.idx_map
            self.lon_range = base_ds.lon_range
            self.lat_range = base_ds.lat_range
            self.info_dict = base_ds.info_dict
        self.dims = base_ds.dims

    def set_plevel_attrs(self):
        self.plevel_attrs = self.ds[self.plevel_name].attrs
        self.plevel_attrs['standard_name'] = 'air_pressure'
        self.plevel_attrs['long_name'] = 'pressure_level'
        self.plevel_attrs['positive'] = 'down'
        self.plevel_attrs['units'] = 'hPa'
        self.plevel_attrs['axis'] = 'Z'
        self.ds[self.plevel_name].attrs.update(self.plevel_attrs)

    def vertical_integration(self, var, c=1):
        """Perform vertical integration over all pressure levels available.

        Args:
            var (str): variable name
            c (int, optional): constant to multiply with. Defaults to 1.

        Returns:
            xr.Dataarray: vertically integrated variable
        """
        v_bar = self.ds[var]
        lats = v_bar.lat
        lats = np.cos(lats*np.pi/180)
        plevels = v_bar.lev
        dp = xr.DataArray(np.diff(plevels, prepend=0)*100.,  # factor x100 because of hPa to bar
                          coords={'lev': plevels})
        gut.myprint(
            f'Integrate {var} from {float(plevels[0])} to {float(plevels[-1])}!')

        vert_int = c*np.cumsum(v_bar*dp, axis=v_bar.dims.index('lev'))
        return vert_int

    def horizontal_gradient(self, var, dim='lon'):
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
        return grad_vbar
