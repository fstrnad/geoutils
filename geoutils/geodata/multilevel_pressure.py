#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 08:53:08 2020
Class for network of rainfall events
@author: Felix Strnad
"""
# %%

import geoutils.geodata.base_dataset as bds
import xarray as xr
import numpy as np
import geoutils.utils.time_utils as tu
import geoutils.utils.general_utils as gut
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

    def __init__(self, data_nc_arr=None,
                 plevels=None,
                 can=True,
                 **kwargs):

        if data_nc_arr is None:
            raise ValueError(
                'Please provide array of dataset files to read!')
        if plevels is None:
            gut.myprint(
                'No Plevel provided! Assuming variable is vertically integrated!')
            plevels = [0]

        init_mask = kwargs.pop('init_mask', False)

        # Dimension name of pressure level
        self.plevel_name = kwargs.pop('plevel_name', 'lev')

        all_ds = []
        gut.myprint(
            f'Load Pressure levels {plevels} as dimension {self.plevel_name}!')
        for idx, plevel in enumerate(plevels):
            load_nc_file = data_nc_arr[idx]
            single_pl_ds = bds.BaseDataset(data_nc=load_nc_file,
                                           can=can,
                                           init_mask=False,  # is initialized later
                                           **kwargs)
            all_ds.append(single_pl_ds.ds.expand_dims(
                {self.plevel_name: 1}).assign_coords({self.plevel_name: [plevel]}))

        # To take all in init defined values also for multi-pressure levels
        # self = single_pl_ds
        gut.myprint(
            f'Plevels {plevels}, now merge all single datasets into one!')
        self.ds = xr.merge(all_ds)
        self.load_dataset_attributes(
            base_ds=single_pl_ds, init_mask=init_mask)
        self.set_plevel_attrs()

    def load_dataset_attributes(self, base_ds, **kwargs):

        self.grid_step = base_ds.grid_step
        self.var_name = base_ds.var_name
        self.grid_type = base_ds.grid_type
        self.an_types = base_ds.an_types
        # Init Mask
        init_mask = kwargs.pop('init_mask', True)
        if init_mask:
            self.def_locs = base_ds.def_locs
            self.key_val_idx_point_dict = base_ds.key_val_idx_point_dict
            self.mask = base_ds.mask
            self.indices_flat = base_ds.indices_flat
            self.idx_map = base_ds.idx_map

    def set_plevel_attrs(self):
        self.plevel_attrs = self.ds[self.plevel_name].attrs
        self.plevel_attrs['standard_name'] = 'air_pressure'
        self.plevel_attrs['long_name'] = 'pressure_level'
        self.plevel_attrs['positive'] = 'down'
        self.plevel_attrs['units'] = 'hPa'
        self.plevel_attrs['axis'] = 'Z'
        self.ds[self.plevel_name].attrs.update(self.plevel_attrs)

    def cut_map(self,  lon_range=[-180, 180], lat_range=[-90, 90]):
        ds = self.ds
        if lon_range != [-180, 180] and lat_range != [-90, 90]:
            ds = self.cut_map(self.ds, lon_range, lat_range)

        return ds

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
