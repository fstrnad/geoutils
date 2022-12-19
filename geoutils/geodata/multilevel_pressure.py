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

    def __init__(self, load_nc_arr=None,
                 load_nc=None,
                 plevels=None,
                 can=True,
                 **kwargs):

        if load_nc is None:
            if load_nc_arr is None:
                raise ValueError(
                    'Please provide array of dataset files to read!')
            if plevels is None:
                gut.myprint(
                    'No Plevel provided! Assuming variable is vertically integrated!')
                plevels = [0]

            all_ds = []
            gut.myprint(f'Load Pressure levels {plevels}!')
            init_mask = kwargs.pop('init_mask', True)
            for idx, plevel in enumerate(plevels):
                load_nc_file = load_nc_arr[idx]
                single_pl_ds = bds.BaseDataset(load_nc=load_nc_file,
                                               can=can,
                                               init_mask=init_mask,
                                               **kwargs)
                all_ds.append(single_pl_ds.ds.expand_dims(
                    {'plevel': 1}).assign_coords({'plevel': [plevel]}))

            # To take all in init defined values also for multi-pressure levels
            # self = single_pl_ds
            gut.myprint(
                f'Plevels {plevels}, now merge all single datasets into one!')
            self.ds = xr.merge(all_ds)
            self.load_dataset_attributes(base_ds=single_pl_ds, init_mask=init_mask)

        else:
            self.load(load_nc)

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

    def cut_map(self,  lon_range=[-180, 180], lat_range=[-90, 90]):
        ds = self.ds
        if lon_range != [-180, 180] and lat_range != [-90, 90]:
            ds = self.cut_map(self.ds, lon_range, lat_range)

        return ds
