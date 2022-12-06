#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 08:53:08 2020
Class for network of rainfall events
@author: Felix Strnad
"""
# %%

from climnet.dataset import BaseRectDataset
import sys
import os
import numpy as np
import xarray as xr
import copy
import climnet.utils.time_utils as tu

class Moisture_Flux(BaseRectDataset):
    """ Dataset for surface pressure.

    Args:
    ----------
    nc_file: str
        filename
    var_name: str
        Variable name of interest
    """

    def __init__(self, load_nc_u=None, load_nc_v=None, load_nc_q=None,
                 load_nc_mf=None,
                 can=True,
                 **kwargs):

        u_kwargs = copy.deepcopy(kwargs)
        v_kwargs = copy.deepcopy(kwargs)
        sh_kwargs = copy.deepcopy(kwargs)
        if load_nc_mf is None:
            # u-winds
            ds_uwind = BaseRectDataset(load_nc=load_nc_u,
                                       can=can,
                                       **u_kwargs)
            # v-winds
            ds_vwind = BaseRectDataset(load_nc=load_nc_v,
                                       can=can,
                                       **v_kwargs)
            # specific humidity
            self.ds_q = BaseRectDataset(load_nc=load_nc_q,
                                        can=can,
                                        **sh_kwargs)
            self.u = ds_uwind.ds['u']
            self.v = ds_vwind.ds['v']
            self.q = self.ds_q.ds['q']
            if self.u.data.shape != self.v.data.shape or self.u.data.shape != self.q.data.shape:
                print(f'u: {self.u.data.shape}')
                print(f'v: {self.v.data.shape}')
                print(f'q: {self.q.data.shape}')
                raise ValueError('ERROR! Datasets not of the same shape!')
            self.grid_step = self.ds_q.grid_step
            # Compute Moisture flux u-direction
            u_mf = self.u * self.q
            self.u_mf = u_mf.rename('u_mf')

            # Compute Moisture flux v-direction
            v_mf = self.v * self.q
            self.v_mf = v_mf.rename('v_mf')
            print("Computed single components of wind dataset. Now merge to 1 dataset!")

            self.can = can

            self.ds = self.get_ds()
            self.vars = self.get_vars()
            an_types = kwargs.pop('an_types', ['dayofyear'])

            if self.can is True:
                for vname in self.vars:
                    for an_type in an_types:
                        self.ds[f'{vname}_an_{an_type}'] = self.compute_anomalies(
                            dataarray=self.ds[vname],
                            group=an_type)
        else:
            self.load(load_nc=load_nc_mf)

    def compute_sf(self, u_name='u_mf', v_name='v_mf'):
        ds_S = rws.Rossby_Wave_Source(
            ds_uwind=self.ds[u_name],
            ds_vwind=self.ds[v_name],
            an_types=['sf']
        )
        S_vars = ds_S.get_vars()
        for var in S_vars:
            self.ds[var] = ds_S[var]
        self.ds[var] = ds_S[var]

    def get_ds(self):
        ds = xr.merge([
            self.u,
            self.v,
            self.u_mf,
            self.v_mf,
            self.q,
        ]
        )

        return ds

    def load(self, load_nc, lon_range=[-180, 180], lat_range=[-90, 90]):
        BaseRectDataset.load(
            self, load_nc=load_nc, lon_range=lon_range, lat_range=lat_range)
        self.u = self.ds['u_mf']
        self.v = self.ds['v_mf']
        self.q = self.ds['q']
