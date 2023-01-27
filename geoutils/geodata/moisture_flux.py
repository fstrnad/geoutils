#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Felix Strnad
"""
# %%
import geoutils.geodata.wind_dataset as wds

import xarray as xr
import numpy as np
import geoutils.utils.time_utils as tu
import geoutils.utils.general_utils as gut
from windspharm.xarray import VectorWind
import copy
from importlib import reload
reload(wds)


class MoistureFlux(wds.Wind_Dataset):
    """ Dataset for surface pressure.

    Args:
    ----------
    nc_file: str
        filename
    var_name: str
        Variable name of interest
    """

    def __init__(self,
                 load_nc_arr_u=None,
                 load_nc_arr_v=None,
                 load_nc_arr_w=None,
                 load_nc_arr_q=None,
                 can=True,
                 **kwargs):
        if load_nc_arr_q is None:
            gut.myprint(f'ERROR! Please provide specific humidity file')

        self.can = can
        ds_ivt = wds.Wind_Dataset(load_nc_arr_u=load_nc_arr_u,
                                  load_nc_arr_v=load_nc_arr_v,
                                  load_nc_arr_w=load_nc_arr_w,
                                  load_nc_arr_fac=load_nc_arr_q,
                                  fac_name='q',
                                  can=False,  # anoamlies are computed later
                                  **kwargs)
        self.ds = ds_ivt.ds

        init_mask = kwargs.pop('init_mask', True)
        self.load_dataset_attributes(base_ds=ds_ivt, init_mask=init_mask)
        self.u_name = ds_ivt.u_name
        self.v_name = ds_ivt.v_name

        self.compute_ivt()
        self.compute_all_anomalies()

    def integrated_vapor_flux(self,
                              g=9.81,):
        gut.myprint(f'Compute Integrated vapor_flux...')
        for var in ['u', 'v']:
            psi = self.vertical_integration(var=var)
            if var == 'v':
                vname = 'northward_vf'
            if var == 'u':
                vname = 'eastward_vf'

            self.ds[vname] = 1/g * psi

            gut.myprint(f'{var} finished')

    def compute_ivt(self):
        self.integrated_vapor_flux()

        ivt = self.compute_windspeed(u=self.ds['northward_vf'],
                                     v=self.ds['eastward_vf'],
                                     ws_name='ivt')
        self.ds['ivt'] = ivt

        return ivt
