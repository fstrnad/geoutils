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
                 data_nc_u=None,
                 data_nc_v=None,
                 data_nc_w=None,
                 data_nc_q=None,
                 can=True,
                 plevels=None,
                 **kwargs):
        if data_nc_q is None:
            gut.myprint(f'ERROR! Please provide specific humidity file')
            raise ValueError('ERROR! Please provide specific humidity file!')
        self.can = can
        ds_ivt = wds.Wind_Dataset(data_nc_u=data_nc_u,
                                  data_nc_v=data_nc_v,
                                  data_nc_w=data_nc_w,
                                  data_nc_fac=data_nc_q,
                                  plevels=plevels,
                                  fac_name='q',
                                  can=False,  # anoamlies are computed later
                                  **kwargs)
        self.ds = ds_ivt.ds

        init_mask = kwargs.pop('init_mask', True)
        self.load_dataset_attributes(base_ds=ds_ivt, init_mask=init_mask)
        self.u_name = ds_ivt.u_name
        self.v_name = ds_ivt.v_name

        vi = kwargs.pop('vi', False)
        if vi:
            self.compute_ivt()
        self.compute_all_anomalies(**kwargs)

    def integrated_vapor_flux(self,
                              g=9.81,):
        gut.myprint(f'Compute Integrated vapor_flux...')
        for var in [self.u_name, self.v_name]:
            psi = self.vertical_integration(var=var)
            if var == self.v_name:
                vname = 'northward_vf'
            if var == self.u_name:
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
