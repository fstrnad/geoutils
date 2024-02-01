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
                 ds_wind=None,
                 ds_q=None,
                 can=True,
                 plevels=None,
                 grad_q=False,
                 **kwargs):

        self.can = can
        init_mask = kwargs.pop('init_mask', False)
        if ds_wind is None or ds_q is None:
            if data_nc_q is None:
                gut.myprint(f'ERROR! Please provide specific humidity file')
                raise ValueError('ERROR! Please provide specific humidity file!')
            ds_ivt = wds.Wind_Dataset(data_nc_u=data_nc_u,
                                      data_nc_v=data_nc_v,
                                      data_nc_w=data_nc_w,
                                      data_nc_fac=data_nc_q,
                                      plevels=plevels,
                                      fac_name='q',
                                      grad_fac=grad_q,  # Compute horizontal gradient of q
                                      can=False,  # anoamlies are computed later
                                      **kwargs)
            self.load_dataset_attributes(base_ds=ds_ivt, init_mask=init_mask)
            self.ds = ds_ivt.ds
            self.u_name = ds_ivt.u_name
            self.v_name = ds_ivt.v_name
        else:
            gut.myprint(f'Load wind and q from given datasets')
            plevels_wind = ds_wind.ds.lev.values
            plevels_q = ds_q.ds.lev.values
            if not np.array_equal(plevels_wind, plevels_q):
                raise ValueError('ERROR! Pressure levels of wind and q do not match!')
            ds_ivt = (ds_wind.ds['U'] * ds_q.ds['q']).to_dataset(name='U')
            gut.myprint(f'Compute V * q')
            ds_ivt['V'] = ds_wind.ds['V'] * ds_q.ds['q']
            self.load_dataset_attributes(base_ds=ds_wind, init_mask=init_mask)
            self.ds = ds_ivt
            self.u_name = ds_wind.u_name
            self.v_name = ds_wind.v_name

        vi = kwargs.pop('vi', True)
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
