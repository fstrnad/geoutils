#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 08:53:08 2020
Class for network of rainfall events
@author: Felix Strnad
"""
# %%

import geoutils.geodata.multilevel_pressure as mp
from importlib import reload
import numpy as np
import xarray as xr
from windspharm.xarray import VectorWind
import copy
import geoutils.utils.time_utils as tu
import geoutils.utils.general_utils as gut
reload(mp)


class Wind_Dataset(mp.MultiPressureLevelDataset):
    """ Dataset for surface pressure.

    Args:
    ----------
    nc_file: str
        filename
    var_name: str
        Variable name of interest
    """

    def __init__(self,
                 data_nc_arr_u=None,
                 data_nc_arr_v=None,
                 data_nc_arr_w=None,
                 data_nc_arr_fac=None,
                 compute_ws=False,
                 plevels=None,
                 can=True,
                 **kwargs):
        reload(mp)
        u_kwargs = copy.deepcopy(kwargs)
        v_kwargs = copy.deepcopy(kwargs)
        w_kwargs = copy.deepcopy(kwargs)
        self.u_name = kwargs.pop('u_name', 'u')
        self.v_name = kwargs.pop('v_name', 'v')
        if data_nc_arr_u is not None:
            ds_uwind = mp.MultiPressureLevelDataset(data_nc_arr=data_nc_arr_u,
                                                    plevels=plevels,
                                                    can=False,  # Anomalies are computed later all together
                                                    **u_kwargs)
            ds_vwind = mp.MultiPressureLevelDataset(data_nc_arr=data_nc_arr_v,
                                                    plevels=plevels,
                                                    can=False,
                                                    **v_kwargs)


            u = ds_uwind.ds[self.u_name]
            v = ds_vwind.ds[self.v_name]
            if self.u_name == 'u':
                u = u.rename('U')
                self.u_name = 'U'
            if self.v_name == 'v':
                v = v.rename('V')
                self.v_name = 'V'

            if data_nc_arr_fac is not None:
                ds_fac = mp.MultiPressureLevelDataset(data_nc_arr=data_nc_arr_fac,
                                                      plevels=plevels,
                                                      can=False,
                                                      **w_kwargs)
                fac_name = kwargs.pop('fac_name', 'fac')

                gut.myprint(f'Multiply u- and v by factor {fac_name}!')
                u = (u*ds_fac.ds[fac_name]).rename(self.u_name)
                v = (v*ds_fac.ds[fac_name]).rename(self.v_name)

            self.grid_step = ds_uwind.grid_step
            self.vert_velocity = False
            w = None
            ds_wwind = None
            if data_nc_arr_w is not None:
                ds_wwind = mp.MultiPressureLevelDataset(data_nc_arr=data_nc_arr_w,
                                                        plevels=plevels,
                                                        can=False,
                                                        **w_kwargs)
                self.w_name = kwargs.pop('w_name', 'w')
                w = ds_wwind.ds[self.w_name].rename('OMEGA')
                self.w_name = 'OMEGA'
                self.vert_velocity = True

            windspeed = None
            if compute_ws:
                windspeed = self.compute_windspeed(u=u, v=v)

            self.ds = self.get_ds(u=u, v=v, w=w, windspeed=windspeed)

            # ds_uwind would be possible as well
            init_mask = kwargs.pop('init_mask', True)
            self.load_dataset_attributes(base_ds=ds_vwind, init_mask=init_mask)

            self.vars = self.get_vars()
            self.can = can
            self.compute_all_anomalies(**kwargs)

            del u, v, w, windspeed, ds_uwind, ds_vwind, ds_wwind
        else:
            gut.myprint('Only Init the Wind Dataset object without data!')

    def get_ds(self, u, v, w=None, windspeed=None):
        ds = xr.merge([u, v])
        if windspeed is not None:
            ds = xr.merge([ds, windspeed])
        if w is not None:
            ds = xr.merge([ds, w])
        return ds

    def compute_windspeed(self, u, v, ws_name='windspeed'):
        windspeed = np.sqrt(u ** 2 + v ** 2)
        windspeed = windspeed.rename(ws_name)
        gut.myprint(
            "Computed single components of wind dataset. Now compute windspeed!")
        return windspeed

    def compute_vertical_shear(self, group='JJAS'):
        shear_wind = self.ds['u'].sel(
            plevel=200) - self.ds['u'].sel(plevel=900)
        shear_wind = shear_wind.rename('vertical_shear')
        shear_wind_an = tu.compute_anomalies(dataarray=shear_wind, group=group)
        self.vertical_shear = xr.merge([shear_wind, shear_wind_an])

        return self.vertical_shear

    def compute_vorticity(self, group='JJAS', can=True):
        """Compute vorticity with windspharm package
        see https://ajdawson.github.io/windspharm/latest/examples/rws_xarray.html
        """

        vw = VectorWind(self.ds[self.u_name], self.ds[self.u_name])

        gut.myprint('Compute relative vorticity...')
        rv = vw.vorticity()  # Relative Vorticity
        self.ds['rv'] = rv.rename('rv')
        if can:
            rv_an = tu.compute_anomalies(dataarray=self.ds['rv'], group=group)
            self.ds[rv_an.name] = rv_an
        return self.ds

    def compute_vertical_velocity_gradient(self, group='JJAS', dp='plevel'):
        w = self.ds['w']
        self.w_grad = w.differentiate(dp).rename(f'w_grad_{dp}')
        self.w_grad_an = tu.compute_anomalies(
            dataarray=self.w_grad, group=group)
        return self.w_grad, self.w_grad_an

    def compute_rossby_wave_source(self, can=True, group='JJAS'):
        """Compute rossby wave source from u and v components
        see https://ajdawson.github.io/windspharm/latest/examples/rws_xarray.html
        """

        vw = VectorWind(self.ds[self.u_name], self.ds[self.v_name])

        gut.myprint(f'Compute Vorticity...')
        eta = vw.absolutevorticity()
        gut.myprint(f'Compute Divergence...')
        div = vw.divergence()
        gut.myprint(f'Compute Rotation and gradient...')
        uchi, vchi = vw.irrotationalcomponent()
        etax, etay = vw.gradient(eta)

        gut.myprint("Compute Rossby Wave Source...")
        # Combine the components to form the Rossby wave source term.
        S = eta * -1. * div - (uchi * etax + vchi * etay)
        self.ds['S'] = S.rename('S')
        gut.myprint("... Computed Rossby Wave Source!")
        if can:
            self.ds[f'S_an_{group}'] = tu.compute_anomalies(dataarray=self.ds['S'], group=group)

        return self.ds

# %%
