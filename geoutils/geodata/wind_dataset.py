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
                 data_nc_u=None,
                 data_nc_v=None,
                 data_nc_w=None,
                 data_nc_fac=None,
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
        if data_nc_u is not None:
            ds_uwind = mp.MultiPressureLevelDataset(data_nc=data_nc_u,
                                                    plevels=plevels,
                                                    can=False,  # Anomalies are computed later all together
                                                    **u_kwargs)
            ds_vwind = mp.MultiPressureLevelDataset(data_nc=data_nc_v,
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

            if data_nc_fac is not None:
                ds_fac = mp.MultiPressureLevelDataset(data_nc=data_nc_fac,
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
            if data_nc_w is not None:
                ds_wwind = mp.MultiPressureLevelDataset(data_nc=data_nc_w,
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
        gut.myprint(f'Merge u, v')
        ds = xr.Dataset({self.u_name: u,
                         self.v_name: v})
        if windspeed is not None:
            gut.myprint(f'Merge u, v, windspeed')
            ds[self.ws_name] = windspeed
        if w is not None:
            gut.myprint(f'Merge u, v, omega')
            ds[self.w_name] = w
        return ds

    def compute_windspeed(self, u, v, ws_name='windspeed'):
        windspeed = np.sqrt(u ** 2 + v ** 2)
        self.ws_name = ws_name
        windspeed = windspeed.rename(self.ws_name)
        gut.myprint(
            "Computed single components of wind dataset. Now compute windspeed!")
        return windspeed

    def compute_vertical_shear(self, plevel_up=200, plevel_low=850,
                               group='JJAS'):
        gut.myprint(f'Compute Vertical shear winds.')
        shear_wind = self.ds[self.u_name].sel(
            lev=plevel_up) - self.ds[self.u_name].sel(lev=plevel_low)
        self.ds['vertical_shear'] = shear_wind.rename('vertical_shear')
        for group in self.an_types:
            shear_wind_an = tu.compute_anomalies(
                dataarray=self.ds['vertical_shear'], group=group)
            self.ds[shear_wind_an.name] = shear_wind_an

        return self.ds

    def compute_vorticity(self, can=True):
        """Compute vorticity with windspharm package
        see https://ajdawson.github.io/windspharm/latest/examples/rws_xarray.html
        """

        vw = VectorWind(self.ds[self.u_name], self.ds[self.u_name])

        gut.myprint('Compute relative vorticity...')
        rv = vw.vorticity()  # Relative Vorticity
        self.ds['rv'] = rv.rename('rv')
        if can:
            for group in self.an_types:
                rv_an = tu.compute_anomalies(
                    dataarray=self.ds['rv'], group=group)
                self.ds[rv_an.name] = rv_an
        return self.ds

    def compute_divergence(self, can=True):
        """Compute vorticity with windspharm package
        see https://ajdawson.github.io/windspharm/latest/examples/rws_xarray.html
        """

        vw = VectorWind(self.ds[self.u_name], self.ds[self.u_name])

        gut.myprint('Compute divergence...')
        div = vw.divergence()  # divergence
        self.ds['div'] = div.rename('div')
        if can:
            for group in self.an_types:
                div_an = tu.compute_anomalies(
                    dataarray=self.ds['div'], group=group)
                self.ds[div_an.name] = div_an
        return self.ds

    def compute_vertical_velocity_gradient(self, dp='lev', can=True):
        w = self.ds[self.w_name]
        w_grad = w.differentiate(dp).rename(f'w_grad_{dp}')
        self.ds['vert_grad'] = w_grad
        if can:
            for group in self.an_types:
                vert_grad_an = tu.compute_anomalies(
                    dataarray=self.ds['vert_grad'], group=group)
                self.ds[vert_grad_an.name] = vert_grad_an
        return self.ds

    def compute_rossby_wave_source(self, can=True):
        """Compute rossby wave source from u and v components
        see https://ajdawson.github.io/windspharm/latest/examples/rws_xarray.html
        """

        vw = VectorWind(self.ds[self.u_name], self.ds[self.v_name])

        gut.myprint(f'Compute Vorticity...')
        eta = vw.absolutevorticity()
        if 'div' not in self.get_vars()():
            gut.myprint(f'Compute Divergence...')
            self.compute_divergence(can=can)
        div = self.ds['div']
        gut.myprint(f'Compute Rotation and gradient...')
        uchi, vchi = vw.irrotationalcomponent()
        etax, etay = vw.gradient(eta)

        gut.myprint("Compute Rossby Wave Source...")
        # Combine the components to form the Rossby wave source term.
        S = eta * -1. * div - (uchi * etax + vchi * etay)
        self.ds['S'] = S.rename('S')
        gut.myprint("... Computed Rossby Wave Source!")
        if can:
            for group in self.an_types:
                self.ds[f'S_an_{group}'] = tu.compute_anomalies(
                    dataarray=self.ds['S'], group=group)

        return self.ds

    def compute_streamfunction(self, can=True):
        vw = VectorWind(self.ds[self.u_name], self.ds[self.v_name])

        gut.myprint(f'Compute Stream Function...')
        sf, vp = vw.sfvp()
        self.ds['sf'] = sf.rename('sf')
        self.ds['vp'] = vp.rename('vp')

        if can:
            for an_type in self.an_types:
                sf_an = tu.compute_anomalies(
                    dataarray=self.ds['sf'], group=an_type)
                self.ds[sf_an.name] = sf_an
                vp_an = tu.compute_anomalies(
                    dataarray=self.ds['vp'], group=an_type)
                self.ds[vp_an.name] = vp_an

        return self.ds

    def helmholtz_decomposition(self):
        """
        Compute Helmholtz decomposition from u and v components
        see https://ajdawson.github.io/windspharm/latest/api/windspharm.standard.html
        """
        gut.myprint(f'Init Helmholtz decomposition wind vector...')
        vw = VectorWind(self.ds[self.u_name], self.ds[self.v_name])

        # Compute variables
        gut.myprint(f'Compute Helmholtz decomposition...')
        u_chi, v_chi, upsi, vpsi = vw.helmholtz()

        self.ds['u_chi'] = u_chi
        self.ds['v_chi'] = v_chi
        self.ds['u_psi'] = upsi
        self.ds['v_psi'] = vpsi

        return self.ds

    def compute_massstreamfunction(self,
                                   a=6376.0e3,
                                   g=9.81,
                                   meridional=True,
                                   c=None,
                                   can=True,
                                   ):
        """Calculate the mass streamfunction for the atmosphere.
        Based on a vertical integral of the meridional wind.
        Ref: Physics of Climate, Peixoto & Oort, 1992.  p158.

        `a` is the radius of the planet (default Isca value 6376km).
        `g` is surface gravity (default Earth 9.8m/s^2).
        lon_range allows a local area to be used by specifying boundaries as e.g. [70,100]
        dp_in - if no phalf and if using regularly spaced pressure levels, use this increment for
                integral. Units hPa.
        intdown - choose integratation direction (i.e. from surface to TOA, or TOA to surface).

        Returns an xarray DataArray of mass streamfunction.

        """

        for var in ['v_chi', 'u_chi']:
            if var == 'v_chi':
                lats = self.ds.lat
                lats = np.cos(lats*np.pi/180)
            else:
                var = 'u_chi'
                lats = 1  # No cosine factor for longitudes

            if var not in self.get_vars():
                self.helmholtz_decomposition()

            if c is None:
                c = 2*np.pi*a*lats / g

            # Compute Vertical integral of the Mass Streamfunction
            Psi = self.vertical_integration(var=var, c=c)

            if meridional:
                vname = 'msf_v'
            else:
                vname = 'msf_u'

            self.ds[vname] = Psi.rename(vname)

        if can:
            for an_type in self.an_types:
                sf_an = tu.compute_anomalies(
                    dataarray=self.ds[vname], group=an_type)
                self.ds[sf_an.name] = sf_an

        return self.ds


# %%
