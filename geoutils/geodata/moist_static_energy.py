#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 08:53:08 2020
Class for network of rainfall events
@author: Felix Strnad
"""
# %%

import geoutils.geodata.multilevel_pressure as mp
import metpy.calc as metcalc
from metpy.units import units
from importlib import reload
import numpy as np
import xarray as xr
import copy
import geoutils.utils.general_utils as gut
reload(mp)


class MoistStaticEnergy(mp.MultiPressureLevelDataset):
    """ Dataset for surface pressure.

    Args:
    ----------
    nc_file: str
        filename
    var_name: str
        Variable name of interest
    """

    def __init__(self,
                 data_nc_q=None,
                 data_nc_t=None,
                 data_nc_z=None,
                 plevels=None,
                 load_nc=None,
                 can=True,
                 vi=False,
                 **kwargs):
        reload(mp)
        q_kwargs = copy.deepcopy(kwargs)
        t_kwargs = copy.deepcopy(kwargs)
        z_kwargs = copy.deepcopy(kwargs)
        if load_nc is None:
            z_name = kwargs.pop('z_name', 'z')
            q_name = kwargs.pop('q_name', 'q')
            t_name = kwargs.pop('t_name', 't')

            ds_z = mp.MultiPressureLevelDataset(data_nc=data_nc_z,
                                                plevels=plevels,
                                                can=False,
                                                **z_kwargs)
            z = metcalc.geopotential_to_height(
                ds_z.ds[z_name] * units.m**2 / units.seconds**2)
            z = z.rename('z')

            ds_q = mp.MultiPressureLevelDataset(data_nc=data_nc_q,
                                                plevels=plevels,
                                                can=False,
                                                **q_kwargs)
            q = ds_q.ds[q_name]
            ds_t = mp.MultiPressureLevelDataset(data_nc=data_nc_t,
                                                plevels=plevels,
                                                can=False,
                                                **t_kwargs)
            t = ds_t.ds[t_name] * units.kelvin

            self.grid_step = ds_q.grid_step

            print(
                "Computed single components of wind dataset. Now compute moist_static_energy!")
            moist_static_energy = metcalc.moist_static_energy(height=z,
                                                              temperature=t,
                                                              specific_humidity=q)

            moist_static_energy = moist_static_energy.rename('mse')

            self.can = can
            self.ds = self.get_ds(q=q,
                                  t=t,
                                  z=z,
                                  moist_static_energy=moist_static_energy)
            self.vars = self.get_vars()

            self.compute_all_anomalies(**kwargs)
            # ds_q would be possible as well
            init_mask = kwargs.pop('init_mask', False)
            self.load_dataset_attributes(base_ds=ds_t,
                                         init_mask=init_mask,)
        else:
            self.load(load_nc)

        if vi:
            self.set_vertical_integral()

    def get_vertical_integral(self,
                              c=1):
        """Calculate the mass streamfunction for the atmosphere.
        Based on a vertical integral of the meridional wind.
        Ref: Physics of Climate, Peixoto & Oort, 1992.  p158.

        dp_in - if no phalf and if using regularly spaced pressure levels, use this increment for
                integral. Units hPa.
        intdown - choose integratation direction (i.e. from surface to TOA, or TOA to surface).

        Returns an xarray DataArray of mass streamfunction.

        """

        gut.myprint('Compute Vertical Integral of MSE!')
        if 'mse' not in self.get_vars():
            self.compute_moist_static_energy()

        mse_plevel = self.ds['mse']

        dp = xr.DataArray(np.diff(mse_plevel.plevel, prepend=0)*100.,  # 100 because of hPa to bar
                          coords={'plevel': mse_plevel.plevel})

        # Compute Vertical integral along the pressure level dimension
        # Integral is from top of atmosphere to surface
        vi_mse = c*np.cumsum(mse_plevel*dp,
                             axis=mse_plevel.dims.index('plevel'))

        return vi_mse

    def set_vertical_integral(self,
                              can=True,
                              an_types=None,
                              c=1):
        vname = 'vi_mse'

        vi_mse = self.get_vertical_integral(c=c)

        self.ds[vname] = vi_mse
        if an_types is None:
            an_types = self.an_types
        if can:
            for an_type in an_types:
                self.ds[f'{vname}_an_{an_type}'] = self.compute_anomalies(
                    dataarray=vi_mse,
                    group=an_type)

    def get_ds(self, q, t, z, moist_static_energy):
        ds = xr.merge([q,
                       t,
                       z,
                       moist_static_energy]
                      )

        return ds


# %%
