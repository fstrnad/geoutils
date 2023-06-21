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
import geoutils.utils.file_utils as fut
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
                 vi=True,
                 **kwargs):
        reload(mp)
        q_kwargs = copy.deepcopy(kwargs)
        t_kwargs = copy.deepcopy(kwargs)
        z_kwargs = copy.deepcopy(kwargs)
        if load_nc is None:
            self.z_name = kwargs.pop('z_name', 'z')
            self.q_name = kwargs.pop('q_name', 'q')
            self.t_name = kwargs.pop('t_name', 't')

            all_files = data_nc_q + data_nc_t + data_nc_z
            for file in all_files:
                fut.print_file_location_and_size(filepath=file, verbose=False)
            time_range = fut.get_file_time_range(all_files)

            ds_z = mp.MultiPressureLevelDataset(data_nc=data_nc_z,
                                                plevels=plevels,
                                                can=False,
                                                time_range=time_range,
                                                **z_kwargs)
            z = metcalc.geopotential_to_height(
                ds_z.ds[self.z_name] * units.m**2 / units.seconds**2)
            z = z.rename('z')

            ds_q = mp.MultiPressureLevelDataset(data_nc=data_nc_q,
                                                plevels=plevels,
                                                can=False,
                                                time_range=time_range,
                                                **q_kwargs)
            q = ds_q.ds[self.q_name]
            ds_t = mp.MultiPressureLevelDataset(data_nc=data_nc_t,
                                                plevels=plevels,
                                                can=False,
                                                time_range=time_range,
                                                **t_kwargs)
            t = ds_t.ds[self.t_name] * units.kelvin

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

            if vi:
                self.ds['vi_mse'] = self.vertical_integration(var='mse')

            self.compute_all_anomalies(**kwargs)
            # ds_q would be possible as well
            init_mask = kwargs.pop('init_mask', False)
            self.load_dataset_attributes(base_ds=ds_t,
                                         init_mask=init_mask,)
        else:
            self.load(load_nc)

    def get_ds(self, q, t, z, moist_static_energy):
        ds = xr.Dataset({self.q_name: q,
                         self.t_name: t,
                         self.z_name: z,
                         'mse': moist_static_energy})

        return ds


# %%
