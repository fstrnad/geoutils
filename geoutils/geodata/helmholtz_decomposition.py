import geoutils.geodata.wind_dataset as wds

import numpy as np
import xarray as xr
import geoutils.utils.time_utils as tu
import geoutils.utils.general_utils as gut
from windspharm.xarray import VectorWind
from importlib import reload
reload(wds)


class HelmholtzDecomposition(wds.Wind_Dataset):
    """ Dataset for surface pressure.

    Args:
    ----------
    nc_file: str
        filename
    var_name: str
        Variable name of interest
    """

    def __init__(self, ds_wind=None,
                 ds_uwind=None,
                 ds_vwind=None,
                 grid_step=None,
                 load_nc=None,
                 can=True,
                 u_name='u',
                 v_name='v',
                 init_msf=True,
                 **kwargs):

        self.an_types = kwargs.pop('an_types', ['dayofyear'])
        self.can = can
        self.u_name = u_name
        self.v_name = v_name
        if load_nc is None:
            if ds_wind is not None:
                ds = ds_wind.ds
                self.grid_step = ds_wind.grid_step
                self.wind_vars = ds_wind.get_vars()

            else:
                ds = xr.merge(
                    [ds_uwind[self.u_name], ds_vwind[self.v_name]])
                self.grid_step = grid_step

            S_dict = self.helmholtz_decomposition(ds=ds)
            self.ds = self.get_ds(S_dict=S_dict)
            del ds  # Delete the unnecessary wind dataset!

            if init_msf:
                self.set_massstreamfunction(c=1,
                                            can=False)

            self.compute_all_anomalies()

            # self.load_dataset_attributes(base_ds=ds_wind)

        else:
            self.load(load_nc=load_nc)


    def get_ds(self, S_dict,
               ds_wind=None):

        S_dict_keys = list(S_dict.keys())
        var_names = ['u_chi', 'v_chi', 'u_psi', 'v_psi']
        for vname in var_names:
            if vname not in S_dict_keys:
                raise ValueError(
                    f'ERROR! Variable {vname} not in wind dataset')
        self.ds = xr.merge([
            # self.ds,
            S_dict['u_chi'],
            S_dict['v_chi'],
            S_dict['u_psi'],
            S_dict['v_psi']])
        return self.ds

    

    def set_massstreamfunction(self,
                               a=6376.0e3,
                               g=9.81,
                               can=True,
                               an_types=None,
                               c=None):
        gut.myprint(f'Compute Mass Streamfunction...')
        for meridional in [True, False]:
            vname = 'mf'

            psi = self.get_massstreamfunction(a=a,
                                              g=g,
                                              meridional=meridional,
                                              c=c)
            if meridional:
                vname += '_v'
            else:
                vname += '_u'

            self.ds[vname] = psi

        gut.myprint(f'... Finished')
