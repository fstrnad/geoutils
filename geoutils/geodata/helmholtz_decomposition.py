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


    def helmholtz_decomposition(self, ds):
        """
        Compute Helmholtz decomposition from u and v components
        see https://ajdawson.github.io/windspharm/latest/api/windspharm.standard.html
        """
        gut.myprint(f'Init Helmholtz decomposition wind vector...')
        vw = VectorWind(ds[self.u_name], ds[self.v_name])
        return_dict = dict(w=vw)
        # Compute variables
        gut.myprint(f'Compute Helmholtz decomposition...')
        u_chi, v_chi, upsi, vpsi = vw.helmholtz()

        return_dict['u_chi'] = u_chi
        return_dict['v_chi'] = v_chi
        return_dict['u_psi'] = upsi
        return_dict['v_psi'] = vpsi

        return return_dict

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

    def get_streamfunction(self):
        vw = VectorWind(self.u, self.v)
        return_dict = dict(w=vw)
        print(f'Compute Stream Function...', flush=True)
        sf, vp = vw.sfvp()
        return_dict['sf'] = sf
        return_dict['vp'] = vp

        return return_dict

    def get_massstreamfunction(self,
                               a=6376.0e3,
                               g=9.81,
                               meridional=True,
                               c=None,
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

        if meridional:
            var = 'v_chi'
            lats = v_bar.lat
            lats = np.cos(lats*np.pi/180)
        else:
            var = 'u_chi'
            lats = 1 # No cosine factor for longitudes

        if c is None:
            c = 2*np.pi*a*lats / g

        # Compute Vertical integral of the Mass Streamfunction
        Psi = self.vertical_integration(var=var)

        return Psi

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
