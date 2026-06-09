from geoutils.geodata.helmholtz_decomposition import HelmholtzDecomposition
import xarray as xr
from windspharm.xarray import VectorWind


class Rossby_Wave_Source(HelmholtzDecomposition):
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
                 var_names=['S'],
                 init_hd=True,
                 **kwargs):

        super().__init__(ds_wind=ds_wind, ds_uwind=ds_uwind, ds_vwind=ds_vwind,
                         grid_step=grid_step,
                         load_nc=load_nc, can=False,  # Anomalies are later computed again!
                         init_hd=init_hd,
                         u_name=u_name, v_name=v_name,
                         **kwargs)

        if load_nc is None:
            self.S_dict = self.compute_rossby_wave_source(var_names=var_names)
            self.var_name = 'S'
            self.can = can

            self.ds['S'] = self.get_ds(S_dict=self.S_dict, var_names=var_names)['S']
            self.vars = self.get_vars()
            an_types = kwargs.pop('an_types', ['dayofyear'])

            if self.can is True:
                for vname in self.vars:
                    for an_type in an_types:
                        self.ds[f'{vname}_an_{an_type}'] = self.compute_anomalies(
                            dataarray=self.ds[vname],
                            group=an_type)
        else:
            self.load(load_nc=load_nc)

    def compute_rossby_wave_source(self, var_names):
        """Compute rossby wave source from u and v components
        see https://ajdawson.github.io/windspharm/latest/examples/rws_xarray.html
        """

        vw = VectorWind(self.u, self.v)
        return_dict = dict(w=vw)

        print(f'Compute Vorticity...', flush=True)
        eta = vw.absolutevorticity()
        rv = vw.vorticity()  # Relative Vorticity
        print("Compute Rossby Wave Source...",
              flush=True)
        div = vw.divergence()
        uchi, vchi = vw.irrotationalcomponent()
        etax, etay = vw.gradient(eta)

        # Combine the components to form the Rossby wave source term.
        S = eta * -1. * div - (uchi * etax + vchi * etay)
        S = xr.DataArray(data=S, coords=self.u.coords,
                         dims=self.u.dims, name='S')
        print("... Computed Rossby Wave Source!", flush=True)

        return_dict['S'] = S
        return_dict['eta'] = eta
        return_dict['div'] = div
        return_dict['rv'] = rv

        return return_dict

    def get_ds(self, S_dict, var_names=['S']):
        xr_arr = []
        S_dict_keys = list(S_dict.keys())
        for vname in var_names:
            if vname not in S_dict_keys:
                raise ValueError(f"{vname} not in S dict keys {S_dict_keys}!")
            xr_arr.append(S_dict[vname])
        ds = xr.merge(xr_arr)
        return ds
