"""PCA for spatio-temporal data."""

import numpy as np
import xarray as xr

from geoutils.tsa.pca.custom_pca import CustomPCA
import geoutils.utils.statistic_utils as sut
import geoutils.utils.spatial_utils as sput
import geoutils.utils.general_utils as gut


class SpatioTemporalPCA:
    """PCA of spatio-temporal data.

    Wrapper for sklearn.decomposition.PCA with xarray.DataArray inpsut.

    Parameters:
    -----------
    da: geoutils.BaseDataset of  dim=(time, x, y)
        Input dataarray to perform PCA on.
    n_components: int
        Number of components for PCA
    """

    def __init__(self, dataset, n_components,
                 rotation=None,
                 normalize_ts=False,
                 var_name=None,
                 **kwargs):

        self.ds = dataset
        if var_name is None:
            self.da = self.ds.get_da()
            self.var_name = self.ds.var_name
        else:
            self.da = self.ds.ds[var_name]
            self.var_name = var_name
        # create data matrix
        if normalize_ts:
            self.da = sut.standardize(self.da)

        gut.myprint(
            f'Prepare the dataset to TxN array for variable {self.var_name}!')
        self.F = self.ds.flatten_array(self.var_name)

        # run pca
        self.run_pca(n_components=n_components,
                     rotation=rotation,
                     **kwargs)
        self.n_components = self.pca.n_components
        gut.myprint(f'Finished!')

    def run_pca(self, n_components,
                rotation=None, **kwargs):
        # run pca
        if rotation not in ['varimax', 'None', None]:
            raise ValueError(f'rotation {rotation} is not available!')
        gut.myprint(f"Now run PCA with rotation {rotation}!")
        self.pca = CustomPCA(n_components=n_components,
                             rotation=rotation,
                             **kwargs)
        self.pca.fit(self.F)

        self.n_components = self.pca.n_components

    def get_components(self, scale=None, q=None, inverse=False):
        """Return components of PCA.

        Parameters:
        -----------
        normalize: str
            Normalization type of components.

        Return:
        -------
        components: xr.dataarray (n_components, N_x, N_y)
        """

        components = self.scale_components(scale=scale)

        comp_map = []
        q_map_arr = []
        for i, comp in enumerate(components):
            pca_map = self.ds.get_map(comp,
                                      name=f'EOF')
            if inverse is True:
                max_val = np.max(pca_map)
                min_val = np.min(pca_map)
                if np.abs(min_val) > np.abs(max_val):  # Varimax might flip the sign!
                    pca_map *= -1
            comp_map.append(pca_map)
            if q is not None:
                q_map = xr.where(
                    pca_map >= np.nanquantile(pca_map, q=q),
                    1, 0)
                q_map_arr.append(q_map)
        return xr.concat(comp_map, dim='comp')

    def scale_components(self, scale=None):
        # scale components by its variance explanation power
        if scale == "var":
            components = np.einsum('i, ij -> ij',
                                   self.pca.explained_variance_, self.pca.components_)
        # scale components by its standard deviation
        elif scale == "std":
            std_comp = 1 / np.std(self.pca.components_, axis=1)
            components = np.einsum('i, ij -> ij',
                                   std_comp, self.pca.components_)
        elif scale is None:
            components = self.pca.components_
        else:
            raise ValueError(
                f"Chosen normalization type '{scale}' does not exist!")
        return components

    def get_q_maps(self, scale=None, q=None):
        components = self.scale_components(scale=scale)

        q_map_arr = []
        if q is not None:
            for i, comp in enumerate(components):
                pca_map = self.ds.get_map(comp,
                                          name=f'EOF')

                q_map = xr.where(
                    pca_map >= np.nanquantile(pca_map, q=q),
                    1, 0)
                q_map_arr.append(q_map)
            return xr.concat(q_map_arr, dim='comp')
        else:
            return None

    def get_timeEvolution(self,
                          q=None,
                          inverse=False,
                          def_map=None,
                          ):
        """Returns time evolution of components.

        Args:
        -----
        normalize: str or None
            Method to normalize the time-series

        Return:
        ------
        time_evolution: np.ndarray (n_components, time)
        """
        time_evolution = []
        pca_comp = self.pca.components_
        if inverse is True:
            pca_comp *= -1

        if q is not None:
            q_val = np.nanquantile(pca_comp, q=q)
            pca_comp = np.where(pca_comp >= q_val, pca_comp, 0)

        time = self.ds.ds.time
        for i, comp in enumerate(pca_comp):
            if def_map is not None:
                flat_def_map = self.ds.flatten_array(def_map,
                                                     time=False, check=False)
                comp = np.where(flat_def_map == 1, comp, 0)

                n_nans = np.count_nonzero(~np.isnan(comp))
                print(n_nans)

            ts = self.F @ comp
            # ts /= ts.std()
            ts = sut.standardize(ts)  # standardize data
            xr_ts = xr.DataArray(data=ts,
                                 name=f'eof{i}',
                                 coords={"time": time},
                                 dims=["time"])
            time_evolution.append(xr_ts)
        return xr.merge(time_evolution)

    def get_explainedVariance(self):
        return self.pca.explained_variance_ratio_

    def reconstruction(self, name='reconstruction'):
        """Reconstruct the dataset from components and time-evolution."""
        ts = self.get_timeEvolution()
        rec_F = ts.T @ self.pca.components_

        rec_map = []
        for rec_arr in rec_F:
            rec_map.append(self.ds.get_map(rec_arr, name=name))

        rec_map = xr.concat(rec_map, dim=self.da.time)

        return rec_map

    def get_pca_loc_dict(self, q=None, inverse=False, tq=None):
        if self.ds.mask is None:
            gut.myprint('Init as well a mask for dataset!')
            self.ds.init_mask(init_mask=True)
        maps = self.get_components(q=q, inverse=inverse)
        q_maps = self.get_q_maps(q=q)
        ts = self.get_timeEvolution(q=tq, inverse=inverse)
        expl_var = self.get_explainedVariance()
        if len(maps) != len(ts) or len(maps) != len(expl_var):
            raise ValueError(
                'ERROR Time compents and spatial_components not of same length!')
        pca_dict = {}
        gut.myprint('Now prepare PCA dictionary...')
        for idx, this_map in enumerate(maps):

            pca_dict[idx] = dict(ts=ts[f'eof{idx}'],
                                 map=this_map,
                                 q_map=q_maps[idx] if q is not None else None,
                                 ev=expl_var[idx]
                                 )

        gut.myprint('... finished!')
        return pca_dict
