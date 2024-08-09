"""PCA for spatio-temporal data.
Parts adopted from: https://github.com/jakob-schloer/LatentGMM/tree/main"""
import geoutils.utils.general_utils as gut
import geoutils.utils.statistic_utils as sut
import geoutils.utils.time_utils as tu
import geoutils.tsa.pca.pca_utils as pca_utils
import numpy as np
import xarray as xr
from sklearn.decomposition import PCA

from importlib import reload
reload(sut)
reload(tu)
reload(pca_utils)


class SpatioTemporalPCA:
    """PCA of spatio-temporal data.

    Wrapper for sklearn.decomposition.PCA with xarray.DataArray input.

    Args:
        ds (xr.DataArray or xr.Dataset): Input dataarray to perform PCA on.
            Array dimensions (time, 'lat', 'lon')
        n_components (int): Number of components for PCA
    """

    def __init__(self, ds, n_components, **kwargs):
        self.ds = ds
        self.dims = gut.get_dims(self.ds)
        if 'time' in self.dims:
            self.dims.remove('time')
        self.X, self.ids_notNaN = pca_utils.map2flatten(self.ds)
        self.dims = gut.get_dims(self.ids_notNaN)
        # PCA
        self.pca = self.apply_pca(X=self.X, n_components=n_components,
                                  **kwargs)

        self.n_components = self.pca.n_components
        self.eof_nums = None

    def apply_pca(self, X: np.ndarray, n_components: int, **kwargs):
        """Apply PCA to a dataset.

        Args:
        -----
        ds (xr.Dataset): Dataset to apply PCA on.
        n_components (int): Number of components for PCA.
        """
        # PCA
        pca = PCA(n_components=n_components, whiten=True)
        pca.fit(X.data)

        return pca

    def get_eofs(self):
        """Return components of PCA.

        Parameters:
        -----------
        normalize: str
            Normalization type of components.

        Return:
        -------
        components: xr.dataarray (n_components, N_x, N_y)
        """
        # EOF maps
        components = self.pca.components_
        eof_map = []
        for i, comp in enumerate(components):
            eof = pca_utils.flattened2map(comp, self.ids_notNaN)
            eof_map.append(eof)

        return xr.concat(eof_map, dim='eof')

    def get_principal_components(self):
        """Returns time evolution of components.

        Args:
        -----
        normalize: str or None
            Method to normalize the time-series

        Return:
        ------
        time_evolution: np.ndarray (n_components, time)
        """
        pc = self.pca.transform(self.X.data)
        da_pc = xr.DataArray(
            data=pc,
            coords=dict(time=self.X['time'], eof=np.arange(
                0, self.n_components)),
        )
        self.eof_nums = da_pc.eof.data
        return da_pc

    def get_eof_nums(self):
        if self.eof_nums is None:
            self.eof_nums = self.get_principal_components().eof.data
        return self.eof_nums

    def explained_variance(self):
        return self.pca.explained_variance_ratio_

    def transform(self, x: xr.Dataset):
        x_flat, ids_notNaN = pca_utils.map2flatten(x)
        assert len(x_flat['z']) == len(self.X['z'])
        z = self.pca.transform(x_flat.data)

        return z

    def transform_reduced(self, x: xr.Dataset,
                          reduzed_eofs: np.ndarray) -> np.ndarray:
        z_all = self.transform(x)
        reduzed_eofs = np.array(reduzed_eofs)
        eofs = self.get_eof_nums()
        if np.array_equal(reduzed_eofs, eofs):
            return z_all
        else:
            for red_eof in reduzed_eofs:
                assert red_eof in eofs, f'EOF {red_eof} not in dataset'
            z_trafo = z_all.T[reduzed_eofs]

            return z_trafo.T

    def transform_reduced_ts_corr(self, x: xr.Dataset,
                                  ts: xr.DataArray,
                                  num_eofs=None, min_corr=0):
        sel_eofs = pca_utils.get_reduced_eofs(x=x, sppca=self, ts=ts,
                                              num_eofs=num_eofs, min_corr=min_corr)

        z_trafo = self.transform_reduced(x=x, reduzed_eofs=sel_eofs)

        return z_trafo

    def inverse_transform(self, z: np.ndarray, newdim='time', coords=None) -> xr.Dataset:
        """Transform from eof space to data space.

        Args:
            z (np.ndarray): Principal components of shape (n_samples, n_components)
            newdim (str, optional): Name of dimension of n_samples.
                Can be also a pd.Index object. Defaults to 'time'.

        Returns:
            xr.Dataset: Transformed PCs to grid space.
        """
        x_hat_flat = self.pca.inverse_transform(z)

        x_hat_map = []
        for x_flat in x_hat_flat:
            x_hat = pca_utils.flattened2map(x_flat, self.ids_notNaN)
            x_hat = x_hat.drop([dim for dim in list(
                x_hat.dims) if dim not in self.dims])
            x_hat_map.append(x_hat)

        x_hat_map = xr.concat(x_hat_map, dim='time')

        if newdim != 'time':
            x_hat_map = x_hat_map.rename({'time': newdim})
            x_hat_map = x_hat_map.assign_coords({newdim: coords})

        return x_hat_map

    def reconstruction(self, x: xr.Dataset) -> xr.Dataset:
        """Reconstruct the dataset from components and time-evolution.
        This is a encoder-decoder process."""
        stack_dim = [dim for dim in list(x.dims) if dim not in [
            'lat', 'lon']][0]
        z = self.transform(x)
        x_hat = self.inverse_transform(z, newdim=x[stack_dim])

        return x_hat

    def eofs_real_space(self):
        """Return EOFs in real space."""
        num_eofs = self.get_eof_nums()
        len_eofs = len(num_eofs)
        basis_vectors = gut.identity_matrix(len_eofs)
        real_eofs = self.inverse_transform(basis_vectors,
                                           newdim='eof',
                                           coords=num_eofs)
        return real_eofs
