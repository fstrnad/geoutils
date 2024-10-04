import geoutils.utils.general_utils as gut
import geoutils.utils.statistic_utils as sut
import geoutils.utils.time_utils as tu
import numpy as np
import xarray as xr
import geoutils.tsa.pca.eof as eof
import geoutils.tsa.pca.pca_utils as pca_utils
from importlib import reload
reload(sut)
reload(tu)
reload(gut)
reload(pca_utils)


class MultivariatePCA(eof.SpatioTemporalPCA):
    """PCA of spatio-temporal data for multivariate datasets.

    Wrapper for sklearn.decomposition.PCA with xarray.DataArray input.

    Args:
        ds (xr.DataArray or xr.Dataset): Input dataarray to perform PCA on.
            Array dimensions (time, 'lat', 'lon')
        n_components (int): Number of components for PCA
    """

    def __init__(self, ds, n_components,
                 normalize=True, **kwargs):

        self.normalize = normalize
        self.X, self.ids_notNaN = self.get_input_data(ds=ds)
        self.dims = gut.get_dims(ds)
        if 'time' in self.dims:
            self.dims.remove('time')
        self.pca = self.apply_pca(X=self.X, n_components=n_components,
                                  **kwargs)

        self.n_components = self.pca.n_components
        self.eof_nums = None
        if self.normalize:
            self.std = ds.std(dim='time')
            self.mean = ds.mean(dim='time')

    def get_input_data(self, ds: xr.Dataset) -> np.ndarray:
        """Get input data for PCA.

        Args:
        -----
        ds (xr.Dataset): Dataset to apply PCA on.

        Returns:
        --------
        X (np.ndarray): Input data for PCA.
        """
        self.mvars = True
        self.vars = gut.get_varnames_ds(ds=ds)
        if isinstance(ds, xr.Dataset):
            if len(self.vars) > 1:
                gut.myprint(f'Multivariate PCA vars: {self.vars}')
            else:
                gut.myprint(f'Univariate PCA var: {self.vars[0]}')
        else:
            gut.myprint(f'Univariate PCA var with dataarray')
            self.mvars = False

        X, idsnotnan = pca_utils.map2flatten(ds,
                                             normalize=self.normalize)

        return X, idsnotnan

    def get_variables(self):
        """Get variables of PCA.

        Returns:
        --------
        vars (list): List of variables.
        """
        if self.mvars:
            if self.vars is None:
                raise ValueError('No variables found.')

            return self.vars
        else:
            gut.myprint(f'Univariate PCA with dataarray')
            return [self.vars]
