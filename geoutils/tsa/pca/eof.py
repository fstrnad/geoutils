"""PCA for spatio-temporal data."""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from climnet import preproc
from sklearn.preprocessing import normalize


class SpatioTemporalPCA:
    """PCA of spatio-temporal data.

    Wrapper for sklearn.decomposition.PCA with xarray.DataArray input.

    Args:
        ds (xr.DataArray or xr.Dataset): Input dataarray to perform PCA on.
            Array dimensions (time, 'lat', 'lon')
        n_components (int): Number of components for PCA
        weighting: None or 'coslat' or 1-dim weight array containing weight for each grid point
    """
    def __init__(self, ds, n_components,weighting='uniform',covestim=None, orienteofs=None, **kwargs):
        self.ds = ds
        self.n_components = n_components
        self.X, self.ids_notNaN = preproc.map2flatten(self.ds)
        if weighting is None:
            self.weights = None
        else:
            # in SpatioTemporalPCA implement weighting from North (1982):
            # A = sum_i w_i
            # instead of C_ij use sqrt(w_i) C_ij sqrt(w_j)/A
            # compute eigenvecs and eigenvals, transform eigenvecs to the grid-invariant EOFs via: eigenvecs_i * sqrt(A)/sqrt(w_i)
            if weighting == 'coslat':
                self.weights = np.cos(self.X['lat'].data*np.pi/180)
            elif weighting == 'uniform':
                self.weights = np.ones_like(self.X['lat'].data)
            else:
                assert len(weighting) == self.X.data.shape[1]
                self.weights = weighting
            if np.any(self.weights<0):
                raise ValueError('Area weights should always be >=0.')
            self.totalweight=np.sum(self.weights)
            # downweigh data does the same as downweighting C_ij. not true!!!
            #self.X_weighted = self.X.data * np.tile(np.sqrt(self.weights),(self.X.shape[0],1))
            
        if self.weights is None:
            if covestim is not None:
                raise ValueError('covestim without weights is not implemented.')
            self.pca = PCA(n_components=n_components)
            self.pca.fit(self.X.data)
        else:
            Wsqrt=np.diag(np.sqrt(self.weights))
            if covestim is None:
                eigvals, eigvecs = np.linalg.eigh((Wsqrt @ self.X.data.T @ self.X.data @ Wsqrt)/np.sum(self.weights))
            else:
                assert covestim.shape == (len(self.weights),len(self.weights))
                eigvals, eigvecs = np.linalg.eigh((Wsqrt @ covestim @ Wsqrt)/np.sum(self.weights))
            eigvals = eigvals[::-1]
            eigvecs = eigvecs[:,::-1]
            eofs = eigvecs * np.sqrt(np.sum(self.weights)) / np.tile(np.sqrt(self.weights).reshape((len(self.weights),1)),(1,eigvecs.shape[1]))
            #normalize matrix by columns
            self.eofs = normalize(eofs, axis=0, norm='l2')
            self.eigvals = eigvals
            if orienteofs is not None:
                for idx in range(n_components):
                    if np.linalg.norm(self.eofs.T[idx,:]-orienteofs[idx,:])>np.linalg.norm(-self.eofs.T[idx,:]-orienteofs[idx,:]):
                        self.eofs[:,idx] = -self.eofs[:,idx]


    def get_eofs(self):
        """Return principal spatial directions of PCA.

        Return:
            components (xr.dataarray): Size (n_components, N_x, N_y)
        """
        # EOF maps
        eof_map = []
        if self.weights is None:
            eofarray=self.pca.components_
        else:
            eofarray=self.eofs.T[:self.n_components,:]
        for i, comp in enumerate(eofarray):
            eof = preproc.flattened2map(comp, self.ids_notNaN)
            eof = eof.drop(['time'])
            eof_map.append(eof)
        return xr.concat(eof_map, dim='eof')
    

    def get_principal_components(self):
        """Returns time evolution of components.

        Return:
            time_evolution (xr.Dataarray): Principal components of shape (n_components, time)
        """
        time_evolution = []
        if self.weights is None:
            eofarray=self.pca.components_
        else:
            eofarray=self.eofs.T[:self.n_components,:]
        for i, comp in enumerate(eofarray):
            ts = self.X.data @ comp

            da_ts = xr.DataArray(
                data=ts,
                dims=['time'],
                coords=dict(time=self.X['time']),
            )
            time_evolution.append(da_ts)
        return xr.concat(time_evolution, dim='eof')
    

    def get_explainedVariance(self):
        # this is conserved
        if self.weights is None:
            return self.pca.explained_variance_ratio_
        else:
            return self.eigvals[:self.n_components] / np.sum(self.eigvals)
    
    def inverse_transform(self, z, newdim='x'):
        """Reconstruct the dataset from components and time-evolution.

        Args:
            z (np.ndarray): Low dimensional vector of size (time, n_components)
            newdim (str, optional): Name of dimension. Defaults to 'x'.

        Returns:
            _type_: _description_
        """
        if self.weights is None:
            comps = self.pca.components_
        else:
            comps = self.eofs.T[:self.n_components,:]
        # if self.weights is not None:
        #     # invert weighting transform to grid-invariant EOFs
        #     comps = comps * np.sqrt(self.totalweight) / np.tile(np.sqrt(self.weights),(comps.shape[0],1))
        # TODO: this was z.T but should be z?!
        reconstruction = z @ comps

        rec_map = []
        for rec in reconstruction:
            x = preproc.flattened2map(rec, self.ids_notNaN)
            rec_map.append(x)

        rec_map = xr.concat(rec_map, dim=newdim)

        return rec_map