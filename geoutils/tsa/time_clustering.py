import scipy as sp
import geoutils.utils.statistic_utils as sut
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, OPTICS
from sklearn.mixture import GaussianMixture
import numpy as np
import xarray as xr
from kneed import KneeLocator
import sklearn.metrics as skm
from importlib import reload
import geoutils.utils.general_utils as gut
import geoutils.plotting.plots as cplt
from tqdm import tqdm
reload(gut)
reload(cplt)


def k_means_clustering(data,
                       **kmeans_kwargs):

    k_method = kmeans_kwargs.pop('k_method', ' elbow')
    max_iter = kmeans_kwargs.pop('max_iter', 1000)
    n_init = kmeans_kwargs.pop('n_init', 50)
    plot_statistics = kmeans_kwargs.pop('plot_statistics', False)
    rem_outlayers = kmeans_kwargs.pop('rm_ol', False)
    sc_th = kmeans_kwargs.pop('sc_th', 0.05)

    k = kmeans_kwargs.pop('n_clusters', None)
    if k is not None:
        kmeans = KMeans(n_clusters=k,
                        init="k-means++",
                        max_iter=max_iter,
                        n_init=n_init,
                        **kmeans_kwargs)
        kmeans.fit(data)

    else:
        gut.myprint(f'Select number of clusters with {k_method}')
        sse = []
        sscore = []
        krange = np.arange(2, 11)
        for k in tqdm(krange):
            kmeans = KMeans(n_clusters=k,
                            init="k-means++",
                            max_iter=max_iter,
                            **kmeans_kwargs)
            kmeans.fit(data)
            score = skm.silhouette_score(data, kmeans.labels_)
            sse.append(kmeans.inertia_)
            sscore.append(score)
        if plot_statistics:
            cplt.plot_xy(x_arr=[krange], y_arr=[sscore],
                         xlabel="Number of Clusters",
                         ylabel="Shilhouette Score",
                         )

        if k_method == 'elbow':
            kl = KneeLocator(
                krange, sse, curve="convex", direction="decreasing"
            )
            k = int(kl.elbow)
        elif k_method == 'silhouette':
            k = krange[np.argmax(sscore)]
        gut.myprint(f'Method:{k_method}: Get {k} number of clusters!')

        kmeans = KMeans(n_clusters=k,
                        init="k-means++",
                        max_iter=max_iter,
                        **kmeans_kwargs)
        kmeans.fit(data)

    Z = kmeans.predict(data)

    if plot_statistics:
        sample_sc = skm.silhouette_samples(data, Z)
        cplt.plot_xy(x_arr=[np.arange(len(Z))],
                     y_arr=[sample_sc],
                     ls_arr=['None'],
                     mk_arr=['.'],
                     ylabel="Shilhouette Score",
                     ylim=(0., .1))

    if rem_outlayers:
        gut.myprint('Remove Outlayers...')
        sample_sc = skm.silhouette_samples(data, Z)
        sign_Z = np.where(sample_sc >= sc_th)[0]
        gut.myprint(f'Removed {1 - np.count_nonzero(sign_Z)/len(Z)} of all tps!')
        return Z, sign_Z

    return Z, None


def gm_clustering(data,
                  **kwargs):

    max_iter = kwargs.pop('max_iter', 1000)
    n_init = kwargs.pop('n_init', 10)
    k = kwargs.pop('n_clusters', 2)

    gut.myprint(f'Get {k} clusters!')

    gm = GaussianMixture(n_components=k,
                         init_params="k-means++",
                         max_iter=max_iter,
                         n_init=n_init,
                         **kwargs).fit(data)

    Z = gm.predict(data)
    return Z


def dbscan_clustering(data, **kwargs):
    """Perform DBSCAN clustering from vector array or distance matrix.
    See as well https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

    Args:
        data (np.Array): 2-d feature array

    Returns:
        labels: -1 considered as noise
    """
    # The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    eps = kwargs.pop('eps', 0.99)
    # The metric to use when calculating distance between instances in a feature array
    metric = kwargs.pop('metric', 'correlation')

    min_samples = kwargs.pop('min_samples', 5)

    # Perform DBSCAN clustering from vector array or distance matrix.
    clustering = DBSCAN(eps=eps, min_samples=min_samples,
                        metric=metric).fit(data)

    # Cluster labels for each point in the dataset given to fit(). Noisy samples are given the label -1.

    Z = clustering.labels_

    return Z


def optics_clustering(data, **kwargs):
    """Perform OPTICS clustering from vector array or distance matrix.

    Args:
        data (np.Array): 2-d feature array

    Returns:
        labels: -1 considered as noise
    """
    # The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    eps = kwargs.pop('eps', None)
    # The metric to use when calculating distance between instances in a feature array
    metric = kwargs.pop('metric', 'correlation')

    min_samples = kwargs.pop('min_samples', 2)

    # Perform DBSCAN clustering from vector array or distance matrix.
    clustering = OPTICS(eps=eps, min_samples=min_samples,
                        metric=metric).fit(data)

    # Cluster labels for each point in the dataset given to fit(). Noisy samples are given the label -1.

    Z = clustering.labels_

    return Z


def agglomerative_clustering(data, **kwargs):
    """Perform DBSCAN clustering from vector array or distance matrix.
    See as well https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

    Args:
        data (np.Array): 2-d feature array

    Returns:
        labels: -1 considered as noise
    """
    # The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    n_clusters = kwargs.pop('n_clusters', None)
    # The metric to use when calculating distance between instances in a feature array
    metric = kwargs.pop('metric', 'correlation')

    linkage = kwargs.pop('linkage', 'complete')

    # if metric == 'correlation':
    #     distances =

    # Perform DBSCAN clustering from vector array or distance matrix.
    clustering = AgglomerativeClustering(n_clusters=n_clusters,
                                         linkage=linkage,
                                         affinity=metric).fit(data)

    # Cluster labels for each point in the dataset given to fit(). Noisy samples are given the label -1.

    Z = clustering.labels_

    return Z


def tps_cluster_2d_data(data_arr, tps,
                        method='kmeans',
                        gf=(0, 0),
                        **kwargs):
    """possible key-word args:
    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,
        }

    Args:
        data_arr (list): list of 3-d arrays.
        n (int): number of clusters
        tps (list): time points.
    """

    # The data needs to be  reshped in a row-wise of array. Therefore, each row is an object or data.
    # We always check if data is 3d (x,y,time)
    coll_data = []
    key_names = kwargs.pop('key_names', None)
    for data in data_arr:
        if len(data.shape) != 3:
            raise ValueError('Data needs to be of shape 3 (time, x, y)')

        if len(tps) != data.shape[0]:
            raise ValueError(
                f'Number of tps and time points in data are not equal!'
            )

        if isinstance(data, xr.DataArray):
            data = data.data

        if gf[0] != 0 or gf[1] != 0:
            gut.myprint(f'Apply Gaussian Filter with sigma = {gf}!')
            sigma = [gf[1], gf[0]]  # sigma_y, sigma_x

            for idx, dp in enumerate(data):
                data[idx] = sp.ndimage.filters.gaussian_filter(
                    dp, sigma, mode='constant')

        # Reshapedata
        new_arr = data.reshape(*data.shape[:1], -1)
        coll_data.append(new_arr)

    rm_ol = kwargs.pop('rm_ol', False)

    # concatenate along 1 dimension
    data_input = np.concatenate(coll_data, axis=1)
    if method == 'kmeans':
        Z, sign_Z = k_means_clustering(data=data_input,  rm_ol=rm_ol, **kwargs)
    elif method == 'gm':
        Z, sign_Z = gm_clustering(data=data_input, **kwargs)
    elif method == 'dbscan':
        Z, sign_Z = dbscan_clustering(data=data_input, **kwargs)
    elif method == 'optics':
        Z, sign_Z = optics_clustering(data=data_input, **kwargs)
    elif method == 'agglomerative':
        Z, sign_Z = agglomerative_clustering(data=data_input, **kwargs)
    else:
        raise ValueError(f'Method {method} not implemented yet!')

    if rm_ol:
        Z = Z[sign_Z]
        tps = tps[sign_Z]

    grp_ids = np.unique(Z)
    grp_tps_dict = dict()
    for gid in grp_ids:
        idx_grp = np.where(Z == gid)[0]
        if key_names is not None:
            if len(key_names) != len(grp_ids):
                raise ValueError(f'Not same number of key names as groups!')
            keyname = key_names[gid]
            grp_tps_dict[keyname] = tps[idx_grp]
        else:
            grp_tps_dict[gid] = tps[idx_grp]

    return grp_tps_dict
