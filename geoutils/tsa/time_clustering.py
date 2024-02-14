import pandas as pd
import hdbscan
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
import geoutils.utils.spatial_utils as sput
import scipy as sp
import geoutils.utils.statistic_utils as sut
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, OPTICS, Birch
from sklearn.mixture import GaussianMixture
import scipy.cluster.hierarchy as shr
from scipy.spatial.distance import pdist
import numpy as np
import xarray as xr
from kneed import KneeLocator
import sklearn.metrics as skm
from importlib import reload
import geoutils.utils.general_utils as gut
import geoutils.utils.time_utils as tu
import geoutils.plotting.plots as cplt
from tqdm import tqdm
reload(gut)
reload(cplt)
reload(sput)


def plot_statistics(data, sc_th, Z):
    sample_sc = skm.silhouette_samples(data, Z)
    im = cplt.plot_xy(x_arr=[np.arange(len(Z))],
                      y_arr=[sample_sc],
                      ls_arr=['None'],
                      mk_arr=['.'],
                      ylabel="Shilhouette Score",
                      ylim=(0., .5)
                      )
    if sc_th is not None:
        cplt.plot_hline(
            y=sc_th, ls='--', color='k',
            ax=im['ax'])
    return im


def remove_outlayers(data, sc_th, Z):
    """
    Remove outlayers from the data based on the silhouette coefficient threshold.

    Parameters:
    data (numpy.ndarray): The input data.
    sc_th (float): The silhouette coefficient threshold.
    Z (numpy.ndarray): The cluster labels.

    Returns:
    numpy.ndarray: The indices of the data points that are not considered as outlayers.
    """
    gut.myprint('Remove Outlayers...')
    sample_sc = skm.silhouette_samples(data, Z)
    sign_Z = np.where(sample_sc >= sc_th)[0]
    gut.myprint(
        f'Removed {1 - np.count_nonzero(sign_Z)/len(Z)} of all inputs!')

    return sign_Z


def k_means_clustering(data,
                       verbose=True,
                       **kmeans_kwargs):

    k_method = kmeans_kwargs.pop('k_method', 'silhouette')
    max_iter = kmeans_kwargs.pop('max_iter', 1000)
    n_init = kmeans_kwargs.pop('n_init', 100)
    plot_stats = kmeans_kwargs.pop('plot_statistics', False)
    rem_outlayers = kmeans_kwargs.pop('rm_ol', False)
    sc_th = kmeans_kwargs.pop('sc_th', 0.05)
    minibatch = kmeans_kwargs.pop('minibatch', True)
    k = kmeans_kwargs.pop('n_clusters', None)
    if k is not None:
        if minibatch:
            gut.myprint(
                f'Using MiniBatchKMeans with {k} clusters', verbose=verbose)
            kmeans = MiniBatchKMeans(n_clusters=k,
                                     max_iter=max_iter,
                                     n_init=n_init,
                                     **kmeans_kwargs)
        else:
            kmeans = KMeans(n_clusters=k,
                            init="k-means++",
                            max_iter=max_iter,
                            n_init=n_init,
                            **kmeans_kwargs)
        kmeans.fit(data)
        score = skm.silhouette_score(data, kmeans.labels_)
        gut.myprint(f'Silhouette Score: {score}', verbose=verbose)

    else:
        gut.myprint(f'Select number of clusters using the {k_method} method')
        sse = []
        sscore = []
        krange = np.arange(2, 11)
        for ki in tqdm(krange):
            kmeans = KMeans(n_clusters=ki,
                            # init="k-means++",
                            max_iter=max_iter,
                            **kmeans_kwargs)
            kmeans.fit(data)
            score = skm.silhouette_score(data, kmeans.labels_)
            sse.append(kmeans.inertia_)
            sscore.append(score)
        if plot_stats:
            cplt.plot_xy(x_arr=[krange], y_arr=[sscore],
                         xlabel="Number of Clusters",
                         ylabel="Shilhouette Score",
                         )

        if k_method == 'elbow':
            kl = KneeLocator(
                krange, sse, curve="convex", direction="decreasing"
            )
            k = int(kl.elbow)
        if k_method == 'silhouette':
            k = int(krange[np.argmax(sscore)])
            print(sscore, np.argmax(sscore), k)
        else:
            k_method = 'None'
            k = 2
        gut.myprint(f'Method:{k_method}: Get {k} number of clusters!')

        kmeans = KMeans(n_clusters=k,
                        init="k-means++",
                        max_iter=max_iter,
                        **kmeans_kwargs)
        kmeans.fit(data)

    Z = kmeans.predict(data)

    if plot_stats:
        im = plot_statistics(data, sc_th, Z)
    else:
        im = None

    if rem_outlayers or sc_th != 0.05:
        rem_outlayers = True
        sign_Z = remove_outlayers(data, sc_th, Z)

    return {'cluster': Z,
            'significance': sign_Z if rem_outlayers else None,
            'fit': skm,
            'im': im,
            'model': kmeans,
            }


def gm_clustering(data,
                  **kwargs):

    max_iter = kwargs.pop('max_iter', 1000)
    n_init = kwargs.pop('n_init', 10)
    k = kwargs.pop('n_clusters', 2)
    plot_stats = kwargs.pop('plot_statistics', False)
    sc_th = kwargs.pop('sc_th', 0.05)
    rm_outlayers = kwargs.pop('rm_ol', False)

    gm = GaussianMixture(n_components=k,
                         #  init_params="k-means++",
                         max_iter=max_iter,
                         n_init=n_init,
                         covariance_type='full',
                         #  **kwargs
                         ).fit(data)

    Z = gm.predict(data)

    if plot_stats:
        plot_statistics(data, sc_th, Z)
    if rm_outlayers or sc_th != 0.05:
        rm_outlayers = True
        sign_Z = remove_outlayers(data, sc_th, Z)

    return {'cluster': Z,
            'significance': sign_Z if rm_outlayers else None,
            'fit': gm,
            'model': gm}


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
    compute_full_tree = kwargs.pop('compute_full_tree', 'auto')
    if n_clusters is None:
        compute_full_tree = True
    plot_stats = kwargs.pop('plot_statistics', False)
    rem_outlayers = kwargs.pop('rm_ol', False)
    sc_th = kwargs.pop('sc_th', 0.05)
    # The metric to use when calculating distance between instances in a feature array
    metric = kwargs.pop('metric', 'l2')
    if 'metric' == 'chebyshev':
        metric = sp.spatial.distance.chebyshev

    distance_threshold = sc_th if n_clusters is None else None

    linkage = kwargs.pop('linkage', 'complete')

    # Perform DBSCAN clustering from vector array or distance matrix.
    clustering = AgglomerativeClustering(n_clusters=n_clusters,
                                         linkage=linkage,
                                         affinity=metric,
                                         distance_threshold=distance_threshold,
                                         compute_full_tree=compute_full_tree).fit(data)

    if plot_stats:
        # plot the top three levels of the dendrogram
        None

    # Cluster labels for each point in the dataset given to fit(). Noisy samples are given the label -1.
    Z = clustering.labels_

    return {'cluster': Z,
            'significance': None,
            'fit': skm,
            'model': clustering}


def birch_clustering(data,
                     **kwargs):

    threshold = kwargs.pop('threshold', 0.01)
    k = kwargs.pop('n_clusters', 2)
    plot_stats = kwargs.pop('plot_statistics', False)
    rm_ol = kwargs.pop('rm_ol', False)
    sc_th = kwargs.pop('sc_th', 0.05)
    gut.myprint(f'Get {k} clusters!')

    br = Birch(threshold=threshold,
               n_clusters=k).fit(data)

    Z = br.predict(data)
    if plot_stats:
        plot_statistics(data=data, sc_th=sc_th, Z=Z)
    if rm_ol:
        sign_Z = remove_outlayers(data, sc_th, Z)
    return {'cluster': Z,
            'significance': sign_Z if rm_ol else None,
            'fit': skm,
            'model': br}


def affinity_clustering(data,
                        **kwargs):

    damping = kwargs.pop('damping', 0.99)

    br = AffinityPropagation(damping=damping).fit(data)

    Z = br.predict(data)
    return {'cluster': Z,
            'significance': None,
            'fit': skm,
            'model': br}


def spectral_clustering(data,
                        **kwargs):

    k = kwargs.pop('n_clusters', 2)

    gut.myprint(f'Get {k} clusters!')

    Z = SpectralClustering(n_clusters=k).fit_predict(data)

    return {'cluster': Z,
            'significance': None,
            'fit': skm}


def mean_shift_clustering(data,
                          **kwargs):

    bandwidth = kwargs.pop('bandwidth', None)

    Z = MeanShift(bandwidth=bandwidth).fit_predict(data)

    return {'cluster': Z,
            'significance': None,
            'fit': skm}


def hdbscan_clustering(data, **kwargs):
    plot_statistics = kwargs.pop('plot_statistics', False)
    min_samples = kwargs.pop('min_samples', 5)
    this_scan = hdbscan.HDBSCAN(min_samples=min_samples)
    labels = np.unique(this_scan.fit_predict(data))
    if plot_statistics:
        hdbscan.condensed_tree_.plot(select_clusters=False,)

    return labels, None


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
    print(clustering.labels_)
    Z = clustering.labels_

    return {'cluster': Z,
            'significance': None,
            'fit': skm,
            'model': clustering}


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

    min_samples = kwargs.pop('min_samples', 50)

    # Perform DBSCAN clustering from vector array or distance matrix.
    clustering = OPTICS(eps=eps, min_samples=min_samples,
                        metric=metric).fit(data)

    # Cluster labels for each point in the dataset given to fit(). Noisy samples are given the label -1.

    Z = clustering.labels_

    return {'cluster': Z,
            'significance': None,
            'fit': clustering,
            'model': clustering}


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

    cluster_names = kwargs.pop('cluster_names', None)
    data_input = create_2d_clustering_data(data_arr, tps, gf)

    grp_tps_dict = apply_cluster_data(data=data_input,
                                      objects=tps,
                                      method=method,
                                      cluster_names=cluster_names,
                                      return_model=False,
                                      **kwargs)

    grp_tps_dict['keys'] = list(grp_tps_dict.keys())
    # grp_tps_dict['Z'] = cluster_dict['cluster']
    # grp_tps_dict['im'] = cluster_dict['im']

    return grp_tps_dict


def create_2d_clustering_data(data_arr, tps, gf=(0, 0)):
    coll_data = []
    for data in data_arr:
        if isinstance(data, xr.DataArray):
            if len(tps) != len(data.time):
                tps, data = tu.equalize_time_points(tps, data)
            dims = gut.get_dims(data)
            if len(dims) == 3 and 'time' in dims:
                new_dims = gut.move_item_to_first(dims, 'time')
                data = sput.transpose_3D_data(
                    data, dims=new_dims)
            elif gut.compare_lists(list(data.dims), ['time', 'ids']):
                data = sput.transpose_2D_data(data, dims=['time', 'ids'])
            elif 'time' not in dims:
                gut.myprint(f'WARNING! No time dimension in data!')
        if len(data.shape) > 3:
            gut.myprint(
                f'Data shape {data.shape} larger dimension than (time, x, y)!')

        if len(tps) != data.shape[0]:
            raise ValueError(
                f'Number of tps {len(tps)} and time points {data.shape[0]} in data are not equal!'
            )

        if isinstance(data, xr.DataArray):
            data = data.data

        if len(data.shape) == 3:
            if gf[0] != 0 or gf[1] != 0:
                gut.myprint(f'Apply Gaussian Filter with sigma = {gf}!')
                sigma = [gf[1], gf[0]]  # sigma_y, sigma_x
                for idx, dp in enumerate(data):
                    data[idx] = sp.ndimage.filters.gaussian_filter(
                        dp, sigma, mode='constant')

        # Reshapedata to 2d array
        new_arr = data.reshape(*data.shape[:1], -1)
        coll_data.append(new_arr)

    # concatenate along 1 dimension
    data_input = np.concatenate(coll_data, axis=1)
    gut.myprint(f'Shape of input data_input: {data_input.shape}')
    return data_input


def get_cluster_dict(Z, cluster_x, cluster_names=None, verbose=True):
    grp_ids = gut.sort_by_frequency(arr=Z)

    grp_tps_dict = dict()
    for idx, gid in enumerate(grp_ids):
        idx_grp = np.where(Z == gid)[0]
        if cluster_names is not None:
            if len(cluster_names) != len(grp_ids):
                raise ValueError(f'Not same number of key names as groups!')
            keyname = cluster_names[idx]
            gut.myprint(f'Cluster {keyname} : {len(idx_grp)} objects',
                        verbose=verbose)
            grp_tps_dict[keyname] = cluster_x[idx_grp]
        else:
            gut.myprint(f'Cluster {idx} : {len(idx_grp)} objects',
                        verbose=verbose)
            grp_tps_dict[idx] = cluster_x[idx_grp]
    return grp_tps_dict


def apply_cluster_data(data,
                       objects=None,
                       method='kmeans',
                       cluster_names=None,
                       standardize=True,
                       return_model=False,
                       verbose=True,
                       **kwargs):
    """
    Apply clustering algorithm to the input data.

    Parameters:
        data (array-like): The input data to be clustered.
        objects (array-like, optional): The objects associated with the data. Default is None.
        method (str, optional): The clustering method to be used. Default is 'kmeans'.
        cluster_names (array-like, optional): The names of the clusters. Default is None.
        standardize (bool, optional): Whether to standardize the data before clustering. Default is True.
        return_model (bool, optional): Whether to return the clustering model. Default is False.
        **kwargs: Additional keyword arguments specific to the chosen clustering method.

    Returns:
        dict or tuple: A dictionary containing the clustered data and associated information.
                      If return_model is True, a tuple is returned with the dictionary and the clustering model.

    Raises:
        ValueError: If the input data is not in the correct 2D format.

    """
    if isinstance(data, list):
        data = np.array(data)

    if len(data.shape) != 2:
        raise ValueError(
            f'Data not in correct input 2D-format. Shape is {data.shape}!')

    if standardize:
        gut.myprint(f'Standardize data!', verbose=verbose)
        data = sut.standardize(dataset=data, axis=0)
        if gut.count_nans(data) != 0:
            gut.myprint(
                f'Data contains Nans: {gut.count_nans(data)}!', verbose=verbose)
    rm_ol = kwargs.pop('rm_ol', False)
    gut.myprint(f'Shape of input data_input: {data.shape}', verbose=verbose)
    if method == 'kmeans':
        cluster_dict = k_means_clustering(data=data,
                                          rm_ol=rm_ol,
                                          verbose=verbose,
                                          **kwargs)
    elif method == 'gm':
        cluster_dict = gm_clustering(data=data,
                                     rm_ol=rm_ol, **kwargs)
    elif method == 'dbscan':
        cluster_dict = dbscan_clustering(data=data, **kwargs)
    elif method == 'optics':
        cluster_dict = optics_clustering(data=data, **kwargs)
    elif method == 'agglomerative':
        cluster_dict = agglomerative_clustering(data=data, **kwargs)
    elif method == 'birch':
        cluster_dict = birch_clustering(data=data,
                                        rm_ol=rm_ol, **kwargs)
    elif method == 'spectral':
        cluster_dict = spectral_clustering(data=data, **kwargs)
    elif method == 'affinity':
        cluster_dict = affinity_clustering(data=data, **kwargs)
    elif method == 'mean_shift':
        cluster_dict = mean_shift_clustering(data=data, **kwargs)
    else:
        raise ValueError(f'Method {method} not implemented yet!')

    if objects is None:
        objects = np.arange(len(data))

    Z = cluster_dict['cluster']
    if rm_ol:
        sign_Z = cluster_dict['significance']
        if sign_Z is not None:
            Z = Z[sign_Z]
            objects = objects[sign_Z]

    grp_cluster_dict = get_cluster_dict(Z=Z, cluster_x=objects,
                                        cluster_names=cluster_names,
                                        verbose=verbose)

    if return_model:
        return cluster_dict['model']
    else:
        return grp_cluster_dict


def apply_bic(z_events, num_classes=10, n_runs=10):
    n_classes = np.arange(1, num_classes, 1)
    n_runs = 10
    result = []
    method = 'gm'
    for k in tqdm(n_classes):
        for r in range(n_runs):
            model = apply_cluster_data(data=z_events.data,
                                       n_clusters=k,
                                       method=method,
                                       return_model=True,
                                       verbose=False,
                                       standardize=False,
                                       )
            result.append(
                {'k': k, 'bic': model.bic(z_events.data), 'gmm': model}
            )
    result = pd.DataFrame(result)
    return result
