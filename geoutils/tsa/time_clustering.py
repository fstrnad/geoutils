from latgmm.utils import eof
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
import geoutils.utils.spatial_utils as sput
import scipy as sp
import geoutils.utils.statistic_utils as sut
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, OPTICS, Birch
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
import scipy.cluster.hierarchy as shr
from scipy.spatial.distance import pdist
import numpy as np
import xarray as xr
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
    sample_sc = get_silhouette_scores(data, Z)
    im = cplt.plot_xy(x_arr=[np.arange(len(Z))],
                      y_arr=[sample_sc],
                      ls_arr=['None'],
                      mk_arr=['.'],
                      ylabel="Shilhouette Score",
                      ylim=(0., 1)
                      )
    if sc_th is not None:
        cplt.plot_hline(
            y=sc_th, ls='--', color='k',
            ax=im['ax'])
    return im


def plot_kNN(data, n_neihbors=5):
    """Plot the k-Nearest Neighbors graph of the input data.
    """
    nbrs = NearestNeighbors(n_neighbors=n_neihbors).fit(data)
    # Find the k-neighbors of a point
    neigh_dist, neigh_ind = nbrs.kneighbors(data)
    # sort the neighbor distances (lengths to points) in ascending order
    # axis = 0 represents sort along first axis i.e. sort along row
    sort_neigh_dist = np.sort(neigh_dist, axis=0)
    k_dist = sort_neigh_dist[:, n_neihbors-1]
    im = cplt.plot_xy(x_arr=[np.arange(len(data))],
                      y_arr=[k_dist],
                      ls_arr=['None'],
                      mk_arr=['.'],
                      ylabel="k-NN Distance",
                      xlabel=f'Sorted observations ({n_neihbors-1} neighbors)',
                      )
    from kneed import KneeLocator
    kneedle = KneeLocator(x=range(1, len(neigh_dist)+1), y=k_dist, S=1.0,
                          curve="concave", direction="increasing", online=True)

    # get the estimate of knee point
    gut.myprint(f'Knee: {kneedle.knee_y}')

    return float(kneedle.knee_y)


def get_silhouette_scores(data, Z):
    """Gets the silhouette scores for the input data of all samples
    """
    cluster_groups = np.unique(Z)
    if len(cluster_groups) > 1 and len(cluster_groups) < len(Z):
        return skm.silhouette_samples(data, Z)
    else:
        if len(cluster_groups) == len(Z):
            gut.myprint('All samples are their own cluster!', verbose=True)
        else:
            gut.myprint('Only one cluster found!', verbose=True)
        return np.zeros(len(Z))


def get_silhouette_score(data, Z):
    """
    Returns the silhouette score for the input data as a float.

    Parameters:
    data (array-like): The input data.
    Z (array-like): The cluster labels.

    Returns:
    float: The silhouette score.
    """
    cluster_groups = np.unique(Z)
    if len(cluster_groups) > 1 and len(cluster_groups) < len(Z):
        sscore = float(skm.silhouette_score(data, Z))
    else:
        if len(cluster_groups) == len(Z):
            gut.myprint('All samples are their own cluster!', verbose=True)
        else:
            gut.myprint('Only one cluster found!', verbose=True)
        sscore = 0.
    return np.around(sscore, 3)


def remove_noise(Z):
    """
    Remove noise from the data based on the cluster labels.

    Parameters:
    data (numpy.ndarray): The input data.
    Z (numpy.ndarray): The cluster labels.

    Returns:
    numpy.ndarray: The indices of the data points that are not considered as noise.
    """
    gut.myprint('Remove Noise...')
    sign_Z = np.where(Z != -1)[0]
    if len(sign_Z) == 0:
        gut.myprint('Noisy samples are all samples!')
    if len(sign_Z) != len(Z):
        gut.myprint(
            f'Removed noisy {1 - len(sign_Z)/len(Z)} of all inputs!')

    return sign_Z


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
    sample_sc = get_silhouette_scores(data, Z)
    sign_Z = np.where(sample_sc >= sc_th)[0]
    gut.myprint(
        f'Removed {1 - len(sign_Z)/len(Z)} of all inputs!')

    return sign_Z


def k_means_clustering(data,
                       verbose=True,
                       **kmeans_kwargs):

    k_method = kmeans_kwargs.pop('k_method', 'silhouette')
    max_iter = kmeans_kwargs.pop('max_iter', 1000)
    n_init = kmeans_kwargs.pop('n_init', 100)
    plot_stats = kmeans_kwargs.pop('plot_statistics', False)
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

    return {'cluster': Z,
            'fit': skm,
            'model': kmeans,
            }


def gm_clustering(data,
                  verbose=True,
                  **kwargs):

    max_iter = kwargs.pop('max_iter', 1000)
    n_init = kwargs.pop('n_init', 10)
    k = kwargs.pop('n_clusters', 2)

    gm = GaussianMixture(n_components=k,
                         #  init_params="k-means++",
                         max_iter=max_iter,
                         n_init=n_init,
                         covariance_type='full',
                         #  **kwargs
                         ).fit(data)

    Z = gm.predict(data)

    return {'cluster': Z,
            'fit': gm,
            'model': gm}


def agglomerative_clustering(data, **kwargs):
    """
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
    sc_th = kwargs.pop('sc_th', 0.05)
    distance_threshold = sc_th if n_clusters is None else None

    # The metric to use when calculating distance between instances in a feature array
    metric = kwargs.pop('metric', 'l2')
    if 'metric' == 'chebyshev':
        metric = sp.spatial.distance.chebyshev

    linkage = kwargs.pop('linkage', 'complete')

    clustering = AgglomerativeClustering(n_clusters=n_clusters,
                                         linkage=linkage,
                                         affinity=metric,
                                         distance_threshold=distance_threshold,
                                         compute_full_tree=compute_full_tree).fit(data)

    # Cluster labels for each point in the dataset given to fit(). Noisy samples are given the label -1.
    Z = clustering.labels_

    return {'cluster': Z,
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
    return {'cluster': Z,
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
    import hdbscan

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
    # The radius of neighborhood samples
    eps = kwargs.pop('eps', 2)
    # The metric to use when calculating distance between instances in a feature array
    metric = kwargs.pop('metric', 'euclidean')
    min_samples = kwargs.pop('min_samples', 5)
    plot_statistics = kwargs.pop('plot_statistics', False)
    if plot_statistics:
        eps = plot_kNN(data, n_neihbors=min_samples)

    # Perform DBSCAN clustering from vector array or distance matrix.
    clustering = DBSCAN(eps=eps,
                        min_samples=min_samples,
                        metric=metric
                        ).fit(data)

    # Cluster labels for each point in the dataset given to fit(). Noisy samples are given the label -1.
    Z = clustering.labels_

    return {'cluster': Z,
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
            gut.myprint(f'Cluster {keyname} : {len(idx_grp)} samples',
                        verbose=verbose)
            grp_tps_dict[keyname] = cluster_x[idx_grp]
        else:
            gut.myprint(f'Cluster {idx} : {len(idx_grp)} samples',
                        verbose=verbose)
            grp_tps_dict[idx] = cluster_x[idx_grp]
    return grp_tps_dict


def apply_cluster_data(data,
                       objects=None,
                       method='kmeans',
                       cluster_names=None,
                       standardize=True,
                       rm_ol=False,
                       return_model=False,
                       verbose=True,
                       score='silhouette',
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
    plot_stats = kwargs.pop('plot_statistics', False)
    sc_th = kwargs.pop('sc_th', 0.05)
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
                f'Data contains Nans: {gut.count_nans(data)}!',
                verbose=verbose)
    gut.myprint(f'Shape of input data_input: {data.shape}',
                verbose=verbose)
    gut.myprint(f'Cluster based on {method}!', verbose=verbose)
    if method == 'kmeans':
        cluster_dict = k_means_clustering(data=data,
                                          verbose=verbose,
                                          **kwargs)
    elif method == 'gm' or method == 'gmm':
        cluster_dict = gm_clustering(data=data,
                                     verbose=verbose,
                                     **kwargs)
    elif method == 'dbscan':
        cluster_dict = dbscan_clustering(data=data,
                                         verbose=verbose,
                                         plot_statistics=plot_stats,
                                         **kwargs)
    elif method == 'optics':
        cluster_dict = optics_clustering(data=data, **kwargs)
    elif method == 'agglomerative':
        cluster_dict = agglomerative_clustering(data=data, **kwargs)
    elif method == 'birch':
        cluster_dict = birch_clustering(data=data,
                                        **kwargs)
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
    if len(objects) != len(data):
        raise ValueError(
            f'Number of objects {len(objects)} and data {len(data)} are not equal!')

    Z = cluster_dict['cluster']
    if score == 'silhouette':
        score = get_silhouette_score(data, Z)
        gut.myprint(f'Silhouette Score: {score}',
                    verbose=verbose, bold=True, color='green')
        sscores = get_silhouette_scores(data, Z)

    if plot_stats:
        plot_statistics(data, sc_th, Z)

    sign_Z = remove_noise(Z)
    if rm_ol or sc_th != 0.05:
        rm_ol = True
        sign_Z_rm = remove_outlayers(data, sc_th, Z)
        sign_Z = np.intersect1d(sign_Z, sign_Z_rm)

    if sign_Z is not None:
        Z = Z[sign_Z]
        objects = objects[sign_Z]
    grp_cluster_dict = get_cluster_dict(Z=Z, cluster_x=objects,
                                        cluster_names=cluster_names,
                                        verbose=verbose)

    grp_cluster_dict['keys'] = list(grp_cluster_dict.keys())
    grp_cluster_dict['Z'] = cluster_dict['cluster']
    grp_cluster_dict['sscores'] = sscores if score == 'silhouette' else None

    if return_model:
        return cluster_dict['model']
    else:
        grp_cluster_dict['model'] = cluster_dict['model']
        grp_cluster_dict['input'] = data
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


def apply_sscore(z_events, num_classes=10, n_runs=10, method='kmeans',
                 score='silhouette'):
    if num_classes < 2:
        raise ValueError('Number of classes must be at least 2!')
    if method != 'gm' and score == 'bic':
        raise ValueError(
            'BIC score is only available for Gaussian Mixture Models!')

    if score == 'bic':
        nclasses_start = 1
    else:
        nclasses_start = 2
    n_classes = np.arange(nclasses_start, num_classes+nclasses_start, 1)
    result = {}
    for k in tqdm(n_classes):
        this_scores = []
        for r in range(n_runs):
            model = apply_cluster_data(data=z_events.data,
                                       n_clusters=k,
                                       method=method,
                                       score=score,
                                       return_model=True,
                                       verbose=False,
                                       standardize=False,
                                       )
            Z = model.predict(z_events.data)
            if score == 'silhouette':
                this_scores.append(skm.silhouette_score(z_events.data, Z))
            elif score == 'bic':
                this_scores.append(model.bic(z_events.data))
            else:
                raise ValueError(f'Score {score} not implemented yet!')

        result[k] = this_scores
    return result


def grid_search(data_input, tps, num_classes=10, num_eofs=5, steps=15,
                ts=None, min_corr=0, max_num_eofs=None,
                n_runs=5, method='kmeans', score='silhouette'):

    if max_num_eofs is None:
        max_num_eofs = num_eofs
    else:
        if max_num_eofs < num_eofs:
            raise ValueError(
                'Maximum number of EOFs must be larger than number of EOFs!')
    n_eofs_min = 1
    grid = np.zeros((num_eofs, num_classes))
    sppca = eof.SpatioTemporalPCA(data_input, n_components=max_num_eofs)
    gut.myprint(
        f"Explained variance by {max_num_eofs} EOFs: {np.sum(sppca.explained_variance())}")
    for i, n_components in enumerate(np.arange(n_eofs_min, num_eofs+n_eofs_min, 1)):
        gut.myprint(f"Get {n_components} EOFs!")

        # Get latent encoding
        z_events = eof.spatio_temporal_latent_volume(sppca, data_input,
                                                     tps=tps,
                                                     ts=ts,
                                                     min_corr=min_corr,
                                                     num_eofs=n_components,
                                                     steps=steps)

        n_results = apply_sscore(z_events=z_events,
                                 num_classes=num_classes,
                                 n_runs=n_runs,
                                 method=method,
                                 score=score
                                 )
        for j, k in enumerate(n_results):
            grid[i, j] = np.mean(n_results[k])

    return grid
