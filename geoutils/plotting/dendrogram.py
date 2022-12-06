import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, to_tree


def get_link_colors(self, Z, group_levels, cut_level, color_branch=True):
    """
    Logic:
    * rows in Z correspond to "inverted U" links that connect clusters
    * rows are ordered by increasing distance
    * if the colors of the connected clusters match, use that color for link

    """
    from matplotlib.colors import rgb2hex
    if cut_level < 1:
        raise ValueError(f"ERROR! Cut Level (={cut_level}) has to be >=1!")
    level = cut_level-1
    num_clusters = len(self.count_elements(group_levels[level]))
    # Attention check again if this is true!
    ground_level_ids = group_levels[0]

    cmap, norm = self.discrete_cmap(0, num_clusters)
    colors = np.array([rgb2hex(cmap(norm(rgb)))
                       for rgb in range(num_clusters)])

    color_ground_cluster = colors[ground_level_ids]
    # Color mapping

    link_cols = {}
    for i, link in enumerate(Z[:].astype(int)):
        if not color_branch:
            link_cols[i+1+len(Z)] = self.dflt_col
        else:
            idx_0, idx_1, cluster_level, num_groups = link
            c1, c2 = (link_cols[x] if x > len(Z) else color_ground_cluster[x]
                      for x in (idx_0, idx_1))
            if cluster_level > cut_level:
                link_cols[i+1+len(Z)] = self.dflt_col
            else:
                if c1 == c2:
                    link_cols[i+1+len(Z)] = c1

    return link_cols, colors, num_clusters


def get_id_to_coord(self, Z, ddata, ax=None):
    def flatten(lt):
        return [item for sublist in lt for item in sublist]
    X = flatten(ddata['icoord'])
    Y = flatten(ddata['dcoord'])
    # get leave coordinates, which are at y == 0
    leave_coords = [(x, y) for x, y in zip(X, Y) if y == 0]

    # in the dendogram data structure,
    # leave ids are listed in ascending order according to their x-coordinate
    order = np.argsort([x for x, y in leave_coords])
    # <- main data structure
    id_to_coord = dict(
        zip(ddata['leaves'], [leave_coords[idx] for idx in order]))

    # ----------------------------------------
    # get coordinates of other nodes

    # map endpoint of each link to coordinates of parent node
    children_to_parent_coords = dict()
    for i, d in zip(ddata['icoord'], ddata['dcoord']):
        x = (i[1] + i[2]) / 2
        y = d[1]  # or d[2]
        parent_coord = (x, y)
        left_coord = (i[0], d[0])
        right_coord = (i[-1], d[-1])
        children_to_parent_coords[(left_coord, right_coord)] = parent_coord
    # traverse tree from leaves upwards and populate mapping ID -> (x,y)
    root_node, node_list = to_tree(Z, rd=True)
    ids_left = range(len(ddata['leaves']), len(node_list))

    while len(ids_left) > 0:

        for ii, node_id in enumerate(ids_left):
            node = node_list[node_id]
            if not node.is_leaf():
                if (node.left.id in id_to_coord) and (node.right.id in id_to_coord):
                    left_coord = id_to_coord[node.left.id]
                    right_coord = id_to_coord[node.right.id]
                    id_to_coord[node_id] = children_to_parent_coords[(
                        left_coord, right_coord)]

        ids_left = [node_id for node_id in range(
            len(node_list)) if node_id not in id_to_coord]

    # plot result on top of dendrogram
    if ax is not None:
        for node_id, (x, y) in id_to_coord.items():
            if not node_list[node_id].is_leaf():
                if node_id > 16700:
                    ax.plot(x, y, 'ro')
                    ax.annotate(str(node_id), (x, y), xytext=(0, -8),
                                textcoords='offset points',
                                va='top', ha='center')
    return id_to_coord


def fancy_dendrogram(*args, **kwargs):
    """
    Inspired from
    https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/

    Returns
    -------
    Dendogramm data as scipy dataset.

    """
    from matplotlib import ticker

    ax = kwargs['ax']

    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)
    lw = kwargs.pop('lw', None)
    if lw is None:
        lw = 1
    with plt.rc_context({'lines.linewidth': lw}):
        ddata = dendrogram(*args, **kwargs)
    if not kwargs.get('no_plot', False):

        ax.set_xlabel('sample index (cluster size) ')
        ax.set_ylabel('Cluster Level')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                ax.plot(x, y, 'o', c=c)
                ax.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                            textcoords='offset points',
                            va='top', ha='center')
        if max_d:
            ax.axhline(y=max_d, c='k', lw=4, ls='--')
    for axis in [ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))

    return ddata


def plot_path(self, ddata, node_path, ax, color, label=None):
    lw = 2
    node_id_to_coord = ddata['node_id_to_coord']
    xcoord = None
    ycoord = None
    for i, node_id in enumerate(node_path):
        this_node_coord = node_id_to_coord[node_id]
        if i < len(node_path)-1:
            next_id = node_path[i+1]
            next_node_coord = node_id_to_coord[next_id]
            xcoord = [this_node_coord[0],
                      this_node_coord[0], next_node_coord[0]]
            ycoord = [this_node_coord[1],
                      next_node_coord[1], next_node_coord[1]]
            ax.plot(xcoord, ycoord, lw=lw, ls='dashed',
                    color=color, label=None)
    if xcoord is not None:
        ax.plot(xcoord, ycoord, lw=lw, ls='dashed',
                color=color, label=label)


def plot_dendrogram(self, Z=None, group_levels=None, cut_level=None, title=None,
                    node_ids=None, fig=None, ax=None, colors=None, labels=None,
                    color_branch=True, plot_Z=True, savepath=None,):

    if Z is None:
        Z = self.Z

    if group_levels is None:
        group_levels = self.group_levels
    node_levels, _ = self.node_level_arr(group_levels)
    this_d_tree = Dendro_tree(Z, node_levels)

    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 8))
    else:
        savepath = None

    if cut_level is not None and plot_Z:
        link_colors, cluster_colors, num_clusters = self.get_link_colors(Z,
                                                                         group_levels,
                                                                         cut_level,
                                                                         color_branch=color_branch
                                                                         )

        ddata = self.fancy_dendrogram(Z,
                                      ax=ax,
                                      leaf_rotation=90.,  # rotates the x axis labels
                                      annotate_above=10,
                                      max_d=cut_level+0.5,
                                      link_color_func=lambda x: link_colors[x],
                                      lw=1,
                                      no_plot=False,
                                      )
    else:
        num_clusters = len(self.count_elements(group_levels[0]))
        ddata = dendrogram(Z)

        if not plot_Z:
            # use [:] to get a copy, since we're adding to the same list
            for c in ax.collections[:]:
                # Remove the original LineCollection
                ax.collections.remove(c)

    if node_ids is not None:

        # if ax is not None, id numbers are plotted
        id_to_coord = self.get_id_to_coord(Z, ddata, ax=None)

        ddata['node_id_to_coord'] = id_to_coord
        for i, node_id in enumerate(node_ids):
            if colors is not None:
                color = colors[i]
            else:
                color = 'red'
            if labels is not None:
                label = labels[i]
            else:
                label = None
            node_path = this_d_tree.foreward_path(node_id)
            node_id_path = [this_node.id for this_node in node_path]

            self.plot_path(ddata, node_id_path, ax, color, label)

        if labels is not None:
            ax.legend(bbox_to_anchor=(1, 1), loc='upper left',
                      fancybox=True, shadow=True, ncol=1)

    y_title = 1.05
    if title is None:
        ax.set_title(
            f'Hierarchical Clustering Dendrogram with {num_clusters} groups', y=y_title)
    else:
        ax.set_title(title, y=y_title)

    if savepath is not None:
        fig.tight_layout()

        plt.savefig(savepath)

    return ddata
