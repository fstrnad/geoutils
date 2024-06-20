"""Equidistant points on a sphere.

Fibbonachi Spiral:
https://bduvenhage.me/geometry/2019/07/31/generating-equidistant-vectors.html

Fekete points:
https://arxiv.org/pdf/0808.1202.pdf

Geodesic grid: (sec. 3.2)
https://arxiv.org/pdf/1711.05618.pdf

Review on geodesic grids:
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.113.997&rep=rep1&type=pdf

"""
# %%
import math
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
from scipy.spatial.transform import Rotation as Rot
import climnet.grid.fekete as fk
import geoutils.plotting.plots as cplt
import geoutils.utils.general_utils as gut
import geoutils.utils.file_utils as fut
import geoutils.utils.spatial_utils as sput
from importlib import reload
RADIUS_EARTH = 6371  # radius of earth in km
PATH = os.path.dirname(os.path.abspath(__file__))


class BaseGrid():
    """Base Grid

    Parameters:
    -----------

    """

    def __init__(self):
        self.grid = None
        self.savefolder = f'{PATH}/grids/'
        return

    def create_grid(self):
        """Create grid."""
        gut.myprint("Function should be overwritten by subclasses!")
        return None

    def cut_grid(self, lon_range, lat_range):
        """Cut the grid in lat and lon range.

        TODO: allow taking regions around the date line

        Args:
        -----
        lat_range: list
            [min_lon, max_lon]
        lon_range: list
            [min_lon, max_lon]
        """
        if lon_range[0] > lon_range[1]:
            raise ValueError(
                "Ranges around the date line are not yet defined.")
        else:
            print(f"Cut grid in range lat: {lat_range} and lon: {lon_range}")

        idx = np.where((self.grid['lat'] >= lat_range[0])
                       & (self.grid['lat'] <= lat_range[1])
                       & (self.grid['lon'] >= lon_range[0])
                       & (self.grid['lon'] <= lon_range[1]))[0]
        cutted_grid = {'lat': self.grid['lat'][idx],
                       'lon': self.grid['lon'][idx]}

        return cutted_grid

    def save(self, filename):
        savepath = f'{self.savefolder}/{filename}'
        fut.save_np_dict(self.grid, savepath)
        gut.myprint('Done')

    def load(self, filename):
        savepath = f'{self.savefolder}/{filename}'
        self.grid = fut.load_np_dict(savepath)
        x, y, z = spherical2cartesian(lon=self.grid['lon'],
                                      lat=self.grid['lat']
                                      )
        self.X = arange_xyz(x, y, z)
        gut.myprint(f'Loaded grid {savepath}!')

    def get_distance_equator(self):
        """Return distance between points at the equator."""
        d_lon = sput.degree2distance_equator(self.grid_step_lon)
        return d_lon


class GaussianGrid(BaseGrid):
    """Gaussian Grid of the earth which is the classical grid type.

    Args:
    ----
    grid_step_lon: float
        Grid step in longitudinal direction in degree
    grid_step_lat: float
        Grid step in longitudinal direction in degree

    """

    def __init__(self, grid_step_lon, grid_step_lat, grid=None):
        super().__init__()
        self.grid_step_lon = grid_step_lon
        self.grid_step_lat = grid_step_lat
        self.grid = grid

        self.create_grid()

    def create_grid(self):
        init_lat = np.arange(-89.5, 90.5, self.grid_step_lat)
        init_lon = np.arange(-179.5, 180.5, self.grid_step_lon)

        lon_mesh, lat_mesh = np.meshgrid(init_lon, init_lat)

        self.grid = {'lat': lat_mesh.flatten(), 'lon': lon_mesh.flatten()}

        return self.grid


class Reduced_GaussianGrid(BaseGrid):
    """Gaussian Grid of the earth which is the classical grid type.

    Args:
    ----
    grid_step_lon: float
        Grid step in longitudinal direction in degree
    grid_step_lat: float
        Grid step in longitudinal direction in degree

    """

    def __init__(self, n_lats=None, grid_step=None, grid=None):
        super().__init__()
        if n_lats is None and grid_step is None:
            raise ValueError('Either n_lats or grid_step must be given.')
        if grid_step is not None and n_lats is not None:
            raise ValueError('Only one of n_lats or grid_step must be given.')
        if n_lats is None:
            gut.myprint('Grid step is the grid step in latitudinal direction!')
            n_lats = int(90 / grid_step) # 90 because of the n_lats is only from poles to equator

        self.n_lats = n_lats
        self.grid_step_lon = None
        self.grid_step_lat = grid_step
        self.grid = grid

        self.create_grid()

    def create_grid(self):
        n_lats = 2 * self.n_lats
        latitudes = np.linspace(-90, 90, n_lats)
        reduced_gaussian_grid = []
        for i in range(n_lats):
            n_lons = 4 * i + 16 if i < n_lats // 2 else 4 * \
                (n_lats - i - 1) + 16
            longitudes = np.linspace(-180, 180, n_lons)
            for j in range(n_lons):
                reduced_gaussian_grid.append((longitudes[j], latitudes[i]))
        reduced_gaussian_grid = np.array(
            reduced_gaussian_grid, dtype=np.float32)

        self.grid = {'lat': reduced_gaussian_grid[:, 1],
                     'lon': reduced_gaussian_grid[:, 0],
                     'grid': reduced_gaussian_grid}

        self.grid_step_lat = latitudes[1] - latitudes[0]
        self.grid_step_lon = longitudes[1] - longitudes[0]

        return self.grid


class FibonacciGrid(BaseGrid):
    """Fibonacci sphere creates a equidistance grid on a sphere.

    Parameters:
    -----------
    distance_between_points: float
        Distance between the equidistance grid points in km.
    grid: dict
        If grid is already computed, e.g. {'lon': [], 'lat': []}. Default: None
    """

    def __init__(self,
                 grid=None,
                 load_grid=None):
        super().__init__()
        self.num_points = self.get_num_points()
        self.grid = grid
        gut.myprint(f'Try to load grid {load_grid}...')
        if load_grid is None:
            self.create_grid()
        else:
            self.load(filename=load_grid)

    def create_grid(self,
                    ):
        """Create Fibonacci grid."""
        print(f'Create fibonacci grid with {self.num_points} points.')
        cartesian_grid = self.fibonacci_sphere(self.num_points)
        lon, lat = cartesian2spherical(cartesian_grid[:, 0],
                                       cartesian_grid[:, 1],
                                       cartesian_grid[:, 2])

        # lon, lat = cut_lon_lat(
        #     lon, lat, lon_range=lon_range, lat_range=lat_range)
        self.grid = {'lat': lat, 'lon': lon, 'grid': gut.zip_2_lists(lon, lat)}
        self.X = cartesian_grid
        return self.grid

    def fibonacci_sphere(self, num_points=1):
        """Creates the fibonacci sphere points on a unit sphere.
        Code inspired by:
        https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
        """
        points = []
        phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

        for i in range(num_points):
            y = 1 - (i / float(num_points - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = math.cos(theta) * radius
            z = math.sin(theta) * radius

            points.append([x, y, z])

        return np.array(points)

    def get_num_points(self):
        """Relationship between distance and num of points of fibonacci sphere.

        num_points = a*distance**k
        """
        # obtained by log-log fit
        k = -2.01155176
        a = np.exp(20.0165958)

        return int(a * self.distance**k)

    def get_distance_equator(self):
        """Return distance between points at the equator."""
        return self.distance


class FeketeGrid(BaseGrid):
    """Fibonacci sphere creates a equidistance grid on a sphere.

    Parameters:
    -----------
    distance_between_points: float
        Distance between the equidistance grid points in km.
    grid: dict (or 'old' makes old version of fib grid, 'maxmin' to maximize min. min-distance)

        If grid is already computed, e.g. {'lon': [], 'lat': []}. Default: None
    """

    def __init__(self, num_points, num_iter=1000,
                 grid=None,
                 pre_proccess_type=None,
                 load_grid=None):
        super().__init__()

        self.distance = get_distance_from_num_points(num_points)
        self.num_points = num_points
        self.num_iter = num_iter
        self.epsilon = None
        self.grid = grid
        self.reduced_grid = None

        gut.myprint(f'Try to load grid {load_grid}...')

        if load_grid is None or not fut.exist_file(f'{self.savefolder}/{load_grid}'):
            self.create_grid(num_points=self.num_points,
                             num_iter=self.num_iter,
                             pre_proccess_type=pre_proccess_type)
        else:
            self.load(filename=load_grid)

    def create_grid(self, num_points=None,
                    num_iter=1000,
                    pre_proccess_type=None):

        if num_points is None:
            num_points = self.num_points
        print(
            f'\nCreate Fekete grid with {num_points} points with {num_iter} iterations.')
        # This runs the Fekete Algorithm
        X_pre = None
        if pre_proccess_type is not None:
            X_pre = self.get_preprocessed_grid(grid_type=pre_proccess_type)
        self.X, self.dq = fk.bendito(N=num_points,
                                     maxiter=num_iter,
                                     X=X_pre)
        print('... Finished', flush=True)
        lon, lat = cartesian2spherical(x=self.X[:, 0],
                                       y=self.X[:, 1],
                                       z=self.X[:, 2])
        self.grid = {'lon': lon, 'lat': lat}

        return self.grid

    def get_preprocessed_grid(self, grid_type='fibonacci'):
        print(f'Start preprocessed grid {grid_type}...',
              flush=True)
        if grid_type == 'gaussian':
            Grid = GaussianGrid(self.grid_step, self.grid_step)
        elif grid_type == 'fibonacci':
            Grid = FibonacciGrid(self.distance)
        else:
            raise ValueError(f'Grid type {grid_type} does not exist.')

        x, y, z = spherical2cartesian(lon=Grid.grid['lon'],
                                      lat=Grid.grid['lat'])

        grid_points = np.array([x, y, z]).T

        return grid_points

    def nudge_grid(self, n_iter=1, step=0.01):  # a 100th of a grid_step
        if self.reduced_grid is None:
            raise KeyError('First call keep_original_points')
        leng = len(self.grid['lon'])
        delta = 2 * np.pi * step * self.distance / 6371
        regx, regy, regz = spherical2cartesian(
            self.reduced_grid['lon'], self.reduced_grid['lat'])
        for iter in range(n_iter):
            perm = np.random.permutation(leng)
            for i in range(leng):
                i = perm[i]
                x, y, z = spherical2cartesian(
                    self.grid['lon'], self.grid['lat'])
                r = np.array([x[i], y[i], z[i]])
                vec2 = np.array([regx[i] - x[i], regy[i] - y[i], regz[i]-z[i]])
                vec2 = vec2 - np.dot(vec2, r) * r
                rot_axis = np.cross(r, vec2)
                rot = Rot.from_rotvec(
                    rot_axis * delta / np.linalg.norm(rot_axis))
                new_grid_cart = rot.as_matrix() @ np.array([x, y, z])
                new_lon, new_lat = cartesian2spherical(
                    new_grid_cart[0, :], new_grid_cart[1, :], new_grid_cart[2, :])
                self.grid = {'lon': new_lon, 'lat': new_lat}

    def get_distance_equator(self):
        """Return distance between points at the equator."""
        return self.distance

    def keep_original_points(self, orig_grid, regular=True):
        if self.grid is None:
            self.create_grid()
        if regular:
            new_lon, used_dists, delete, dists, possible_coords, new_coords = [], [], [], [], [], []
            lons = np.sort(np.unique(orig_grid['lon']))
            lats = np.sort(np.unique(orig_grid['lat']))
            for i in range(len(self.grid['lat'])):
                lo = self.grid['lon'][i]
                la = self.grid['lat'][i]
                pm_lon = np.array([(lons[j]-lo)*(lons[j+1]-lo)
                                   for j in range(len(lons)-1)])
                pm_lat = np.array([(lats[j]-la)*(lats[j+1]-la)
                                   for j in range(len(lats)-1)])
                if np.where(pm_lon < 0)[0].shape[0] == 0:
                    rel_lon = [lons[0], lons[-1]]
                else:
                    lon_idx = np.where(pm_lon < 0)[0][0]
                    rel_lon = [lons[lon_idx], lons[lon_idx+1]]
                if np.where(pm_lat < 0)[0].shape[0] == 0:
                    rel_lat = [lats[0], lats[-1]]
                else:
                    lat_idx = np.where(pm_lat < 0)[0][0]
                    rel_lat = [lats[lat_idx], lats[lat_idx+1]]
                these_dists = np.array(
                    [gdistance((l2, l1), (la, lo)) for l1 in rel_lon for l2 in rel_lat])
                these_coords = np.array([(l1, l2)
                                         for l1 in rel_lon for l2 in rel_lat])
                prio = np.argsort(these_dists)
                dists.append(these_dists[prio])
                possible_coords.append(these_coords[prio])

            # choose the nearest unused neighbor
            for idx in np.argsort(np.array(dists)[:, 0]):
                i = 0
                while i < 4 and np.any([np.all(possible_coords[idx][i, :] == coord) for coord in new_coords]):
                    if i == 3:
                        delete.append(idx)
                        warnings.warn(
                            f'No neighbors left for {self.grid["lon"][idx], self.grid["lat"][idx]}. Removing this point.')
                    i += 1
                if i < 4:
                    new_coords.append(possible_coords[idx][i, :])
                    used_dists.append(dists[idx][i])
                # ids = map(id, new_coords)
            dists2 = np.delete(dists, delete, 0)
            new_coords = np.array(new_coords)[np.argsort(
                np.argsort(np.array(dists2)[:, 0]))]  # inverse permutation
            self.reduced_grid = {'lon': np.array(
                new_coords)[:, 0], 'lat': np.array(new_coords)[:, 1]}
            self.grid['lon'] = np.delete(self.grid['lon'], delete, 0)
            self.grid['lat'] = np.delete(self.grid['lat'], delete, 0)
            return used_dists
        else:
            raise KeyError('Only regular grids!')

    def min_dists(self, grid2=None):
        if grid2 is None:
            lon1, lon2 = self.grid['lon'], self.grid['lon']
            lat1, lat2 = self.grid['lat'], self.grid['lat']
            d = 9999 * np.ones((len(lon1), len(lon2)))
            for i in range(len(lon1)):
                for j in range((len(lon1))):
                    if i < j:
                        d[i, j] = gdistance(
                            (lat1[i], lon1[i]), (lat2[j], lon2[j]))
                    elif i > j:
                        d[i, j] = d[j, i]
            return d.min(axis=1)
        else:  # min dist from self.grid point to other grid
            lon1, lon2 = self.grid['lon'], grid2['lon']
            lat1, lat2 = self.grid['lat'], grid2['lat']
            d = 9999 * np.ones((len(lon1), len(lon2)))
            for i in range(len(lon1)):
                for j in range(len(lon2)):
                    d[i, j] = gdistance((lat1[i], lon1[i]), (lat2[j], lon2[j]))
            return d.min(axis=1)


def cartesian2spherical(x, y, z):
    """Cartesian coordinates to lon and lat.

    Args:
    -----
    x: float or np.ndarray
    y: float or np.ndarray
    z: float or np.ndarray
    """
    lat = np.degrees(np.arcsin(z))
    lon = np.degrees(np.arctan2(y, x))

    return lon, lat


def spherical2cartesian(lon, lat, r=1):
    lon = lon * 2 * np.pi / 360
    lat = lat * np.pi / 180
    x = np.cos(lon) * np.cos(lat)
    y = np.sin(lon) * np.cos(lat)
    z = np.sin(lat)

    return x, y, z


def arange_xyz(x, y, z):
    X = np.c_[x, y, z]
    return X


def get_distance_from_num_points(num_points):
    k = 1/2.01155176    # fitted values
    a = np.exp(20.0165958)  # fitted values
    return (a/num_points)**k


def get_num_points(dist):
    """
    Relationship between distance and num of points of fibonacci sphere.

    num_points = a*distance**k
    """
    # obtained by log-log fit
    k = -2.01155176
    a = np.exp(20.0165958)
    return int(a * dist**k)


def cut_lon_lat(lon, lat, lon_range, lat_range):
    min_lon = np.min(lon_range)
    max_lon = np.max(lon_range)
    min_lat = np.min(lat_range)
    max_lat = np.max(lat_range)
    if min_lon > -180 or max_lon < 180 or min_lat > -90 or max_lat < 90:
        print("WARNING: Cut Map!")
        idx_lon = np.where((lon > min_lon) & (lon < max_lon))
        idx_lat = np.where((lat > min_lat) & (lat < max_lat))[0]
        intersect_lat_lon = np.intersect1d(idx_lon, idx_lat)
        lon = lon[intersect_lat_lon]
        lat = lat[intersect_lat_lon]

    return lon, lat


# %%
if __name__ == "__main__":
    # Test
    grid_step = 2.5
    dist_equator = sput.degree2distance_equator(grid_step)
    num_points = get_num_points(dist_equator)
    # %%
    Fek_rand = FeketeGrid(num_points=num_points,
                          pre_proccess_type=None)
    grid_type = 'fekete'
    sp_grid = f'{grid_type}_{grid_step}.npy'
    Fek_rand.save(filename=sp_grid)
    # %%
    Fib = FibonacciGrid(dist_equator)
    Fek_fib = FeketeGrid(num_points=num_points,
                         pre_proccess_type='fibonacci')
    # %%
    reload(cplt)
    im = cplt.create_multi_plot(nrows=1, ncols=2,
                                subplot_kw=dict(projection='3d'))

    fk.plot_spherical_voronoi(Fib.X, ax=im['ax'][0])
    fk.plot_spherical_voronoi(Fek_fib.X, ax=im['ax'][1])
    # %%
    reload(cplt)
    reload(fk)
    im = cplt.create_multi_plot(nrows=1, ncols=2,
                                subplot_kw=dict(projection='3d'))
    X_rand = fk.points_on_sphere(Fek_rand.num_points)
    fk.plot_spherical_voronoi(X_rand, ax=im['ax'][0])
    fk.plot_spherical_voronoi(Fek_rand.X, ax=im['ax'][1])
    # %%
    grid_type = 'fekete'
    sp_grid = f'{grid_type}_{grid_step}.npy'
    Fek_rand.save(filename=sp_grid)
    # %%
    # Load
    Fek_rand = FeketeGrid(num_points=num_points,
                          pre_proccess_type=None,
                          load_grid=sp_grid)
    x, y, z = spherical2cartesian(Fek_rand.grid['lon'],
                                  Fek_rand.grid['lat'])
    X = arange_xyz(x, y, z)
    # %%
    reload(cplt)
    im = cplt.plot_xy(
        x_arr=[np.arange(len(Fek_fib.dq)), np.arange(len(Fek_rand.dq))],
        y_arr=[Fek_fib.dq, Fek_rand.dq],
        label_arr=[f'Fibonacci', 'Random Points'],
        xlabel='iterations',
        ylabel=f'Error',
        fig_size=(9, 5),
        log=True,
        color=True,
        # ylim=(0, 3),
        loc='upper right',
        lw_arr=[1],
        ls_arr=['-'],
        mk_arr=['None', 'None'],
        stdize=False,
    )
    # %%
    # next nearest neigbor distance
    grid = Fib.grid
    nnn_dists = neighbor_distance(grid['lon'], grid['lat'])
    plt.hist(nnn_dists.flatten(), bins=100)

    # %%
    # Test to compare to classical grid
    grid_step = 2.5
    init_lat = np.arange(0, 90, 10)
    init_lon = np.arange(-179.5, 180.5, grid_step)

    lon_mesh, lat_mesh = np.meshgrid(init_lon, init_lat)
    distance = neighbor_distance(lon_mesh.flatten(), lat_mesh.flatten())

    plt.hist(distance.flatten(), bins=100)

    # %%


# %%
