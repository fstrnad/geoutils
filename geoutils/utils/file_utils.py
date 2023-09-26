import pickle
import numpy as np
import os
import sys
import xarray as xr
import geoutils.utils.general_utils as gut
import geoutils.utils.time_utils as tu
from importlib import reload
reload(tu)
reload(gut)


def get_files_in_folder(folder_path: str, verbose: bool = True) -> list:
    """
    Returns a list of all files in a given folder.

    Args:
        folder_path (str): The folder path to search in.

    Returns:
        list: A list of file paths.
    """
    assert_folder_exists(folder_path)

    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            filepath = os.path.join(root, file)
            if os.path.isfile(filepath):
                file_list.append(filepath)

    # Check that all files in the file list actually exist
    for filepath in file_list:
        assert_file_exists(filepath=filepath)

    gut.myprint(f'Found {len(file_list)} files!',
                verbose=verbose)

    return np.array(file_list)


def find_files_with_string(folder_path: str, search_string: str = None,
                           verbose: bool = True) -> list:
    """
    Finds all file paths that contain a certain string in a given folder and its subfolders.

    Args:
        folder_path (str): The folder path to search in.
        search_string (str): The search string to look for in the files.

    Returns:
        list: A list of file paths that contain the search string.
    """
    assert_folder_exists(folder_path)

    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            filepath = os.path.join(root, file)
            if os.path.isfile(filepath):
                if search_string is None:
                    file_list.append(filepath)
                else:
                    if search_string in file:
                        file_list.append(filepath)

    # Check that all files in the file list actually exist
    for filepath in file_list:
        assert_file_exists(filepath=filepath)

    gut.myprint(f'Found {len(file_list)} files!',
                verbose=verbose)

    return np.array(file_list)


def assert_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        raise ValueError("Folder path does not exist.")


def assert_file_exists(filepath):
    if not os.path.isfile(filepath):
        raise ValueError(f"File {filepath} does not exist.")


def exist_file(filepath, verbose=False):
    if os.path.exists(filepath):
        gut.myprint(f"File {filepath} exists!", verbose=verbose)
        return True
    else:
        return False


def exist_files(filepath_arr, verbose=False):
    """Check whether all files in filepath_arr exist.

    Args:
        filepath_arr (list): list of filepaths provided as strings
        verbose (bool, optional): Verbose results. Defaults to False.

    Returns:
        bool: True if all files exist, False otherwise.
    """
    if len(filepath_arr) == 0:
        gut.myprint(f"Filepath array is empty!", verbose=verbose)
        return False
    for filepath in filepath_arr:
        if not exist_file(filepath=filepath, verbose=False):
            gut.myprint(f"File {filepath} does not exist!", verbose=verbose)
            return False

    return True


def exist_folder(filepath, verbose=True):
    """
    Checks if the folder exists for the given file path.

    Args:
        file_path (str): The file path to check.

    Returns:
        bool: True if the folder exists, False otherwise.
    """
    folder_path = os.path.dirname(filepath)
    if os.path.isdir(folder_path):
        gut.myprint(f"File {filepath} exists!", verbose=verbose)
        return True
    else:
        return False


def create_folders(filepath):
    directory = os.path.dirname(filepath)
    if not exist_folder(filepath=filepath):
        os.makedirs(directory)
        print(f"Created folders for path: {filepath}")
    else:
        print(f"Folders already exist for path: {filepath}")


def print_file_location_and_size(filepath, verbose=True):
    """
    Prints the location and memory size of a given file.

    Args:
        filepath (str): The file path.

    Returns:
        None
    """
    assert_file_exists(filepath)

    file_size = os.path.getsize(filepath)
    size_unit = "bytes"
    if file_size > 1024:
        file_size /= 1024
        size_unit = "KB"
    if file_size > 1024:
        file_size /= 1024
        size_unit = "MB"
    if file_size > 1024:
        file_size /= 1024
        size_unit = "GB"

    gut.myprint(
        f"File location: {os.path.abspath(filepath)}", verbose=verbose)
    gut.myprint(f"File size: {file_size:.2f} {size_unit}", verbose=verbose)

    return None


def get_file_time_range(file_arr, verbose=True):
    time_arr = []
    for datafile in file_arr:
        file = xr.open_dataset(datafile)
        time_arr.append(file.time)

    tr = tu.find_common_time_range(time_arr)
    gut.myprint(f'Load time_range:{tr}', verbose=verbose)

    return tr


def save_np_dict(arr_dict, sp, verbose=True):
    gut.myprint(f'Store to {sp}', verbose=verbose)
    create_folders(filepath=sp)
    np.save(sp, arr_dict, allow_pickle=True)
    print_file_location_and_size(filepath=sp)

    return None


def load_pkl(sp):
    with open(sp, "rb") as fp:   # Unpickling
        all_stats = pickle.load(fp)
    return all_stats


def save_pkl_dict(arr_dict, sp, verbose=True):
    gut.myprint(f'Store to {sp}', verbose=verbose)
    with open(sp, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(arr_dict, f, pickle.HIGHEST_PROTOCOL)
    print_file_location_and_size(filepath=sp)

    return None


def save_ds(ds, filepath, unlimited_dim=None,
            classic_nc=False,
            zlib=True,
            backup=False):
    if os.path.exists(filepath):
        gut.myprint(f"File {filepath} already exists!")
        if backup:
            bak_file = f"{filepath}_backup"
            os.rename(filepath, bak_file)
            gut.myprint(f"Old file stored as {bak_file} as backup written!")
        else:
            gut.myprint(f"File {filepath} will be overwritten!")
            os.remove(filepath)
    dirname = os.path.dirname(filepath)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if isinstance(ds, xr.DataArray):
        zlib = False   # because dataarray has no var attibute

    if zlib:
        encoding = {var: {'zlib': True} for var in ds.data_vars}
        ds.to_netcdf(filepath, encoding=encoding)
    else:
        if classic_nc:
            gut.myprint('Store as NETCDF4_CLASSIC!')
            ds.to_netcdf(filepath, unlimited_dims=unlimited_dim,
                         format='NETCDF4_CLASSIC')
        else:
            ds.to_netcdf(filepath, unlimited_dims=unlimited_dim,
                         engine='netcdf4')

    gut.myprint(f"File {filepath} written!")
    print_file_location_and_size(filepath=filepath)
    return None


def load_np_dict(sp):
    gut.myprint('Load...')
    print_file_location_and_size(filepath=sp)
    return np.load(sp, allow_pickle=True).item()


def load_npy(fname):
    """Load .npy files and convert dict to xarray object.

    Args:
        fname (str): Filename

    Returns:
        converted_dic [dict]: dictionary of stored objects
    """
    gut.myprint('Load...')
    print_file_location_and_size(filepath=fname)
    dic = np.load(fname,
                  allow_pickle=True).item()
    converted_dic = {}
    for key, item in dic.items():
        # convert dict to xarray object
        if isinstance(item, dict):
            if 'data_vars' in item.keys():
                item = xr.Dataset.from_dict(item)
            elif 'data' in item.keys():
                item = xr.DataArray.from_dict(item)
        # store object to new dict
        converted_dic[key] = item

    return converted_dic


def load_xr(filepath):
    assert_file_exists(filepath=filepath)
    data = xr.open_dataset(filepath)
    return data


def load_nx(filepath):
    import networkx as nx
    gut.myprint(f"Load {filepath}...")
    assert_file_exists(filepath=filepath)
    cnx = nx.read_gml(filepath, destringizer=int)
    gut.myprint(f"... Loading {filepath} successful!")

    return cnx
