import pickle
import numpy as np
import os
import sys
import xarray as xr
import geoutils.utils.general_utils as gut


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
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path):
                if search_string is None:
                    file_list.append(file_path)
                else:
                    if search_string in file:
                        file_list.append(file_path)

    # Check that all files in the file list actually exist
    for file_path in file_list:
        assert_file_exists(file_path=file_path)

    gut.myprint(f'Found {len(file_list)} files!',
                verbose=verbose)

    return np.array(file_list)


def assert_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        raise ValueError("Folder path does not exist.")


def assert_file_exists(file_path):
    if not os.path.isfile(file_path):
        raise ValueError(f"File {file_path} does not exist.")


def exist_file(filepath, verbose=True):
    if os.path.exists(filepath):
        gut.myprint(f"File {filepath} exists!", verbose=verbose)
        return True
    else:
        return False


def print_file_location_and_size(file_path, verbose=True):
    """
    Prints the location and memory size of a given file.

    Args:
        file_path (str): The file path.

    Returns:
        None
    """
    assert_file_exists(file_path)

    file_size = os.path.getsize(file_path)
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
        f"File location: {os.path.abspath(file_path)}", verbose=verbose)
    gut.myprint(f"File size: {file_size:.2f} {size_unit}", verbose=verbose)

    return None


def save_np_dict(arr_dict, sp, verbose=True):
    gut.myprint(f'Store to {sp}', verbose=verbose)
    np.save(sp, arr_dict, allow_pickle=True)
    print_file_location_and_size(file_path=sp)

    return None


def save_pkl_dict(arr_dict, sp, verbose=True):
    gut.myprint(f'Store to {sp}', verbose=verbose)
    with open(sp, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(arr_dict, f, pickle.HIGHEST_PROTOCOL)
    print_file_location_and_size(file_path=sp)

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

    if classic_nc:
        gut.myprint('Store as NETCDF4_CLASSIC!')
        ds.to_netcdf(filepath, unlimited_dims=unlimited_dim,
                     format='NETCDF4_CLASSIC')
    else:
        ds.to_netcdf(filepath, unlimited_dims=unlimited_dim,
                     engine='netcdf4')

    if zlib:
        encoding = {var: {'zlib': True} for var in ds.data_vars}
        ds.to_netcdf(filepath, encoding=encoding)
    gut.myprint(f"File {filepath} written!")
    print_file_location_and_size(file_path=filepath)
    return None


def load_np_dict(sp):
    gut.myprint('Load...')
    print_file_location_and_size(file_path=sp)
    return np.load(sp, allow_pickle=True).item()


def load_npy(fname):
    """Load .npy files and convert dict to xarray object.

    Args:
        fname (str): Filename

    Returns:
        converted_dic [dict]: dictionary of stored objects
    """
    gut.myprint('Load...')
    print_file_location_and_size(file_path=fname)
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
    assert_file_exists(file_path=filepath)
    data = xr.open_dataset(filepath)
    return data