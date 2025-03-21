import re
import shutil
import string
import random
import pickle
import comm
import numpy as np
import os
import xarray as xr
import geoutils.utils.general_utils as gut
import geoutils.utils.time_utils as tu

from importlib import reload
reload(tu)
reload(gut)


# ##################### Assert functions #####################

def assert_folder_exists(folder_path):
    folder_path_here = f'{folder_path}'
    folder_path = os.path.dirname(folder_path)

    if not os.path.exists(folder_path):
        if not os.path.exists(folder_path_here):
            gut.myprint(f"Folder at {folder_path} does not exist.")
            raise ValueError(f"Folder {folder_path} does not exist.")


def assert_file_exists(filepath):
    if not os.path.isfile(filepath):
        raise ValueError(f"File {filepath} does not exist.")


def exist_file(filepath, verbose=False):
    if os.path.exists(filepath):
        gut.myprint(f"File {filepath}\n exists!", verbose=verbose)
        return True
    else:
        gut.myprint(f"File {filepath}\n does not exist!", verbose=verbose)
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


def delete_path(filepath):
    """Delete a file or directory, regardless of type."""
    try:
        if os.path.isfile(filepath):  # Check if it's a file
            os.remove(filepath)  # Delete the file
            print(f"File '{filepath}' has been deleted.")
        elif os.path.isdir(filepath):  # Check if it's a directory
            # Delete the directory and all its contents
            shutil.rmtree(filepath)
            print(f"Directory '{filepath}' has been deleted.")
        else:
            print(f"Path '{filepath}' does not exist.")
    except Exception as e:
        print(f"Error deleting '{filepath}': {e}")


def exist_folder(filepath, verbose=False):
    """
    Checks if the folder exists for the given file path.

    Args:
        file_path (str): The file path to check.

    Returns:
        bool: True if the folder exists, False otherwise.
    """
    folder_path = os.path.dirname(filepath)
    if os.path.isdir(folder_path):
        gut.myprint(f"Folder {filepath} exists!", verbose=verbose)
        return True
    else:
        return False


def create_folder(filepath, verbose=True):
    directory = os.path.dirname(filepath)
    if not exist_folder(filepath=filepath):
        os.makedirs(directory)
        gut.myprint(f"Created folders for path: {filepath}",
                    verbose=verbose)


# ##################### Save functions #####################
def save_np_dict(arr_dict, sp, verbose=True):
    gut.myprint(f'Store to {sp}', verbose=verbose)
    create_folder(filepath=sp)
    np.save(sp, arr_dict, allow_pickle=True)
    print_file_location_and_size(filepath=sp)

    return None


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
            only_dim_corrds=False,
            show_progress=False,
            compression=9,
            backup=False):
    if os.path.exists(filepath):
        gut.myprint(f"File {filepath} already exists!")
        if backup:
            backup_file(filepath)
        else:
            gut.myprint(f"File {filepath} will be overwritten!")
            delete_path(filepath)
    dirname = os.path.dirname(filepath)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if isinstance(ds, xr.DataArray):
        zlib = False   # because dataarray has no var attibute
    if only_dim_corrds:
        ds = gut.delete_all_non_dimension_attributes(ds)

    gut.myprint(f"Store to {filepath}...")
    if zlib:
        encoding = {var: {'zlib': zlib,
                        #   "complevel": compression
                          }
                    for var in ds.data_vars}
        write_job = ds.to_netcdf(filepath, encoding=encoding,
                                 compute=False)
    else:
        if classic_nc:
            gut.myprint('Store as NETCDF4_CLASSIC!')
            write_job = ds.to_netcdf(filepath, unlimited_dims=unlimited_dim,
                                     format='NETCDF4_CLASSIC')
        else:
            write_job = ds.to_netcdf(filepath, unlimited_dims=unlimited_dim,
                                     engine='netcdf4', compute=False)

    if show_progress:
        from dask.diagnostics import ProgressBar
        with ProgressBar(minimum=2):
            write_job.compute()
    else:
        write_job.compute()

    gut.myprint(f"File {filepath} written!")
    print_file_location_and_size(filepath=filepath)
    return None


def backup_file(filepath):
    bak_file = f"{filepath}_backup"
    os.rename(filepath, bak_file)
    gut.myprint(f"Old file stored as {bak_file} as backup written!")


def save_to_zarr(ds, filepath, mode=None, backup=False):
    if exist_file(filepath):
        if backup:
            backup_file(filepath)
        else:
            gut.myprint(f"File {filepath} will be overwritten!")
            delete_path(filepath)
    if not exist_folder(filepath):
        dirname = os.path.dirname(filepath)
        os.makedirs(dirname)
    ds.to_zarr(filepath, mode=mode)

# ##################### Load functions #####################


def load_pkl(sp):
    with open(sp, "rb") as fp:   # Unpickling
        all_stats = pickle.load(fp)
    return all_stats


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

# ##################### file operations #####################


def create_random_folder(path='./', k=8, extension=None):
    # Generate a random folder name
    folder_name = ''.join(random.choices(
        string.ascii_letters + string.digits, k=k))

    # Create the full path
    folder_path = os.path.join(path, folder_name)

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        gut.myprint(f"Random folder '{folder_name}' created at: {folder_path}")
    else:
        gut.myprint(f"Folder '{folder_name}' already exists at: {folder_path}")

    return folder_path


def create_filename(folder_path='./', k=1,
                    file_string=None,
                    startstring=None,
                    extension=None):
    while True:
        # Generate a random filename
        if file_string is None:
            file_name = ''.join(random.choices(string.ascii_letters + string.digits,
                                               k=k))
        else:
            file_name = file_string
        # Add startstring if provided
        if startstring is not None:
            file_name = f'{startstring}_{file_name}'

        # Add extension if provided
        if extension:
            file_name += f".{extension.strip('.')}"

        # Create the full file path
        file_path = os.path.join(folder_path, file_name)

        # Check if the file already exists, if not, return the filename
        if not os.path.exists(file_path):
            return file_path
        else:
            file_string = None


def get_human_readable_size(size_in_bytes):
    # List of units in order: bytes, KB, MB, GB, TB, etc.
    units = ['bytes', 'KB', 'MB', 'GB', 'TB', 'PB']

    if size_in_bytes == 0:
        return "0 bytes"

    # Find the appropriate unit
    unit_index = 0
    while size_in_bytes >= 1024 and unit_index < len(units) - 1:
        size_in_bytes /= 1024.0
        unit_index += 1

    # Format the result to two decimal places
    return f"{size_in_bytes:.2f} {units[unit_index]}"


def get_folder_size(folder_path):
    total_size = 0

    # Walk through all the files and subdirectories in the given folder
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            # Get the full path to each file
            file_path = os.path.join(dirpath, filename)
            # Check the size of each file and add it to the total
            total_size += os.path.getsize(file_path)

    total_size = get_human_readable_size(total_size)

    return total_size


def print_file_location_and_size(filepath, verbose=True):
    """
    Prints the location and memory size of a given file.

    Args:
        filepath (str): The file path.

    Returns:
        None
    """
    if isinstance(filepath, str):
        filepath = [filepath]
    for file in filepath:
        if os.path.isfile(file):
            # If it's a file, return the size in bytes
            file_size = os.path.getsize(file)
            total_size = get_human_readable_size(file_size)
        else:
            total_size = get_folder_size(file)

        gut.myprint(f"Total size of '{file}':\n   {total_size} bytes",
                    verbose=verbose)

    return None


def get_file_time_range(file_arr, round_hour=True, verbose=True):
    time_arr = []
    for datafile in file_arr:
        file = xr.open_dataset(datafile)
        time_arr.append(file.time)

    tr = tu.find_common_time_range(time_arr, round_hour=round_hour)
    tr_str = tu.tps2str(tr)
    gut.myprint(f'Load time_range:{tr_str}', verbose=verbose)

    return tr


def check_file_time_equity(file_arr, verbose=True):
    time_arr = []
    for datafile in file_arr:
        file = xr.open_dataset(datafile)
        time_arr.append(file.time)
    for i, timetest in enumerate(time_arr[1:]):
        if not tu.check_hour_equality(da1=time_arr[0],
                                      da2=timetest):
            gut.myprint(f'WARNING! Different hour timeing in {file_arr[i+1]}!',
                        color='red', verbose=verbose)
    return True


def delete_folder(folder_path):
    """
    Deletes a folder and its contents.

    Args:
        folder_path (str): The path to the folder to be deleted.

    Raises:
        OSError: If an error occurs while deleting the folder.

    Returns:
        None
    """
    try:
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' and its contents deleted successfully.")
    except OSError as e:
        print(f"Error: {folder_path} - {e.strerror}")


def get_filename_path(file_path):
    return os.path.basename(file_path)


def extract_numbers(s):
    """Extract all numbers from a string and return as a list of integers."""
    return [int(num) for num in re.findall(r'\d+', s)]


def find_common_numbers(arr):
    """Find numbers that are common across all strings."""
    all_numbers = [set(extract_numbers(s)) for s in arr if extract_numbers(s)]
    if not all_numbers:
        return set()  # No numbers at all in any string
    common_numbers = set.intersection(*all_numbers)
    return common_numbers


def sort_filenames_by_number(arr):
    # Get the common numbers across all strings with numbers
    common_numbers = find_common_numbers(arr)

    # Check if each string has at least one number
    has_number = any(extract_numbers(s) for s in arr)
    if not has_number:
        print("No numbers found in any string!")
        return arr  # No sorting if no numbers are found in any string

    # Define the sorting key function
    def sort_key(s):
        numbers = extract_numbers(s)
        if len(numbers) == len(common_numbers):
            return float('inf')
        unique_numbers = [num for num in numbers if num not in common_numbers]
        # Sort by unique number if exists; otherwise, by first common number
        return unique_numbers[0] if unique_numbers else numbers[0]

    # Sort the array with our custom key
    return sorted(arr, key=sort_key)

# def sort_filenames_by_number(file_list):
#     def extract_number(filename):
#         # Using regular expression to extract the leading number from the filename
#         match = re.match(r'^(\d+)', filename)
#         if match:
#             return int(match.group(1))
#         # Assign a very large number for filenames without numbers
#         return float('inf')

#     # Sort the file paths based on the extracted numbers from filenames
#     sorted_paths = sorted(
#         file_list, key=lambda path: extract_number(os.path.basename(path)))
#     return sorted_paths


def get_files_in_folder(folder_path: str,
                        verbose: bool = True,
                        include_string=None,
                        ignore_string=None,
                        sort=True) -> list:
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

    if include_string is not None:
        file_list = find_files_with_string(folder_path=folder_path,
                                           search_string=include_string,
                                           sort=sort,
                                           verbose=False)

    # Check that all files in the file list actually exist
    for filepath in file_list:
        assert_file_exists(filepath=filepath)
    if sort:
        file_list = sort_filenames_by_number(file_list)

    if ignore_string is not None:
        file_list = [f for f in file_list if ignore_string not in f]

    gut.myprint(f'Found {len(file_list)} files!',
                verbose=verbose)

    return np.array(file_list)


def find_files_with_string(folder_path: str, search_string: str = None,
                           sort: str = 'Number', verbose: bool = True) -> list:
    """
    Finds all file paths that contain a certain string in a given folder and its subfolders.

    Args:
        folder_path (str): The folder path to search in.
        search_string (str): The search string to look for in the files.

    Returns:
        list: A list of file paths that contain the search string.
    """
    print(folder_path)
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

    if sort:
        if sort == 'Number':
            file_list = sort_filenames_by_number(file_list)
        else:
            file_list.sort()

    # Check that all files in the file list actually exist
    for filepath in file_list:
        assert_file_exists(filepath=filepath)

    gut.myprint(f'Found {len(file_list)} files!',
                verbose=verbose)

    return np.array(file_list)
