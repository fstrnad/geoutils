import numpy as np
import os
import sys
import xarray as xr
import geoutils.utils.general_utils as gut


def find_files_with_string(folder_path: str, search_string: str = None) -> list:
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

    gut.myprint(f'Found {len(file_list)} files!')

    return file_list


def assert_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        raise ValueError("Folder path does not exist.")


def assert_file_exists(file_path):
    if not os.path.isfile(file_path):
        raise ValueError(f"File {file_path} does not exist.")


def print_file_location_and_size(file_path):
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

    gut.myprint(f"File location: {os.path.abspath(file_path)}")
    gut.myprint(f"File size: {file_size:.2f} {size_unit}")
