from cdo import Cdo
import numpy as np
from importlib import reload
import geoutils.utils.file_utils as fut
import geoutils.utils.general_utils as gut
import geoutils.preprocessing.open_nc_file as onf
reload(onf)
reload(fut)
reload(gut)


def merge_files(input_files, output_file, open_ds=True):
    """Merge multiple files into one file.

    Args:
        input_files (list): list of file names
        output_file (str): output file name
    """
    fut.exist_files(input_files)
    fut.assert_folder_exists(output_file)
    gut.myprint(f'Merge files: {input_files}')

    if isinstance(input_files, np.ndarray):
        input_files = input_files.tolist()

    cdo = Cdo()
    cdo.mergetime(options='-b F32 -f nc',
                  input=input_files,
                  output=output_file)
    if open_ds:
        ds = onf.open_nc_file(output_file)
        return ds
    return output_file


def daymean(input_file, output_file):
    """Calculate the daily mean of the input file.

    Args:
        input_file (str): input file name
        output_file (str): output file name
    """
    cdo = Cdo()
    cdo.daymean(input=input_file,
                output=output_file)
    return output_file
