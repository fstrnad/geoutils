import geoutils.utils.statistic_utils as sut
import imageio
import matplotlib.pyplot as plt
import numpy as np
from importlib import reload
import geoutils.utils.general_utils as gut
import geoutils.utils.time_utils as tu
import geoutils.utils.file_utils as fut
import geoutils.plotting.plots as cplt
import xarray as xr

import cv2

reload(fut)
reload(cplt)
reload(tu)


def create_video(input_frames,
                 input_folder,
                 output_name,
                 fr=1,
                 format='mp4',
                 verbose=True):
    """
    Creates a video from a sequence of frames.

    Parameters:
    input_frames (str): The path to the directory containing the input frames.
    output_name (str): The name of the output video file.
    fr (int, optional): The frame rate of the output video. Defaults to 10.

    Returns:
    file: The output video file.
    """

    # Combine frames into a video using OpenCV
    frame_rate = fr  # Set frame rate (frames per second)
    if not gut.check_contains_substring(main_string=output_name,
                                        sub_string=format):
        output_name = f'{output_name}.{format}'
    output_video = output_name
    cplt.mk_plot_dir(output_video)
    video_frames = fut.find_files_with_string(input_folder,
                                              search_string=input_frames,
                                              verbose=verbose)
    if len(video_frames) == 0:
        gut.myprint("No frames found!")
        return None
    # Get image shape from the first frame
    img = cv2.imread(video_frames[0])
    height, width, layers = img.shape

    if format == 'mp4':
        video = cv2.VideoWriter(output_video,
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                frame_rate, (width, height))
    elif format == 'avi':
        video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'DIVX'),
                                frame_rate,
                                (width, height))

    frames = []
    for frame in video_frames:  # Change range if the number of frames is different
        if format == 'gif':
            frames.append(cv2.imread(frame))
        else:
            video.write(cv2.imread(frame))

    if format == 'gif':
        # Save frames as a GIF using imageio
        # Adjust duration as needed
        imageio.mimsave(output_video, frames, format='GIF',
                        fps=frame_rate)
    else:
        # Release the video if it was created
        cv2.destroyAllWindows()
        video.release()
    gut.myprint("Video created successfully!", verbose=verbose)

    return output_video


def create_video_map(
        dmap: xr.DataArray,
        output_name: str,
        fr=.8,  # frame rate as frames/second
        start=0,
        end=1,
        step=1,
        tps=None,
        ds: object = None,
        verbose=False,
        time_dim='time',
        plot_sig=False,
        ds_wind=None,
        **kwargs):
    """ Creates a video from a sequence of frames of a DataArray."""

    # Create temporary folder
    tmp_folder = fut.create_random_folder()
    extension = kwargs.pop("extension", "png")
    format = kwargs.pop("format", "mp4")

    plot_type = kwargs.pop("plot_type", "contourf")
    central_longitude = kwargs.pop("central_longitude", None)
    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)
    cmap = kwargs.pop("cmap", 'coolwarm')
    projection = kwargs.pop("projection", None)
    label = kwargs.pop("label", None)
    title_top = kwargs.pop("title", None)
    levels = kwargs.pop("levels", 10)
    significance_mask = kwargs.pop("significance_mask", None)
    lat_range = kwargs.pop("lat_range", None)
    lon_range = kwargs.pop("lon_range", None)
    dateline = kwargs.pop("dateline", False)
    dpi = kwargs.pop("dpi", 'figure')
    an_type = kwargs.pop("an_type", 'JJAS')
    var_type_u = kwargs.pop("uvar", f'U_an_{an_type}')
    var_type_v = kwargs.pop("vvar", f'V_an_{an_type}')
    lev = kwargs.pop("lev", 200)
    steps = gut.crange(start, end, step)
    wind_scale = kwargs.pop("wind_scale", 100)
    key_length = kwargs.pop("key_length", 1)
    wind_unit = kwargs.pop("wind_unit", rf'm$s^{{-1}}$ ({lev}hPa)')
    wsteps = kwargs.pop("wsteps", 2)
    for d, step in enumerate(steps):
        tmp_file_name = fut.create_random_filename(folder_path=tmp_folder,
                                                   extension=extension,
                                                   startstring=f'{d}')
        if time_dim == 'time':
            sel_tps = tu.add_time_step_tps(tps,
                                           time_step=step)
            mean_data, sig_data = tu.get_mean_tps(dmap,
                                                  tps=sel_tps)
        elif time_dim == 'day':
            mean_data = dmap.sel(day=step)
            sig_data = None

        if plot_sig:
            mean_data = mean_data*sig_data
            mean_data = xr.where(sig_data, mean_data, np.nan)

        title = f'Day {step}' if title_top is None else title_top
        im = cplt.plot_map(mean_data,
                           ds=ds,
                           title=title,
                           plot_type=plot_type,
                           cmap=cmap,
                           levels=levels,
                           label=label,
                           vmin=vmin, vmax=vmax,
                           projection=projection,
                           significance_mask=sig_data if significance_mask else None,
                           lon_range=lon_range,
                           lat_range=lat_range,
                           dateline=dateline,
                           central_longitude=central_longitude,
                           **kwargs
                           )
        if ds_wind is not None:
            mean_data_u, sig_data_u = tu.get_mean_tps(
                ds_wind.ds[var_type_u].sel(lev=lev),
                tps=sel_tps)
            mean_data_v, sig_data_v = tu.get_mean_tps(
                ds_wind.ds[var_type_v].sel(lev=lev),
                tps=sel_tps)

            if plot_sig:
                sig_data_uv = sut.sig_multiple_field(sig_mask_x=sig_data_u,
                                                     sig_mask_y=sig_data_v)
            else:
                sig_data_uv = 1
            cplt.plot_wind_field(ax=im['ax'],
                                 u=mean_data_u*sig_data_uv,
                                 v=mean_data_v*sig_data_uv,
                                 scale=wind_scale,
                                 steps=wsteps,
                                 key_length=key_length,
                                 wind_unit=wind_unit,
                                 )

        cplt.save_fig(savepath=tmp_file_name,
                      fig=im['fig'],
                      dpi=dpi)

    # This creates the final video
    create_video(input_frames=extension,
                 input_folder=tmp_folder,
                 fr=fr,
                 format=format,
                 output_name=output_name,
                 verbose=verbose)

    # Delete temporary folder
    fut.delete_folder(tmp_folder)

    return None
