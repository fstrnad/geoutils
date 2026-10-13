import imageio
import geoutils.utils.general_utils as gut
import geoutils.utils.file_utils as fut
import geoutils.plotting.plots as cplt
import cv2



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
                                              sort='Number',
                                              search_string=input_frames,
                                              verbose=verbose)
    if len(video_frames) == 0:
        gut.myprint("No frames found!")
        return None
    # Get image shape from the first frame
    img = imageio.imread(video_frames[0])
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
    if format == 'gif':
        with imageio.get_writer(output_video, fps=frame_rate, loop=0) as writer:
            for path in video_frames:  # Change range if the number of frames is different
                print(path)
                if format == 'gif':
                    frame = imageio.imread(path)
                    writer.append_data(frame)
                else:
                    video.write(cv2.imread(frame))

    #     # Save frames as a GIF using imageio
    #     # Adjust duration as needed
    #     imageio.imwrite(output_video, video_frames, format='GIF',
    #                     fps=frame_rate, loop=0)
    else:
        # Release the video if it was created
        cv2.destroyAllWindows()
        video.release()
    gut.myprint("Video created successfully!", verbose=verbose)

    return output_video
