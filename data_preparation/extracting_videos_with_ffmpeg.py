import argparse
import os
from pathlib import Path


def process_all_videos(arguments):
    ls_videos = os.listdir(arguments.data_input_path)
    ls_videos.sort()
    for video_str in ls_videos:
        if video_str[0] != '.':  # not in ['.DS_Store']:  # ['video23.mp4', 'video24.mp4', 'video25.mp4', 'video26.mp4', 'video27.mp4']:
            video_num = str(int(video_str.split('.')[0][-2:]))
            video_file_path = Path(os.path.join(arguments.data_input_path, video_str))
            output_file_path = Path(os.path.join(arguments.data_output_path, video_num))
            if not os.path.exists(output_file_path):
                os.makedirs(output_file_path)
            process_single_video_file(video_file_path, arguments.reqd_seconds_per_frame, output_file_path)


def process_single_video_file(file, reqd_seconds_per_frame, output_path):
    try:
        os.system(
            f'/usr/local/bin/ffmpeg -i "{file}" -vf "scale=-1:250,fps={reqd_seconds_per_frame}" "{output_path / file.stem}_%06d.png"'
        )
        # os.system(
        #     f'/usr/local/bin/ffmpeg -i "{file}" -vf "scale=720:720,fps={reqd_seconds_per_frame}" "{output_path / file.stem}_%06d.png"'
        # )
    except Exception as e:
        print(f'Problem with video {file.stem}: {e}')


if __name__ == '__main__':
    # these are project-wide arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--reqd_seconds_per_frame', default=5, help='Reqd_seconds_per_frame')
    # parser.add_argument('--data_input_path',
    #                     default="/Users/dorotheeduvaux 1/UCL CSML/MSc Project/RS_data/videos",
    #                     help='Second argument')
    parser.add_argument('--data_input_path',
                        default="/Volumes/Extreme SSD/RS data/Raw videos",
                        help='Second argument')
    # parser.add_argument('--data_output_path',
    #                     default="/Users/dorotheeduvaux 1/UCL CSML/MSc Project/RS_data/video_outputs_test",
    #                     help='Third argument')
    parser.add_argument('--data_output_path',
                        default="/Volumes/Extreme SSD/RS data/Frames at 5fps 250 res",
                        help='Third argument')
    args = parser.parse_args()

    process_all_videos(args)


# python3.9 extracting_videos_with_ffmpeg.py --reqd_seconds_per_frame 5 --data_input_path '/Users/dorotheeduvaux 1/UCL CSML/MSc Project/RS_data/videos' --data_output_path '/Users/dorotheeduvaux 1/UCL CSML/MSc Project/RS_data/video_outputs'


