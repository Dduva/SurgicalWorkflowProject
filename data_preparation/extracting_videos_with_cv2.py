import argparse
import os
import cv2
from pathlib import Path
import pandas
import random
import PIL
from PIL import Image

overall_total_timestamp = {
    1:  '06:14:19',
    4:  '05:54:02',
    6:  '06:49:20',
    7:  '06:40:15',
    8:  '05:17:33',
    9:  '05:37:32',
    11: '03:01:30',
    12: '03:16:10',
    13: '02:32:50',
    14: '04:58:50',
    15: '01:24:16',
    16: '03:41:07',
    17: '06:01:18',
    20: '05:33:36',
    21: '05:24:28',
    22: '06:11:55',
    23: '01:58:34',
    24: '06:52:36',
    25: '05:08:07',
    26: '04:47:45',
    27: '03:21:16',
}


adj_step_map = {
    'tumour_debulking': 0,
    'dissection_medial': 1,
    'dissection_inferior': 2,
    'dissection_superior': 3,
    'dissection_lateral': 4,
    'end': 'end',
    'start': 'start'
}

TOTAL_NUM_VIDEOS = 21
TOTAL_COUNT_PER_STEP = 12000
STEP_NUMBER = 5


def clean_timestamp_data(data):
    ts_data = pandas.read_csv(data)
    # removing rows that have duplicate time stamps and cause issues
    ts_data = ts_data[~ts_data['step'].isin(['Cystic component opened', 'Idle time as drilling?',
                                            'Drill is being used - Idle time as no label',
                                             'This is a cystic tumour - with the cystic possibly already opened at the start of video'])]
    index_to_exclude0 = ts_data.loc[(ts_data['step'] == 'rs_tumour_debunking_and_excision') &
                                    (ts_data['number'].isin([4, 7, 8, 11, 14, 22, 23, 24, 25, 26, 27]))].index
    index_to_exclude1 = ts_data.loc[(ts_data['number'] == 8)
                                    & (ts_data['step'] == 'dissection_lateral')
                                    & (ts_data['timestamp'] == '04:34:22')
                                    ].index
    index_to_exclude2 = ts_data.loc[(ts_data['number'] == 25)
                                    & (ts_data['step'] == 'Cannot see surgical field')
                                    & (ts_data['timestamp'] == '03:30:31')
                                    ].index
    index_to_exclude3 = ts_data.loc[(ts_data['number'] == 15)
                                    & (ts_data['step'] == 'Can cut from 01:11:53 to 01:24:16 (Idle time)')
                                    & (ts_data['timestamp'] == '01:11:53')
                                    ].index
    index_to_exclude4 = ts_data.loc[(ts_data['number'] == 27)
                                    & (ts_data['step'] == 'Video Finished from 03:03:37')
                                    & (ts_data['timestamp'] == '03:03:38')
                                    ].index
    index_to_exclude5 = ts_data.loc[(ts_data['number'] == 6)
                                    & (ts_data['step'] == 'Video finished at 06:49:07')
                                    & (ts_data['timestamp'] == '06:49:08')
                                    ].index

    ts_data = ts_data.drop(index_to_exclude0)
    ts_data = ts_data.drop(index_to_exclude1)
    ts_data = ts_data.drop(index_to_exclude2)
    ts_data = ts_data.drop(index_to_exclude3)
    ts_data = ts_data.drop(index_to_exclude4)
    ts_data = ts_data.drop(index_to_exclude5)
    return ts_data


def process_timestamp_data(timestamp_data, video_num, frame_rate):
    data = timestamp_data[timestamp_data['number'] == int(video_num)]
    final_timestamp = overall_total_timestamp[int(video_num)]
    data = data._append({'approach': 'RS', 'number': int(video_num), 'step': 'end',
                         'timestamp': final_timestamp}, ignore_index=True)
    if min(data['timestamp']) != '00:00:00':
        data = data._append({'approach': 'RS', 'number': int(video_num), 'step': 'start',
                             'timestamp': '00:00:00'}, ignore_index=True)
    data = data.sort_values('timestamp', ascending=True)
    data = data.reset_index().drop(columns='index')
    data['timestamp'] = pandas.to_datetime(data['timestamp'], format='%H:%M:%S')
    data = data.set_index('timestamp')
    try:
        data = data.resample(f'{round(1000/frame_rate, 6)}ms').ffill()
        print('frame_rate: ', frame_rate, ', video num sample rate: ', f'{round(1000/frame_rate, 6)}ms')
    except Exception as e:
        print(e)
    data['frame_number'] = range(1, len(data)+1)
    data['step_number'] = data['step'].map(adj_step_map).fillna("other")
    return data


def process_all_videos(arguments):
    ls_videos = os.listdir(arguments.data_input_path)
    ls_videos.sort()
    frame_data_lists = []
    timestamp_data = clean_timestamp_data(arguments.timestamp_data)
    for idx, video_str in enumerate(ls_videos):
        if video_str != '.DS_Store':
            video_num = str(int(video_str.split('.')[0][-2:]))
            video_file_path = os.path.join(arguments.data_input_path, video_str)
            output_file_path = os.path.join(arguments.data_output_path, video_num)
            if not os.path.exists(output_file_path):
                os.makedirs(Path(output_file_path))
            random.seed(idx)
            frame_log_data = process_single_video_file(video_file_path, arguments.reqd_frames_per_second,
                                                       output_file_path, timestamp_data,
                                                       arguments.process_video)
            frame_data_lists.append(frame_log_data)
    result = pandas.concat(frame_data_lists)
    result.to_csv(os.path.join(arguments.data_output_path, 'reduced_frames_log.csv'), index=False)


def retrieve_random_selections(timestamp_data_5fps):
    outputs = []
    for step in range(STEP_NUMBER):
        try:
            data = timestamp_data_5fps[timestamp_data_5fps['step_number'] == step]
            frame_numbers = data['frame_number'].values.tolist()
            # select approx 480 frames randomly for a given step -> ~ 10000 frames across all videos
            outputs.append(random.sample(frame_numbers, int(TOTAL_COUNT_PER_STEP/TOTAL_NUM_VIDEOS)))
        except Exception as e:
            print(e, f'for step: {step}')
    outputs = [item for sublist in outputs for item in sublist]
    outputs = sorted(outputs)
    return outputs


def adjust_frame_folder_string(value, video_num):
    path = r'/Users/dorotheeduvaux 1/UCL CSML/MSc Project'
    overall_path = os.path.join(path, 'RS_data/video_outputs_test', str(int(video_num)), f'video{video_num}_{value}.png')
    return overall_path


def process_single_video_file(file, reqd_frames_per_second, output_path, timestamp_data, process_video):

    video_capture = cv2.VideoCapture(file)
    # Check if camera opened successfully
    if not video_capture.isOpened():
        print("Error opening video file")
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    reqd_frames_per_second = reqd_frames_per_second  # frame_rate
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # Remove extra frames at end of video due to rounding error in the data_steps.csv
    video_num = file.split('.')[0][-2:]
    single_video_timestamp_data = process_timestamp_data(timestamp_data, video_num, frame_rate)
    single_video_timestamp_data = single_video_timestamp_data[:total_frames]
    single_video_timestamp_data = single_video_timestamp_data.reset_index()

    # Retrieve random selections of each step
    # Note that we consider a maximum number of frames min(len(single_video_timestamp_data), total_frames).
    # The rounding of the end of the video in data_steps.csv could be to the second before or after
    timestamp_data_reqd_rate = single_video_timestamp_data.loc[range(0, min(len(single_video_timestamp_data), total_frames),
                                                               int(frame_rate/reqd_frames_per_second))]
    random_selections = retrieve_random_selections(timestamp_data_reqd_rate)

    selected_data = single_video_timestamp_data[single_video_timestamp_data['frame_number'].isin(random_selections)].copy()
    selected_data['frame_file'] = selected_data['frame_number'].apply(adjust_frame_folder_string, args=(video_num,))

    frames_to_extract = selected_data['frame_number'].values

    if process_video:
        try:
            # Loop through the frames
            # CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
            for frame_num in frames_to_extract:
                # Seek to the frame
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num-1)

                # Read the frame
                ret, frame = video_capture.read()

                # Check if the frame was read successfully
                if not ret:
                    print(f"Could not read frame {frame_num} from video: {file}")
                    break

                # Apply data transformations here
                img_result = cv2.resize(frame, (720, 720))
                # Converting back to RGB
                img_result = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
                img_result = PIL.Image.fromarray(img_result)

                # Save image
                saving_path = os.path.join(output_path, f'video{video_num}_{frame_num}.png')
                # replace with your actual save path
                img_result.save(saving_path)
                # cv2.imwrite(saving_path, frame)

        except Exception as e:
            print(f'Error: {e}')
        # Close the video file
        video_capture.release()

    return selected_data


if __name__ == '__main__':
    # these are project-wide arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--reqd_frames_per_second', default=5, help='reqd_frames_per_second')
    parser.add_argument('--data_input_path',
                        default="/Users/dorotheeduvaux 1/UCL CSML/MSc Project/RS_data/videos",
                        help='Second argument')
    parser.add_argument('--data_output_path',
                        default="/Users/dorotheeduvaux 1/UCL CSML/MSc Project/RS_data/video_outputs_test",
                        help='Third argument')
    parser.add_argument('--timestamp_data',
                        default='/Users/dorotheeduvaux 1/UCL CSML/MSc Project/data_steps.csv',
                        help='csv file with all timestamp data')
    parser.add_argument('--process_video', default=True, help='Flag to process video')
    args = parser.parse_args()

    process_all_videos(args)

