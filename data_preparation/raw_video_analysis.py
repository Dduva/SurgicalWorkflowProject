import cv2
import os

OVERALL_PATH = r'/Users/dorotheeduvaux/PycharmProjects/SurgicalWorkflowProject/raw_data'


def capture_video_details(folder_path):
    ls_videos = os.listdir(folder_path)
    ls_videos.sort()
    for idx, video_str in enumerate(ls_videos):
        if video_str != '.DS_Store':
            video_path = os.path.join(folder_path, video_str)

            # Load the video file
            cap = cv2.VideoCapture(video_path)

            # Check if video opened successfully
            if not cap.isOpened():
                print("Error opening video file")
            else:
                # Get the width, height, and FPS of the video
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)

                # Print the resolution and FPS
                print(f"{video_str}: {int(width)} x {int(height)}")
                print(f"FPS: {fps}")

                # When everything done, release the video capture object
                cap.release()


def main():

    folder_path = "/Users/dorotheeduvaux 1/UCL CSML/MSc Project/RS_data/videos"
    capture_video_details(folder_path)


    # # different datasets
    # '/Users/dorotheeduvaux 1/UCL CSML/MSc Project/Video analytics/reduced_frames_data.csv'  # steps and combined steps
    # '/Users/dorotheeduvaux/PycharmProjects/SurgicalWorkflowProject/raw_data/mod_reduced_phase_frames_data.csv'  # phases
    # '/Users/dorotheeduvaux/PycharmProjects/SurgicalWorkflowProject/raw_data/reduced_coarse_phase_frames_data.csv' # coarse phases


if __name__ == '__main__':
    main()
