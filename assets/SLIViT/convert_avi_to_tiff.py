
import os
import cv2
import time
import numpy as np
import pandas as pd
from PIL import Image


def convert_avi_to_tiff(avi_path, output_dir='output_frames', npy_output_dir='output_npy'):
    # Create the output directory if it doesn't exist
    filename = avi_path.split('/')[-1].split('.')[0]

    # output_dir = os.path.join(output_dir, filename)
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    output_npy_dir = os.path.join(npy_output_dir, filename)
    if not os.path.exists(output_npy_dir):
        os.makedirs(output_npy_dir)

    # Initialize a variable to store dtype
    frame_dtype = None
    frames_list = []

    # Load the video file
    cap = cv2.VideoCapture(avi_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Process each frame
    for i in range(frame_count):
        # Read frame
        ret, frame = cap.read()
        frames_list.append(frame)
        if not ret:
            print("Failed to retrieve frame. Ending process.")
            break
        # Set the dtype based on the first frame
        if frame_dtype is None:
            frame_dtype = frame.dtype
            # print(f"Frame dtype: {frame_dtype}")

        # Convert the frame to RGB format for saving with Pillow
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        # # Save the frame as TIFF with zero-padded suffix
        # save_filename = os.path.join(output_dir, f"{filename}_{str(i).zfill(4)}.tiff")
        # img.save(save_filename)
        # # print(f"Saved frame {i + 1} as {filename}")

    # Stack frames along the third dimension to create a 3D volume
    video_volume = np.stack(frames_list, axis=0)

    # Save the entire video as a 3D NumPy volume
    volume_filename = os.path.join(npy_output_dir, f"{filename}.npy")
    # print(f"Saving video volume as {volume_filename} with shape {video_volume.shape}")
    np.save(volume_filename, video_volume)

    # Release the video capture object
    cap.release()
    print("Conversion complete.")



# # Example usage
# avi_path = parent_dir + 'Videos/0X100E491B3CD58DE2.avi'
# output_dir = parent_dir + 'Videos_tiff/'
# np_output_dir = parent_dir + 'Videos_npy/'
# convert_avi_to_tiff(avi_path, output_dir, np_output_dir)

home_directory = os.path.expanduser('~')

csv_file_path = home_directory + 'OCTCubeM/assets/SLIViT/meta/echonet.csv'

# Load the CSV file
df = pd.read_csv(csv_file_path)
print(df.head(), df.shape)

parent_dir = home_directory + '/EchoNet/EchoNet-Dynamic/'
output_dir = parent_dir + 'Videos_tiff/'
np_output_dir = parent_dir + 'Videos_npy/'
start = time.time()
for idx, row in df.iterrows():
    video_fp = row['path'] + '.avi'
    convert_avi_to_tiff(video_fp, output_dir, np_output_dir)
    if idx % 100 == 0:
        print(f"Processed {idx} videos with time: {time.time() - start}")