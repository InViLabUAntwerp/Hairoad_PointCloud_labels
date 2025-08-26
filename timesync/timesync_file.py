import os
import glob
import cv2
import numpy as np
import open3d as o3d
from timesync_helper import (
    parse_master_timestamps,
    aggregate_all_txt_timestamps,
    find_closest_matches,
    process_sep_data_to_image,
    load_transformation_matrix,
    read_point_cloud_from_files,
    save_as_ply,
    project_and_color_pointcloud
)
from DataClass.GenericDataClass import GenericDataClass
from DataClass.calibration_core import CameraCalibrator
from timesync_crawler import *
# --- ⚙️ 1. Configuration ---
master_file= r"E:\backup Hairoad sync\lablelsv2\Sensorbox Raw Lidar Data\2025-07-11_12-48-59\test_2025_07_11__12_48_59_346239.bin"
CAMERA_DATA_SEARCH_DIRECTORY = r"E:\Synchting\lablelsv2\Sensorbox Raw Camera Data\2025-07-11_12-49-12"
BASE_OUTPUT_DIRECTORY = r"E:\backup Hairoad sync\august 2025\synced_output_4_filter"
# check if the output directory exists, if not create it
if not os.path.exists(BASE_OUTPUT_DIRECTORY):
    os.makedirs(BASE_OUTPUT_DIRECTORY)




if __name__ == "__main__":
    # Find all master timestamp files in the specified directory


    process_master_file(
        master_file,
        CAMERA_DATA_SEARCH_DIRECTORY,
        BASE_OUTPUT_DIRECTORY,
        border=500,
        TIME_DIFF_THRESHOLD=0.05,
        CALIBRATION_FILE='calib1.h5',
        TRANSFORMATION_FILE='transformation_and_intrinsics7.txt',
    )
    print("\n\nAll master files have been processed.")