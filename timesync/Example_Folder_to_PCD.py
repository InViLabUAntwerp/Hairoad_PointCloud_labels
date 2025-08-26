# Example usage of process_master_file function
import os
import glob
from hairoad_calib.timesync.timesync_crawler import process_master_file


# --- ⚙️ 1. Configuration ---
MASTER_FILES_DIRECTORY = r"E:\Synchting\lablelsv2\Sensorbox Raw Lidar Data"
CAMERA_DATA_SEARCH_DIRECTORY = r"E:\Synchting\lablelsv2\Sensorbox Raw Camera Data\Test_2025-06-13"
BASE_OUTPUT_DIRECTORY = r"E:\backup Hairoad sync\juli 2025\synced_output5"
# check if the output directory exists, if not create it
if not os.path.exists(BASE_OUTPUT_DIRECTORY):
    os.makedirs(BASE_OUTPUT_DIRECTORY)

CALIBRATION_FILE = r'calib1.h5'
TRANSFORMATION_FILE = 'transformation_and_intrinsics7.txt'
TIME_DIFF_THRESHOLD = 0.3  # Max allowed time difference in seconds



if __name__ == "__main__":
    # Find all master timestamp files in the specified directory
    master_file_pattern = os.path.join(MASTER_FILES_DIRECTORY, "*.bin.txt")
    master_file_list = glob.glob(master_file_pattern)

    if not master_file_list:
        print(f"❌ No master files found in '{MASTER_FILES_DIRECTORY}'. Please check the path.")
    else:
        print(f"Found {len(master_file_list)} master files to process.")
        for master_file in master_file_list:
            process_master_file(master_file, CAMERA_DATA_SEARCH_DIRECTORY, BASE_OUTPUT_DIRECTORY,border=100)

    print("\n\nAll master files have been processed.")