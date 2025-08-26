import datetime
import os
import glob
import cv2
import numpy as np
from bisect import bisect_left
import open3d as o3d
from DataClass.GenericDataClass import GenericDataClass
from DataClass.calibration_core import CameraCalibrator
from DataClass.load_lidar_calibrate import *
from DataClass.load_lidar_calibrate import *

# --- Helper Functions (No changes needed here) ---

def parse_master_timestamps(filepath):
    """Parses timestamps from the master file, safely skipping non-timestamp lines."""
    # --- NEW: Ensure the filepath ends with .txt for robustness ---
    if not filepath.lower().endswith('.txt'):
        filepath += '.txt'
        print(f"ℹ️ Info: Appended '.txt' to master file path. New path: {filepath}")

    timestamps = []
    try:
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if not parts: continue
                timestamp_str = parts[-1]
                try:
                    dt_obj = datetime.datetime.strptime(timestamp_str, '%Y_%m_%d__%H_%M_%S_%f')
                    timestamps.append(dt_obj)
                except ValueError:
                    pass
    except FileNotFoundError:
        print(f"❌ Error: Master file not found at {filepath}")
    return timestamps


def aggregate_all_txt_timestamps(folder_path):
    """Scans ALL .txt files in a directory to find and aggregate all valid FrameIdx timestamps."""
    all_timestamps = []
    search_path = os.path.join(folder_path, "*.txt")
    file_list = glob.glob(search_path)
    if not file_list:
        print(f"⚠️ Warning: No .txt files found to search in '{folder_path}'")
        return []
    print(f"Scanning {len(file_list)} .txt files for timestamps...")
    for filepath in file_list:
        filename = os.path.basename(filepath)
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith("FrameIdx") and "dropped frame" not in line:
                    try:
                        parts = line.split()
                        frame_idx = int(parts[1])
                        timestamp_str = f"{parts[-2]} {parts[-1]}"
                        dt_obj = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
                        all_timestamps.append((dt_obj, filename, frame_idx))
                    except (ValueError, IndexError):
                        pass
    all_timestamps.sort()
    return all_timestamps


def find_closest_matches(master_timestamps, all_camera_timestamps):
    """Finds the closest camera frame for each master frame."""
    matches = {}
    if not all_camera_timestamps: return matches
    camera_dts = [item[0] for item in all_camera_timestamps]
    for master_idx, master_ts in enumerate(master_timestamps):
        pos = bisect_left(camera_dts, master_ts)
        if pos == 0:
            best_match_idx = 0
        elif pos == len(camera_dts):
            best_match_idx = len(camera_dts) - 1
        else:
            before_ts, after_ts = camera_dts[pos - 1], camera_dts[pos]
            best_match_idx = pos - 1 if master_ts - before_ts < after_ts - master_ts else pos
        matches[master_idx] = all_camera_timestamps[best_match_idx]
    return matches


def process_sep_data_to_image(sep_data):
    """Processes raw sensor data by debayering and rotating the image."""
    if not isinstance(sep_data, np.ndarray): sep_data = np.array(sep_data)
    frame = sep_data
    frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BAYER_RG2BGR)
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    return frame


# --- Main Execution ---
if __name__ == "__main__":
    # --- ⚙️ 1. Configuration ---
    MASTER_TIMESTAMP_FILE = r"E:\Synchting\lablelsv2\Sensorbox Raw Lidar Data\test_2025_06_13__15_12_20_278634.bin.txt"
    SEARCH_DIRECTORY = r"E:\Synchting\lablelsv2\Sensorbox Raw Camera Data\Test_2025-06-13"
    OUTPUT_DIRECTORY = r"E:\backup Hairoad sync\juli 2025\test"
    CALIBRATION_FILE = r'calib1.h5'
    TIME_DIFF_THRESHOLD = 0.3  # Max allowed time difference in seconds
    transformation_matrix ,intrinsics  = load_transformation_matrix('transformation_and_intrinsics7.txt')


    # --- 📁 2. Setup Folders and Calibrator ---
    # Create dedicated output directories for camera and lidar data
    CAMERA_OUTPUT_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, 'camera')
    LIDAR_OUTPUT_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, 'lidar')
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    os.makedirs(CAMERA_OUTPUT_DIRECTORY, exist_ok=True) # Ensure camera subfolder exists
    os.makedirs(LIDAR_OUTPUT_DIRECTORY, exist_ok=True) # Ensure lidar subfolder exists

    print("Initializing camera calibrator...")
    if not os.path.exists(CALIBRATION_FILE):
        print(f"❌ Critical Error: Calibration file not found at '{CALIBRATION_FILE}'");
        exit()
    Calibrator = CameraCalibrator()
    try:
        Calibrator.camera_parameters.load_parameters(CALIBRATION_FILE)
        Calibrator.camera_parameters.calculate_undistort_map();
        print("✅ Calibration loaded.")
    except Exception as e:
        print(f"❌ Error loading calibration file: {e}"); exit()

    # --- ☁️ 3. Load All Lidar Data ---
    lidar_data_file = os.path.splitext(os.path.splitext(MASTER_TIMESTAMP_FILE)[0])[0]
    print(f"\nLoading Lidar point cloud data from: {lidar_data_file}...")
    try:
        points_xyz_list, points_rgb_list = read_point_cloud_from_files(lidar_data_file)
        print("✅ Lidar data loaded.")
    except Exception as e:
        print(f"❌ Error loading Lidar data: {e}");
        exit()

    # --- 🔄 4. Find Timestamp Matches ---
    data_handler = GenericDataClass()
    print("\nFinding timestamp matches...")
    master_ts = parse_master_timestamps(MASTER_TIMESTAMP_FILE)
    camera_ts = aggregate_all_txt_timestamps(SEARCH_DIRECTORY)
    if not master_ts or not camera_ts:
        print("❌ Could not load master or camera timestamps. Check file paths. Exiting.");
        exit()
    closest_matches = find_closest_matches(master_ts, camera_ts)
    print(f"✅ Found matches for {len(closest_matches)} master frames.")

    # --- 💾 5. Process and Save Matched Frames ---
    print("\nProcessing and saving matched frames...")
    time_log_entries = []
    saved_count = 0

    for master_idx, matched_info in closest_matches.items():
        try:
            matched_dt, txt_filename, matched_frame_idx = matched_info
            master_dt = master_ts[master_idx]
            time_diff_seconds = abs(master_dt - matched_dt).total_seconds()

            # Conditionally skip or save based on the time difference threshold
            if time_diff_seconds > TIME_DIFF_THRESHOLD:
                log_line = f"[SKIPPED] Master Frame {master_idx} | Time Difference: {time_diff_seconds:.6f}s > {TIME_DIFF_THRESHOLD}s"
                time_log_entries.append(log_line)
                continue  # Skip to the next frame

            # If we are here, the time difference is acceptable.
            saved_count += 1
            log_line = f"Master Frame {master_idx} matched with Frame {matched_frame_idx} in '{txt_filename}' | Time Difference: {time_diff_seconds:.6f}s"
            time_log_entries.append(log_line)

            # --- Save Camera JPEG ---
            sep_filename = os.path.splitext(os.path.splitext(txt_filename)[0])[0] + ".sep"
            sep_filepath = os.path.join(SEARCH_DIRECTORY, sep_filename)
            if not os.path.exists(sep_filepath):
                print(f"⚠️ Warning: SEP file not found, skipping: {sep_filepath}");
                continue

            data_handler.Open(sep_filepath)
            frame_data, _ = data_handler.GetFrame(matched_frame_idx, return_timestamp=True)
            processed_frame = process_sep_data_to_image(frame_data)
            undistorted_frame = Calibrator.camera_parameters.remap_image(processed_frame)

            jpeg_filename = master_dt.strftime("%Y-%m-%d_%H-%M-%S-%f") + ".jpg"
            # Save the JPEG to the dedicated 'camera' subdirectory
            output_path = os.path.join(CAMERA_OUTPUT_DIRECTORY, jpeg_filename)
            cv2.imwrite(output_path, undistorted_frame)

            # --- Save Lidar PCD ---
            XYZ = points_xyz_list[master_idx]
            colors = points_rgb_list[master_idx]
            colors, XYZ = project_and_color_pointcloud(undistorted_frame, transformation_matrix, intrinsics, XYZ,vis = 0)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(XYZ)
            pcd.colors = o3d.utility.Vector3dVector(colors / 1.0) # Normalize colors to [0,1]

            pcd_filename = master_dt.strftime("%Y-%m-%d_%H-%M-%S-%f") + ".pcd"
            ply_filename = master_dt.strftime("%Y-%m-%d_%H-%M-%S-%f") + ".ply"
            pcd_output_path = os.path.join(LIDAR_OUTPUT_DIRECTORY, pcd_filename)
            ply_output_path = os.path.join(LIDAR_OUTPUT_DIRECTORY, ply_filename)
            o3d.io.write_point_cloud(pcd_output_path, pcd)
            save_as_ply(XYZ, (colors*255), ply_output_path)

            print(f"  -> Saved JPEG & PCD for Master Frame {master_idx} (Time Diff: {time_diff_seconds:.6f}s)")

        except Exception as e:
            print(f"❌ Error processing Master Frame {master_idx}: {e}")

    # --- 📝 6. Write Time Difference Log File ---
    if time_log_entries:
        log_filepath = os.path.join(OUTPUT_DIRECTORY, "time_differences_log.txt")
        print(f"\nWriting time difference log to: {log_filepath}")
        with open(log_filepath, 'w') as f:
            f.write("\n".join(time_log_entries))
        print("✅ Log file saved.")

    print(f"\n🎉 Processing complete. Saved {saved_count} matched frames.")
