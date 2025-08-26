import os
import glob
import cv2
import numpy as np
import open3d as o3d
from timesync.timesync_helper import (
    parse_master_timestamps,
    aggregate_all_txt_timestamps,
    find_closest_matches,
    process_sep_data_to_image,
    load_transformation_matrix,
    read_point_cloud_from_files,
    save_as_ply,
    project_and_color_pointcloud_with_border
)
import matplotlib.cm as cm

import shutil
from DataClass.GenericDataClass import GenericDataClass
from DataClass.calibration_core import CameraCalibrator
import matplotlib
matplotlib.use('Agg')  # <-- SET THE BACKEND HERE
import matplotlib.pyplot as plt


def align_point_cloud_to_plane(pcd):
    """
    Detects the main plane in a point cloud and transforms the cloud
    so the plane is aligned with the XY plane and centered at the origin.

    Args:
        pcd (open3d.geometry.PointCloud): The input point cloud.

    Returns:
        open3d.geometry.PointCloud: The transformed point cloud.
    """
    # 1. Detect the main plane using RANSAC
    # distance_threshold: Max distance a point can be from the plane to be an inlier
    # ransac_n: Number of points sampled to estimate the plane
    # num_iterations: How many times RANSAC is run
    plane_model, inlier_indices = pcd.segment_plane(distance_threshold=0.02,
                                                    ransac_n=3,
                                                    num_iterations=1000)

    [a, b, c, d] = plane_model
    plane_normal = np.array([a, b, c])
    print(f"Detected Plane Equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # 2. Center the point cloud at the origin
    # This fulfills the "put the origin in the middle" requirement.
    center = pcd.get_center()
    pcd.translate(-center)
    print(f"Centered the point cloud by translating by {-center}")

    # 3. Calculate the rotation matrix to align the plane normal with the Z-axis
    target_normal = np.array([0, 0, 1])

    # Ensure the normal is pointing "up" (positive Z). This avoids flipping.
    if np.dot(plane_normal, target_normal) < 0:
        plane_normal = -plane_normal

    # Get the rotation axis and angle
    rotation_axis = np.cross(plane_normal, target_normal)
    rotation_axis /= np.linalg.norm(rotation_axis)
    angle = np.arccos(np.dot(plane_normal, target_normal))

    # Create the rotation matrix
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)

    # 4. Apply the rotation
    pcd.rotate(rotation_matrix, center=(0, 0, 0))
    print("Applied rotation to align plane with XY plane.")

    return pcd



def process_master_file(master_timestamp_file, camera_search_dir, base_output_dir,TIME_DIFF_THRESHOLD = 0.3, border = None,CALIBRATION_FILE=None,TRANSFORMATION_FILE=None):
    """
    Processes a single master timestamp file to find matching camera frames,
    and saves the synchronized camera images and LiDAR point clouds.
    """
    print(f"\n{'=' * 50}")
    print(f"🚀 Starting processing for: {os.path.basename(master_timestamp_file)}")
    print(f"{'=' * 50}")

    # --- 📁 2. Setup Folders for this Master File ---
    master_filename_base = os.path.splitext(os.path.basename(master_timestamp_file))[0]
    output_directory = os.path.join(base_output_dir, master_filename_base)
    camera_output_directory = os.path.join(output_directory, 'camera/front')
    lidar_output_directory = os.path.join(output_directory, 'lidar')
    # copy calib.json to the outpu_directory/calib\camera/front.json if the dir does not exist, make it.
    if not os.path.exists(os.path.join(output_directory, 'calib/camera')):
        os.makedirs(os.path.join(output_directory, 'calib/camera'), exist_ok=True)

        # Copy calib.json to the output_directory/calib/camera/front.json
    try:
        source_json_path = 'calib.json'  # Assuming it's in the execution directory
        dest_json_path = os.path.join(os.path.join(output_directory, 'calib/camera'), 'front.json')
        shutil.copy(source_json_path, dest_json_path)
        print(f"✅ Copied calibration file to: {dest_json_path}")
    except FileNotFoundError:
        print(f"⚠️  Warning: Calibration file '{source_json_path}' not found. Skipping copy.")

    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(camera_output_directory, exist_ok=True)
    os.makedirs(lidar_output_directory, exist_ok=True)
    print(f"📂 Output will be saved to: {output_directory}")

    # --- 🛠️ 3. Load Calibration and Transformation ---
    # Load transformation matrix and intrinsics
    try:
        transformation_matrix, intrinsics = load_transformation_matrix(TRANSFORMATION_FILE)
        print("✅ Transformation matrix and intrinsics loaded.")
    except Exception as e:
        print(f"❌ Critical Error loading transformation file: {e}");
        return

    # Initialize and load camera calibrator
    if not os.path.exists(CALIBRATION_FILE):
        print(f"❌ Critical Error: Calibration file not found at '{CALIBRATION_FILE}'");
        return

    Calibrator = CameraCalibrator()
    try:
        Calibrator.camera_parameters.load_parameters(CALIBRATION_FILE)
        Calibrator.camera_parameters.calculate_undistort_map()
        print("✅ Calibration loaded and ready.")
    except Exception as e:
        print(f"❌ Error loading calibration file: {e}");
        return

    # --- ☁️ 4. Load Lidar Data for this Master File ---
    lidar_data_file = os.path.splitext(os.path.splitext(master_timestamp_file)[0])[0]
    print(f"\nLoading Lidar point cloud data from: {os.path.basename(lidar_data_file)}...")
    print("Loading may take a while...")
    try:
        points_xyz_list, points_rgb_list = read_point_cloud_from_files(lidar_data_file)
        print("✅ Lidar data loaded.")
    except Exception as e:
        print(f"❌ Error loading Lidar data: {e}");
        return

    # --- 🔄 5. Find Timestamp Matches ---
    data_handler = GenericDataClass()
    print("\nFinding timestamp matches...")
    master_ts = parse_master_timestamps(master_timestamp_file)
    camera_ts = aggregate_all_txt_timestamps(camera_search_dir)  # This can be optimized if timestamps don't change
    if not master_ts or not camera_ts:
        print("❌ Could not load master or camera timestamps. Check file paths. Skipping this master file.");
        return

    closest_matches = find_closest_matches(master_ts, camera_ts)
    print(f"✅ Found potential matches for {len(closest_matches)} master frames.")

    # --- 💾 6. Process and Save Matched Frames ---
    print("\nProcessing and saving matched frames...")
    print()
    time_log_entries = []
    saved_count = 0

    for master_idx, matched_info in closest_matches.items():
        try:
            matched_dt, txt_filename, matched_frame_idx = matched_info
            master_dt = master_ts[master_idx]
            time_diff_seconds = abs(master_dt - matched_dt).total_seconds()

            if time_diff_seconds > TIME_DIFF_THRESHOLD:
                log_line = f"[SKIPPED] Master Frame {master_idx} | Time Difference: {time_diff_seconds:.6f}s > {TIME_DIFF_THRESHOLD}s"
                time_log_entries.append(log_line)
                continue

            saved_count += 1
            log_line = f"Master Frame {master_idx} matched with Frame {matched_frame_idx} in '{txt_filename}' | Time Difference: {time_diff_seconds:.6f}s"
            time_log_entries.append(log_line)

            # --- Save Camera JPEG ---
            sep_filename = os.path.splitext(os.path.splitext(txt_filename)[0])[0] + ".sep"
            sep_filepath = os.path.join(camera_search_dir, sep_filename)
            if not os.path.exists(sep_filepath):
                print(f"⚠️ Warning: SEP file not found, skipping: {sep_filepath}");
                continue

            data_handler.Open(sep_filepath)
            frame_data, _ = data_handler.GetFrame(matched_frame_idx, return_timestamp=True)
            processed_frame = process_sep_data_to_image(frame_data)
            undistorted_frame = Calibrator.camera_parameters.remap_image(processed_frame)

            jpeg_filename = master_dt.strftime("%Y%m%d%H%M%S%f") + ".jpg"
            output_path = os.path.join(camera_output_directory, jpeg_filename)
            cv2.imwrite(output_path, undistorted_frame)

            # --- Save Lidar PCD ---
            XYZ = points_xyz_list[master_idx]
            colors = points_rgb_list[master_idx]
            lin = np.int32((np.shape(XYZ)[0] / 16))
            XYZ = XYZ[:lin]
            if border is None:
                colors, XYZ = project_and_color_pointcloud_with_border(undistorted_frame, transformation_matrix, intrinsics, XYZ, vis=0, border_size = 0)
            else:
                colors, XYZ = project_and_color_pointcloud_with_border(undistorted_frame, transformation_matrix, intrinsics, XYZ,vis = 0, border_size=border)
            mask = ~np.all(colors == [0, 0, 0], axis=1)
            XYZ = XYZ[mask]
            colors = colors[mask]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(XYZ)
            pcd.colors = o3d.utility.Vector3dVector(colors / 1.0)  # Normalize for PCD
            pcd = align_point_cloud_to_plane(pcd)
            pcd_filename = master_dt.strftime("%Y%m%d%H%M%S%f") + ".pcd"

            # 2. Normalize the Z-coordinates to the [0, 1] range
            points = np.asarray(pcd.points)
            z_coords = points[:, 2]
            z_min, z_max = np.min(z_coords), np.max(z_coords)
            normalized_z = (z_coords - z_min) / (z_max - z_min)
            normalized_z = np.clip(normalized_z, 0, 1)

            # 3. Get a colormap and apply it to the normalized Z-values
            # You can use other colormaps like 'jet', 'plasma', 'inferno', etc.
            cmap = cm.get_cmap('viridis')
            colors = cmap(normalized_z)[:, :3]  # We only need the RGB values, not Alpha

            # 4. Assign the new colors to the point cloud
            #pcd.colors = o3d.utility.Vector3dVector(colors)
            #o3d.visualization.draw_geometries([pcd])
            ### --- END OF NEW CODE --- ###

            ply_filename = master_dt.strftime("%Y%m%d%H%M%S%f") + ".ply"
            pcd_output_path = os.path.join(lidar_output_directory, pcd_filename)
            ply_output_path = os.path.join(lidar_output_directory, ply_filename)

            o3d.io.write_point_cloud(pcd_output_path, pcd)
            #save_as_ply(XYZ, (colors*255), ply_output_path)  # save_as_ply expects 0-255

            print(f"\r'  -> Saved JPEG & PCD for Master Frame {master_idx}/{len(closest_matches)} (Time Diff: {time_diff_seconds:.6f}s)",
                  end="")
        except IndexError:
            print(f"❌ Error: Master index {master_idx} out of range for LiDAR data. Check for data consistency.")
        except Exception as e:
            print(f"❌ Error processing Master Frame {master_idx}: {e}")

    # --- 📝 7. Write Time Difference Log File ---
    if time_log_entries:
        log_filepath = os.path.join(output_directory, "time_differences_log.txt")
        print(f"\nWriting time difference log to: {log_filepath}")
        with open(log_filepath, 'w') as f:
            f.write("\n".join(time_log_entries))
        print("✅ Log file saved.")

    print(f"\n🎉 Processing for {os.path.basename(master_timestamp_file)} complete. Saved {saved_count} matched frames.")


if __name__ == "__main__":

    # --- ⚙️ 1. Configuration ---
    MASTER_FILES_DIRECTORY = r"E:\Synchting\lablelsv2\Sensorbox Raw Lidar Data"
    CAMERA_DATA_SEARCH_DIRECTORY = r"E:\Synchting\lablelsv2\Sensorbox Raw Camera Data\Test_2025-06-13"
    BASE_OUTPUT_DIRECTORY = r"E:\backup Hairoad sync\juli 2025\synced_output4"
    # check if the output directory exists, if not create it
    if not os.path.exists(BASE_OUTPUT_DIRECTORY):
        os.makedirs(BASE_OUTPUT_DIRECTORY)

    CALIBRATION_FILE = r'calib1.h5'
    TRANSFORMATION_FILE = 'transformation_and_intrinsics7.txt'
    TIME_DIFF_THRESHOLD = 0.3  # Max allowed time difference in seconds

    # Find all master timestamp files in the specified directory
    master_file_pattern = os.path.join(MASTER_FILES_DIRECTORY, "*.bin.txt")
    master_file_list = glob.glob(master_file_pattern)

    if not master_file_list:
        print(f"❌ No master files found in '{MASTER_FILES_DIRECTORY}'. Please check the path.")
    else:
        print(f"Found {len(master_file_list)} master files to process.")
        for master_file in master_file_list:
            process_master_file(master_file, CAMERA_DATA_SEARCH_DIRECTORY, BASE_OUTPUT_DIRECTORY,border=100,CALIBRATION_FILE = CALIBRATION_FILE,TRANSFORMATION_FILE = TRANSFORMATION_FILE)

    print("\n\nAll master files have been processed.")