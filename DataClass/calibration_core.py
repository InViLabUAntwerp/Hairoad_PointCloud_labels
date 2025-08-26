"""Module that contains all code for camera calibration."""

from __future__ import annotations
import numpy.typing as npt
from typing import Tuple, Optional
import cv2
import numpy as np
import h5py
import matplotlib.pyplot as plt
import logging


class CameraCalibrator:
    """Object used to calibrate a camera.

    :var sensor_dimensions: Sensor dimensions in pixels (w, h).
    :var feature_list: List with :py:class:`~PyCamCalib.core.feature_detection.CalibrationFeature` objects for all images.
    :var indices: Indices of all images in :py:attr:`feature_list` that were used for the current calibration.
    :var image_points_list: Contains all image space points used for the current calibration.
    :var object_points_list: Contains all object space points used for the current calibration.
    :var per_view_err: Per view re-projection errors for all images that are listed in :py:attr:`indices`.
    :var rms_reproj_error: The rms re-projection error for the current calibration.
    :var r_vecs: Rotation vectors for each image.
    :var t_vecs: Translation vectors for each image.
    :var extrinsics_std: Standard deviations for extrinsic parameters.
    :var camera_parameters: :py:class:`~PyCamCalib.core.calibration.CameraParameters` for current calibration.
    """

    def __init__(self) -> None:
        """Class constructor."""
        self._logger = logging.getLogger(__name__)
        self.sensor_dimensions = np.zeros(2, dtype=np.int32)
        self.feature_list: list = []
        self.indices: list = []
        self.image_points_list: list = []
        self.object_points_list: list = []
        self.per_view_err: npt.NDArray[np.float64] = np.zeros(1)
        self.rms_reproj_error: np.float64 = np.float64(0)
        self.r_vecs: npt.NDArray[np.float64] = np.zeros((1, 3))
        self.t_vecs: npt.NDArray[np.float64] = np.zeros((1, 3))
        self.extrinsics_std: npt.NDArray[np.float64] = np.zeros(1)
        self.camera_parameters: CameraParameters = CameraParameters()

class CameraParameters:
    """Object that contains all camera calibration parameters.

    :var f: Focal length in pixels (x, y).
    :var f_std: Standard deviation of focal length (x, y).
    :var c: Principal point in pixels (x, y).
    :var c_std: Standard deviation of principal point (x, y).
    :var s: Skew.
    :var s_std: Standard deviation of skew.
    :var radial_dist_coeffs: Radial distortion coefficients.
    :var radial_dist_coeffs: Standard deviations of radial distortion coefficients.
    :var tangential_dist_coeffs: Tangential distortion coefficients.
    :var tangential_dist_coeffs_std: Standard deviations of tangential distortion coefficients.
    :var rms_reproj_error: Overall rms re-projection error.
    :var sensor_dimensions: Sensor dimensions in pixels (w, h).
    :var map_x: x map for distortion correction.
    :var map_y: y map for distortion correction.
    :var roi: ROI used to crop image after distortion correction.
    """

    def __init__(self) -> None:
        """Class constructor."""
        self.f: npt.NDArray[np.float64] = np.zeros(2)
        self.f_std: npt.NDArray[np.float64] = np.zeros(2)
        self.c: npt.NDArray[np.float64] = np.zeros(2)
        self.c_std: npt.NDArray[np.float64] = np.zeros(2)
        self.s: np.float64 = np.float64(0)
        self.s_std: np.float64 = np.float64(0)
        self.radial_dist_coeffs: npt.NDArray[np.float64] = np.zeros(3)
        self.radial_dist_coeffs_std: npt.NDArray[np.float64] = np.zeros(3)
        self.tangential_dist_coeffs: npt.NDArray[np.float64] = np.zeros(2)
        self.tangential_dist_coeffs_std: npt.NDArray[np.float64] = np.zeros(2)
        self.rms_reproj_error: np.float64 = np.float64(0)
        self.sensor_dimensions: npt.NDArray[np.int32] = np.zeros(2, dtype=np.int32)
        self.map_x: Optional[npt.NDArray] = None
        self.map_y: Optional[npt.NDArray] = None
        self.roi: Optional[Tuple[int, int, int, int]] = None

    def get_afov(self) -> Tuple[np.float64, np.float64]:
        """Get the angular field of view in degrees.

        :returns: A tuple with the horizontal and vertical angular field of view in degrees.
        """
        h_afov = np.rad2deg(2 * np.arctan2(self.sensor_dimensions[0], 2 * self.f[0]))
        v_afov = np.rad2deg(2 * np.arctan2(self.sensor_dimensions[1], 2 * self.f[1]))

        return h_afov, v_afov

    def get_fov(self, working_distance: float) -> Tuple[float, float]:
        """Calculate the size of the field of view.

        :param working_distance: The distance between the camera and object.
        :returns: The field of view as (horizontal, vertical).
        """
        afov = self.get_afov()
        h_fov = 2 * working_distance * np.tan(np.deg2rad(afov[0] / 2))
        v_fov = 2 * working_distance * np.tan(np.deg2rad(afov[1] / 2))

        return h_fov, v_fov

    def get_intrinsics_matrix_opencv(self) -> npt.NDArray[np.float64]:
        """Get the intrinsics matrix in OpenCV format.

        :returns: A 3x3 array with the intrinsic camera parameters in OpenCV format.
        """
        intrinsics_matrix = np.array([[self.f[0], 0, self.c[0]], [0, self.f[1], self.c[1]], [0, 0, 1]])

        return intrinsics_matrix

    def get_distortion_coeffs_opencv(self) -> npt.NDArray[np.float64]:
        """Get a vector of the distortion coefficients in OpenCV format.

        :returns: A 5 element vector with the distortion coefficients in opencv format.
        """
        return np.array([self.radial_dist_coeffs[0], self.radial_dist_coeffs[1], self.tangential_dist_coeffs[0],
                         self.tangential_dist_coeffs[1], self.radial_dist_coeffs[2]])

    def calculate_undistort_map(self, alpha: float = 0,
                                fixed_point_maps: bool = False) -> None:
        """Calculate the maps necessary for pixel remapping (removing distortion).

        :param alpha: Free scaling parameter between 0 (when all the pixels in the undistorted image are valid) and 1
            (when all the source image pixels are retained in the undistorted image). If you set this at -1 OpenCV
            automatically pick a value.
        :param fixed_point_maps: Whether to transform the floating points map to a fixed-point representation. This
            speeds up pixel remapping, which might be useful for live video feeds.
        """
        intrinsics_matrix = self.get_intrinsics_matrix_opencv()
        distortion_coeffs = self.get_distortion_coeffs_opencv()
        dimensions = self.sensor_dimensions
        if fixed_point_maps:
            map_type = cv2.CV_16SC2
        else:
            map_type = cv2.CV_32FC1
        new_intrinsics_matrix, self.roi = cv2.getOptimalNewCameraMatrix(intrinsics_matrix, distortion_coeffs,
                                                                        dimensions, alpha)
        self.map_x, self.map_y = cv2.initUndistortRectifyMap(intrinsics_matrix, distortion_coeffs, None,
                                                             new_intrinsics_matrix, dimensions, map_type)

    def remap_image(self, image: npt.NDArray) -> npt.NDArray:
        """Remaps the image to remove distortion

        Undistort an image with the maps that were calculated using :py:meth:`calculate_undistort_map`.

        :param image: The image that needs to be remapped, this is either a 2D or 3D array.
        :returns: The undistorted image.
        """
        try:
            undistorted = cv2.remap(image, self.map_x, self.map_y, cv2.INTER_LINEAR)
        except cv2.error as e:
            raise Exception("You probably did not calculate the undistort map before remapping.") from e

        return undistorted

    def set_parameters_opencv(self, rms_reproj_error: float,
                              intrinsics_matrix: npt.NDArray,
                              dist_coeffs: npt.NDArray,
                              intrinsics_std: npt.NDArray,
                              sensor_dimensions: npt.NDArray) -> None:
        """Save parameters from opencv calibration to object."""
        self.f = np.array([intrinsics_matrix[0, 0], intrinsics_matrix[1, 1]])
        self.f_std = np.array([intrinsics_std[0], intrinsics_std[1]])
        self.c = np.array([intrinsics_matrix[0, 2], intrinsics_matrix[1, 2]])
        self.c_std = np.array([intrinsics_std[2], intrinsics_std[3]])
        self.s = np.float64(0)
        self.s_std = np.float64(0)
        self.radial_dist_coeffs = np.array([dist_coeffs[0], dist_coeffs[1], dist_coeffs[2]])
        self.radial_dist_coeffs_std = np.array([intrinsics_std[4], intrinsics_std[5], intrinsics_std[8]])
        self.tangential_dist_coeffs = np.array([dist_coeffs[2], dist_coeffs[3]])
        self.tangential_dist_coeffs_std = np.array([intrinsics_std[6], intrinsics_std[7]])
        self.rms_reproj_error = rms_reproj_error
        self.sensor_dimensions = sensor_dimensions
        self.map_x = None
        self.map_y = None
        self. roi = None

    def save_parameters(self, full_save_path: str, internal_path: str = "camera_calibration/camera_parameters") -> None:
        """Save calibration parameters to .h5 file

        If the file specified in `full_save_path` does not exist, a new file will be created. If the file already
        exists, the parameters will be added to the specified file. If the file already exists and it already has
        calibration parameters, these parameters will be overwritten.

        :param full_save_path: Full filepath with directory and filename.
        :param internal_path: Internal h5 file directory where the parameters should be saved, use '/'as separator.
        :raises OSError: If the path contains forbidden characters or the selected file is not compatible.
        :raises FileNotFoundError: If the specified directory does not exist.
        """
        with h5py.File(full_save_path, "a") as file:
            try:
                file.create_dataset(internal_path + "/f", data=self.f)
                file.create_dataset(internal_path + "/f_std", data=self.f_std)
                file.create_dataset(internal_path + "/c", data=self.c)
                file.create_dataset(internal_path + "/c_std", data=self.c_std)
                file.create_dataset(internal_path + "/s", data=self.s)
                file.create_dataset(internal_path + "/s_std", data=self.s_std)
                file.create_dataset(internal_path + "/radial_dist_coeffs", data=self.radial_dist_coeffs)
                file.create_dataset(internal_path + "/radial_dist_coeffs_std", data=self.radial_dist_coeffs_std)
                file.create_dataset(internal_path + "/tangential_dist_coeffs", data=self.tangential_dist_coeffs)
                file.create_dataset(internal_path + "/tangential_dist_coeffs_std", data=self.tangential_dist_coeffs_std)
                file.create_dataset(internal_path + "/rms_reproj_err", data=self.rms_reproj_error)
                file.create_dataset(internal_path + "/sensor_dimensions", data=self.sensor_dimensions)
            except ValueError:
                file[internal_path + "/f"][()] = self.f
                file[internal_path + "/f_std"][()] = self.f_std
                file[internal_path + "/c"][()] = self.c
                file[internal_path + "/c_std"][()] = self.c_std
                file[internal_path + "/s"][()] = self.s
                file[internal_path + "/s_std"][()] = self.s_std
                file[internal_path + "/radial_dist_coeffs"][()] = self.radial_dist_coeffs
                file[internal_path + "/radial_dist_coeffs_std"][()] = self.radial_dist_coeffs_std
                file[internal_path + "/tangential_dist_coeffs"][()] = self.tangential_dist_coeffs
                file[internal_path + "/tangential_dist_coeffs_std"][()] = self.tangential_dist_coeffs_std
                file[internal_path + "/rms_reproj_err"][()] = self.rms_reproj_error
                file[internal_path + "/sensor_dimensions"][()] = self.sensor_dimensions

    def load_parameters(self, full_save_path: str, internal_path: str = "camera_calibration/camera_parameters") -> None:
        """Load calibration parameters from .h5 file into object.

        :param full_save_path: Full filepath with directory and filename.
        :param internal_path: Internal h5 file directory where the parameters are saved, use '/'as separator.
        :raises KeyError: If there are no calibration parameters in the h5 file.
        :raises OSError: If the path contains forbidden characters or an incompatible file is used.
        :raises FileNotFoundError: If specified file does not exist.
        """
        with h5py.File(full_save_path, "r") as file:
            try:
                self.f = file[internal_path + "/f"][()]
                self.f_std = file[internal_path + "/f_std"][()]
                self.c = file[internal_path + "/c"][()]
                self.c_std = file[internal_path + "/c_std"][()]
                self.s = file[internal_path + "/s"][()]
                self.s_std = file[internal_path + "/s_std"][()]
                self.radial_dist_coeffs = file[internal_path + "/radial_dist_coeffs"][()]
                self.radial_dist_coeffs_std = file[internal_path + "/radial_dist_coeffs_std"][()]
                self.tangential_dist_coeffs = file[internal_path + "/tangential_dist_coeffs"][()]
                self.tangential_dist_coeffs_std = file[internal_path + "/tangential_dist_coeffs_std"][()]
                self.rms_reproj_error = file[internal_path + "/rms_reproj_err"][()]
                self.sensor_dimensions = file[internal_path + "/sensor_dimensions"][()]
            except KeyError as e:
                raise KeyError("File does not contain calibration parameters.") from e

    def plot_distortion(self) -> None:
        """Plot camera distortion."""
        width = self.sensor_dimensions[0]
        height = self.sensor_dimensions[1]
        m = self.get_intrinsics_matrix_opencv()
        d = self.get_distortion_coeffs_opencv()
        n_steps = 20
        [u, v] = np.meshgrid(np.linspace(0, width - 1, n_steps), np.linspace(0, height - 1, n_steps))
        xyz = np.linalg.solve(m, np.vstack((np.ravel(u, order='F'), np.ravel(v, order='F'), np.ones(u.size))))
        xp = xyz[0, :] / xyz[2, :]
        yp = xyz[1, :] / xyz[2, :]
        r2 = xp ** 2 + yp ** 2
        r4 = r2 ** 2
        r6 = r2 ** 3
        coef = 1 + np.dot(d[0], r2) + np.dot(d[1], r4) + np.dot(d[4], r6)
        xpp = xp * coef + 2 * np.dot(d[2], (xp * yp)) + np.dot(d[3], (r2 + 2 * xp ** 2))
        ypp = yp * coef + np.dot(d[2], (r2 + 2 * yp ** 2)) + 2 * np.dot(d[3], (xp * yp))
        u2 = m[0, 0] * xpp + m[0, 2]
        v2 = m[1, 1] * ypp + m[1, 2]
        du = u2 - np.ravel(u, order='F')
        dv = v2 - np.ravel(v, order='F')
        dr = np.reshape(np.hypot(du, dv), u.shape, order='F')

        # plot
        plt.cla()
        plt.quiver(np.ravel(u, order='F') + 1, np.ravel(v, order='F') + 1, du, dv, color='b')
        plt.plot(width / 2, height / 2, 'x', label='Sensor center')
        plt.plot(m[0, 2], m[1, 2], 'o', label='Principal point')
        contour_set = plt.contour(u[0, :] + 1, v[:, 0] + 1, dr, colors='k')
        plt.clabel(contour_set, inline=1, fontsize=10)
        plt.xlim(1, width)
        plt.ylim(1, height)
        plt.title('Radial distortion model')
        plt.xlabel('Horizontal')
        plt.ylabel('Vertical')
        plt.show()


class StereoCalibrator:
    """Object used to perform stereo calibration.

    :var feature_list_1: List with :py:class:`~PyCamCalib.core.feature_detection.CalibrationFeature` objects for all
       images from camera 1.
    :var feature_list_2: List with :py:class:`~PyCamCalib.core.feature_detection.CalibrationFeature` objects for all
       images from camera 2.
    :var indices: Indices of all images in :py:attr:`feature_list_1` and :py:attr:`feature_list_2` that were
       used for the current calibration.
    :var image_points_list_1: Contains all image space points used for the current calibration for camera 1.
    :var image_points_list_2: Contains all image space points used for the current calibration for camera 2.
    :var object_points_list: Contains all object space points used for the current calibration.
    :var per_view_err: Per view re-projection errors for all images that are listed in :py:attr:`indices`.
    :var rms_reproj_error: The rms re-projection error for the current calibration.
    :var r_vecs: Rotation vectors for each image.
    :var t_vecs: Translation vectors for each image.
    :var stereo_parameters: :py:class:`~PyCamCalib.core.calibration.StereoParameters` for current calibration.
    """

    def __init__(self) -> None:
        """Class constructor."""
        self._logger = logging.getLogger(__name__)
        self.feature_list_1: list = []
        self.feature_list_2: list = []
        self.indices: list = []
        self.image_points_list_1: list = []
        self.image_points_list_2: list = []
        self.object_points_list: list = []
        self.stereo_parameters = StereoParameters()
        self.r_vecs: npt.NDArray[np.float64] = np.zeros((1, 3))
        self.t_vecs: npt.NDArray[np.float64] = np.zeros((1, 3))
        self.per_view_err: npt.NDArray[np.float64] = np.zeros(1)
        self.rms_reproj_error: np.float64 = np.float64(0)

    def calibrate(self,
                  image_array_1: npt.NDArray,
                  image_array_2: npt.NDArray,
                  parameters_1: CameraParameters,
                  parameters_2: CameraParameters,
                  space_between_features: float,
                  board_size: Tuple[int, int],
                  marker: Optional[Tuple[int, int]] = None,
                  **kwargs) -> StereoParameters:
        """Perform stereo calibration.

        Stereo calibrate 2 cameras with 2 matching arrays of images of a calibration target, one for each image.

        :param image_array_1: Array containing all images for camera 1 that will be used for calibration. It can either
           be a 3D array (h,w,n) for grayscale images or a 4D array (h,w,c,n) for BGR images.
        :param image_array_2: Array containing all images for camera 2 that will be used for calibration. It can either
           be a 3D array (h,w,n) for grayscale images or a 4D array (h,w,c,n) for BGR images.
        :param parameters_1: The :py:class:`~PyCamCalib.core.calibration.CameraParameters` for camera 1.
        :param parameters_2: The :py:class:`~PyCamCalib.core.calibration.CameraParameters` for camera 2.
        :param space_between_features: Checker size in mm. You can set this to 1 if you don't care about scaling.
        :param board_size: Size of the board in (rows, columns)
        :param marker: Position of the marker if there is a marker present. Not implemented yet.
        :param kwargs: kwargs passed along to the :py:class:`~PyCamCalib.core.feature_detection.FeatureDetector`.
           Available keywords are:
           `expand`: set this to True, if you want to try to expand the checkerboard past obstructions
           `predict`: set this to True if you want to predict missing checkerboard corners
           `out_of_image`: set this to True if you want the predictions to include points that lie beyond the image
           borders.
        :returns: An object that contains all calibration parameter data.
        :raises TypeError: When image_array_1 or image_array_2 is not a numpy array.
        :raises CalibrationError: When no features were detected in any of the images.
        """
        if not isinstance(image_array_1, np.ndarray) or not isinstance(image_array_2, np.ndarray):
            raise TypeError("``image_array`` should be a numpy array.")
        if marker is not None:
            raise NotImplementedError("Marked checkerboards have not been implemented.")

        self._logger.info("Detecting features")
        self.construct_feature_lists(image_array_1, image_array_2, space_between_features, board_size, marker, **kwargs)

        self.construct_points_lists([])
        if not self.object_points_list:
            raise CalibrationError("Failed to detect common features in all images, unable to perform calibration.")

        self._logger.info("Performing calibration")
        self.stereo_parameters = self.opencv_calibration(parameters_1, parameters_2)

        return self.stereo_parameters

    def calibrate_indices(self, indices: list) -> StereoParameters:
        """Repeat calibration with selected samples.

        Repeat the stereo calibration with the samples specified in indices. This can be used to improve the
        stereo calibration by removing outliers.

        :param indices: A list that contains the indices that correspond to the elements in :py:data:`feature_list`
           which should be used for a new calibration. Passing an empty list will perform a calibration with all good
           images.
        :returns: An object that contains all calibration parameter data.
        :raises TypeError: When indices is not a list.
        :raises CalibrationError: When no images with detected features were selected.
        """
        if not isinstance(indices, list):
            raise TypeError("``indices`` should be a list.")

        self.construct_points_lists(indices)
        if not self.object_points_list:
            raise CalibrationError("No images with detected features remain, unable to perform calibration.")

        self.stereo_parameters = self.opencv_calibration(self.stereo_parameters.camera_parameters_1,
                                                         self.stereo_parameters.camera_parameters_2)

        return self.stereo_parameters

    def construct_feature_lists(self, image_array_1: npt.NDArray, image_array_2: npt.NDArray,
                                space_between_features: float, board_size: Tuple[int, int],
                                marker: Optional[Tuple[int, int]], **kwargs) -> None:
        """Construct a list with the calibration features for all images in image_array.

        Unless you want to perform the calibration steps separately, you should not use this method.
        """

        feature_detector = FeatureDetector(space_between_features, board_size, marker, **kwargs)
        self.feature_list_1 = []
        n_images = image_array_1.shape[-1]
        for idx in range(n_images):
            feature = feature_detector.detect_feature(image_array_1[..., idx])
            self.feature_list_1.append(feature)
            if not feature.score > 0:
                self._logger.info("Failed feature detection on cam 1 image nr " + str(idx + 1) + ".")

        self.feature_list_2 = []
        n_images = image_array_2.shape[-1]
        for idx in range(n_images):
            feature = feature_detector.detect_feature(image_array_2[..., idx])
            self.feature_list_2.append(feature)
            if not feature.score > 0:
                self._logger.info("Failed feature detection on cam 2 image nr " + str(idx + 1) + ".")

    def construct_points_lists(self, indices: list) -> None:
        """Construct lists of image points and object points for calibration.

        If indices is empty all images where a feature was detected will be used, otherwise only images that
        correspond to the elements in indices will be used. Unless you want to perform the calibration steps separately,
        you should not use this method.
        """
        self.indices = []
        self.object_points_list = []
        self.image_points_list_1 = []
        self.image_points_list_2 = []
        if not indices:
            indices = range(len(self.feature_list_1))
        for idx in indices:
            try:  # Safeguard for when indices that don't exist are passed into the function.
                feature_1 = self.feature_list_1[idx]
                feature_2 = self.feature_list_2[idx]
            except IndexError:
                pass
            else:
                if feature_1.score >= 2 and feature_2.score >= 2:
                    common_indices = np.where((feature_1.object_points == feature_2.object_points[:, None]).all(-1))
                    if common_indices[0].size != 0:
                        self.image_points_list_1.append(feature_1.image_points[common_indices[1]])
                        self.image_points_list_2.append(feature_2.image_points[common_indices[0]])
                        self.object_points_list.append(feature_1.object_points[common_indices[1]])
                        self.indices.append(idx)

    def opencv_calibration(self, parameters_1: CameraParameters, parameters_2: CameraParameters) -> StereoParameters:
        """Regular OpenCV stereo calibration.

        Unless you want to perform the calibration steps separately, you should not use this method.
        """
        self.rms_reproj_error, _, _, _, _, R, T, E, F, self.r_vecs, self.t_vecs, self.per_view_err \
            = cv2.stereoCalibrateExtended(self.object_points_list,
                                          self.image_points_list_1,
                                          self.image_points_list_2,
                                          parameters_1.get_intrinsics_matrix_opencv(),
                                          parameters_1.get_distortion_coeffs_opencv(),
                                          parameters_2.get_intrinsics_matrix_opencv(),
                                          parameters_2.get_distortion_coeffs_opencv(),
                                          parameters_1.sensor_dimensions,  # Doesn't matter
                                          None,
                                          None,
                                          flags=cv2.CALIB_FIX_INTRINSIC)

        calibration_parameters = StereoParameters()
        calibration_parameters.set_parameters_opencv(self.rms_reproj_error, R, T, E, F)
        calibration_parameters.camera_parameters_1 = parameters_1
        calibration_parameters.camera_parameters_2 = parameters_2

        return calibration_parameters

    def plot_reproj_error(self) -> None:
        """Plot mean re-projection error and re-projection error for each calibration image."""
        indices = list(map(str, self.indices))
        cam_errs = {'camera 1': self.per_view_err[:, 0], 'camera 2': self.per_view_err[:, 0]}

        fig, ax = plt.subplots()
        ax.axhline(self.rms_reproj_error, color='g', linestyle='--', label='RMS')
        x = np.arange(len(indices))
        width = 0.25
        multiplier = 0
        for camera, measurement in cam_errs.items():
            offset = width * multiplier
            ax.bar(x + offset, measurement, width, label=camera)
            multiplier += 1

        ax.set_xlabel("Image index")
        ax.set_ylabel("Reprojection error")
        ax.set_title("Reprojection error for each detected image")
        ax.set_xticks(x + width, indices)
        ax.legend(ncols=3)

        plt.show()


class StereoParameters:
    """
    Object that contains all camera calibration parameters, this includes the calibration parameters for both cameras.

    :var camera_parameters_1: :py:class:`~PyCamCalib.core.calibration.CameraParameters` for camera 1.
    :var camera_parameters_2: :py:class:`~PyCamCalib.core.calibration.CameraParameters` for camera 2.
    :var rms_reproj_error: Overall rms re-projection error.
    :var R: The rotation matrix.
    :var T: The translation matrix.
    :var E: The essential matrix.
    :var F: The fundamental matrix.
    :var R_1: Rectification transform of camera 1.
    :var R_2: Rectification transform of camera 2.
    :var P_1: Projection matrix of camera 1.
    :var P_2: Projection matrix of camera 2.
    :var Q: Disparity-to-depth mapping matrix.
    :var roi_1: ROI where all pixels for camera 1 are valid.
    :var roi_2: ROI where all pixels for camera 2 are valid.
    :var map_1_x: x map for distortion correction and rectification for camera 1.
    :var map_1_y: y map for distortion correction and rectification for camera 1.
    :var map_2_x: x map for distortion correction and rectification for camera 2.
    :var map_2_y: y map for distortion correction and rectification for camera 2.
    """
    def __init__(self) -> None:
        """Class constructor."""
        self.camera_parameters_1: CameraParameters = CameraParameters()
        self.camera_parameters_2: CameraParameters = CameraParameters()
        self.rms_reproj_error: np.float64 = np.float64(0)
        self.R: npt.NDArray[np.float4] = np.zeros((3, 3))
        self.T: npt.NDArray[np.float4] = np.zeros((3, 1))
        self.E: npt.NDArray[np.float4] = np.zeros((3, 3))
        self.F: npt.NDArray[np.float4] = np.zeros((3, 3))
        self.R_1 = None
        self.R_2 = None
        self.P_1 = None
        self.P_2 = None
        self.Q = None
        self.roi_1 = None
        self.roi_2 = None
        self.map_1_x = None
        self.map_1_y = None
        self.map_2_x = None
        self.map_2_y = None

    def set_parameters_opencv(self, rms_reproj_error: float, R: npt.NDArray, T: npt.NDArray, E: npt.NDArray,
                              F: npt.NDArray) -> None:
        """Set stereo parameters obtained from OpenCV calibration."""
        self.rms_reproj_error = rms_reproj_error
        self.R = R
        self.T = T
        self.E = E
        self.F = F

    def save_parameters(self, full_save_path: str) -> None:
        """Save calibration parameters to .h5 file

        If the file specified in `full_save_path` does not exist, a new file will be created. If the file already
        exists, the parameters will be added to the specified file. If the file already exists and it already has
        calibration parameters, these parameters will be overwritten.

        :param full_save_path: Full filepath with directory and filename.
        :raises OSError: If the path contains forbidden characters or the selected file is not compatible.
        :raises FileNotFoundError: If the specified directory does not exist.
        """
        self.camera_parameters_1.save_parameters(full_save_path, "camera_calibration/camera_1_parameters")
        self.camera_parameters_2.save_parameters(full_save_path, "camera_calibration/camera_2_parameters")
        with h5py.File(full_save_path, "a") as file:
            try:
                file.create_dataset("camera_calibration/stereo_parameters/rms_reproj_error", data=self.rms_reproj_error)
                file.create_dataset("camera_calibration/stereo_parameters/R", data=self.R)
                file.create_dataset("camera_calibration/stereo_parameters/T", data=self.T)
                file.create_dataset("camera_calibration/stereo_parameters/E", data=self.E)
                file.create_dataset("camera_calibration/stereo_parameters/F", data=self.F)
            except ValueError:
                file["camera_calibration/stereo_parameters/rms_reproj_error"][()] = self.rms_reproj_error
                file["camera_calibration/stereo_parameters/R"][()] = self.R
                file["camera_calibration/stereo_parameters/T"][()] = self.T
                file["camera_calibration/stereo_parameters/E"][()] = self.E
                file["camera_calibration/stereo_parameters/F"][()] = self.F

    def load_parameters(self, full_save_path: str) -> None:
        """Load calibration parameters from .h5 file into object.

        :param full_save_path: Full filepath with directory and filename.
        :raises KeyError: If there are no calibration parameters in the h5 file.
        :raises OSError: If the path contains forbidden characters or an incompatible file is used.
        :raises FileNotFoundError: If specified file does not exist.
        """
        self.camera_parameters_1.load_parameters(full_save_path, "camera_calibration/camera_1_parameters")
        self.camera_parameters_2.load_parameters(full_save_path, "camera_calibration/camera_2_parameters")
        with h5py.File(full_save_path, "r") as file:
            try:
                self.rms_reproj_error = file["camera_calibration/stereo_parameters/rms_reproj_error"][()]
                self.R = file["camera_calibration/stereo_parameters/R"][()]
                self.T = file["camera_calibration/stereo_parameters/T"][()]
                self.E = file["camera_calibration/stereo_parameters/E"][()]
                self.F = file["camera_calibration/stereo_parameters/F"][()]
            except KeyError as e:
                raise KeyError("File does not contain stereo calibration parameters.") from e

    def calculate_undistort_rectify_maps(self, alpha: float = 0, fixed_point_maps: bool = False) -> None:
        """Calculate rectification transforms and maps necessary for remapping.

        :param alpha: Free scaling parameter between 0 (when all the pixels in the undistorted image are valid) and 1
           (when all the source image pixels are retained in the undistorted image). If you set this at -1 OpenCV
           automatically pick a value.
        :param fixed_point_maps: Whether to transform the floating points map to a fixed-point representation. This
           speeds up pixel remapping, which might be useful for live video feeds.
        """
        sensor_dim_1 = self.camera_parameters_1.sensor_dimensions
        sensor_dim_2 = self.camera_parameters_2.sensor_dimensions
        if not np.array_equal(sensor_dim_1, sensor_dim_2):
            raise NotImplementedError("Rectification for different sensor sizes has not been implemented.")
        intrinsics_1 = self.camera_parameters_1.get_intrinsics_matrix_opencv()
        distortion_1 = self.camera_parameters_1.get_distortion_coeffs_opencv()
        intrinsics_2 = self.camera_parameters_2.get_intrinsics_matrix_opencv()
        distortion_2 = self.camera_parameters_2.get_distortion_coeffs_opencv()

        self.R_1, self.R_2, self.P_1, self.P_2, self.Q, self.roi_1, self.roi_2 \
            = cv2.stereoRectify(intrinsics_1,
                                distortion_1,
                                intrinsics_2,
                                distortion_2,
                                sensor_dim_1,
                                self.R,
                                self.T,
                                flags=cv2.CALIB_ZERO_DISPARITY,
                                alpha=alpha)

        if fixed_point_maps:
            map_type = cv2.CV_16SC2
        else:
            map_type = cv2.CV_32FC1
        self.map_1_x, self.map_1_y = cv2.initUndistortRectifyMap(intrinsics_1,
                                                                 distortion_1,
                                                                 self.R_1,
                                                                 self.P_1,
                                                                 sensor_dim_1,
                                                                 map_type)
        self.map_2_x, self.map_2_y = cv2.initUndistortRectifyMap(intrinsics_2,
                                                                 distortion_2,
                                                                 self.R_2,
                                                                 self.P_2,
                                                                 sensor_dim_2,
                                                                 map_type)

    def remap_images(self, frame_1: npt.NDArray, frame_2: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        """Remap the images so they are undistorted and rectified.

        :param frame_1: The image for camera 1 that needs to be remapped, this is either a 2D or 3D array.
        :param frame_2: The image for camera 2 that needs to be remapped, this is either a 2D or 3D array.
        :returns: The undistorted and rectified images.
        """
        try:
            rectified_1 = cv2.remap(frame_1, self.map_1_x, self.map_1_y, cv2.INTER_LINEAR)
            rectified_2 = cv2.remap(frame_2, self.map_2_x, self.map_2_y, cv2.INTER_LINEAR)
        except cv2.error as e:
            raise Exception("You probably did not calculate the undistort and rectification map before remapping.") from e

        return rectified_1, rectified_2
