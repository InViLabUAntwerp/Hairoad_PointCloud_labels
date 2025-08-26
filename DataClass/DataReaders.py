# @defgroup cam_python CAMERA_TOOLBOX_python
#
# These classes are part of the Op3Mech CAMERA toolbox the toolbox is
# available on bitbucket. go to : https://bitbucket.org/SeppeSels/camera_toolbox_python
#
# copyright Seppe Sels/Thomas De Kerf Invilab University of Antwerp 03-09-2021
#
# This code is for internal use only (Uantwerpen), please follow CC BY-NC-SA with the additional restriction
# that it cannot be shared outside of the University of Antwerp.
# Bugs, bugfixes and additions to the code need to be reported to Invilab (contact: Seppe Sels)
# for other licences, contact author.
#
# @ingroup cam_python

import os
import numpy as np
import re
import warnings
from datetime import datetime
import h5py


class _GenericReader:
    """Contains attributes and methods that are shared between data reader classes.

    :var file: The absolute path of file that is currently opened.
    :var currentFrame: Index of the frame that will be returned when GetFrame() is used without arguments.
    :var dataType: The datatype of the image data.
    :var height: Image height.
    :var width: Image width.
    :var numberOfFrames: Amount of recorded frames in the data set.
    :var numberOfChannels: Amount of channels one image has, e.g., mono data has 1 channel.
    :var cameraName: Vendor name, model name and serial number of the camera, or something similar to that.
    :var droppedFrames: List of indices of dropped frames if there are any.
    :var timeStamps: List of timeStamps for each recorded frame if it was enabled.
    """

    def __init__(self):
        """Class constructor."""
        self.file = None
        self.currentFrame = 0
        self.dataType = None
        self.pixelFormat = ''  # Not every measurement file has a pixel format and default needs to be savable to HDF5.
        self.height = None
        self.width = None
        self.numberOfFrames = None
        self.numberOfChannels = None
        self.cameraName = ''
        self.recordDate = ''  # Record date is not always retrievable and default needs to be savable to HDF5.
        self.droppedFrames = []
        self.timeStamps = []

    def _CheckFile(self):
        """Check if a file has been opened."""
        if self.file is None:
            raise RuntimeError("You must open a data file before you can manipulate data.")

    def _CheckFrameIndices(self, frame_idx):
        """Check and process provided frame indices."""
        if frame_idx is None:
            if not self.currentFrame < self.numberOfFrames:
                raise IndexError("You've reached the final image in the sequence. Use `ResetCounter()` to loop.")
            start_idx = self.currentFrame
            end_idx = None
            n_selected_frames = 1
            self.currentFrame += 1
        elif isinstance(frame_idx, int):
            if not frame_idx < self.numberOfFrames:
                raise IndexError(f"Frame index {frame_idx} is out of bounds: max {self.numberOfFrames - 1}.")
            start_idx = frame_idx
            end_idx = None
            n_selected_frames = 1
        elif isinstance(frame_idx, tuple):
            start_idx = frame_idx[0]
            end_idx = frame_idx[1]
            if not end_idx < self.numberOfFrames:
                raise IndexError(f"Last frame index {end_idx} is out of bounds: max {self.numberOfFrames - 1}.")
            n_selected_frames = end_idx - start_idx + 1
        else:
            raise TypeError("`frame_idx` should be None, an int or a tuple of size 2.")

        return start_idx, end_idx, n_selected_frames

    def ResetCounter(self):
        """Reset the frame sequence counter.

        The frame sequence counter is used when you use GetFrame() without passing a frame argument.
        """
        self.currentFrame = 0


class SepReader(_GenericReader):
    """Data reader for opening .sep files."""
    def __init__(self):
        super().__init__()
        self._byteMul = None

    def Open(self, file):
        """Open the data file and load metadata so frames can be retrieved.

        :param file: Path and file name of the .sep file.
        """
        file = os.path.abspath(file)
        path_sep_txt = file + "sep.txt"
        if os.path.isfile(file) and os.path.isfile(path_sep_txt):
            self.file = file
            self.currentFrame = 0
            self._ReadParameters()
        else:
            self.file = None  # Just making sure in case you reuse the same object.
            raise ValueError("Could not find .sep or corresponding sep.txt file.")

    def _ConvertFormatString(self, type_string):
        """Convert type string to data parameters."""
        if type_string == 'CV_8UC1':
            self.dataType = np.uint8
            self._byteMul = 1
            self.numberOfChannels = 1
        elif type_string == 'CV_8UC3':
            self.dataType = np.uint8
            self._byteMul = 1
            self.numberOfChannels = 3
        elif type_string == 'CV_8SC3':
            self.dataType = np.int8
            self._byteMul = 1
            self.numberOfChannels = 3
        elif type_string == 'CV_16UC1':
            self.dataType = np.uint16
            self._byteMul = 2
            self.numberOfChannels = 1
        elif type_string == 'CV_16UC3':
            self.dataType = np.uint16
            self._byteMul = 2
            self.numberOfChannels = 3
        elif type_string == 'CV_16SC3':
            self.dataType = np.int16
            self._byteMul = 2
            self.numberOfChannels = 3
        elif type_string == 'float321':
            self.dataType = np.float32
            self._byteMul = 4
            self.numberOfChannels = 1
        else:  # Default to mono uint8 and hope it works
            self.dataType = np.uint8
            self._byteMul = 1
            self.numberOfChannels = 1

    def _ReadParameters(self):
        """Read parameters from sep.txt file."""
        self._CheckFile()  # Redundant but it doesn't hurt to check.

        # Reset some parameters to defaults
        self.cameraName = None
        self.numberOfFrames = None
        self.droppedFrames = []
        self.timeStamps = []

        # Loop over all lines in the file and assign parameters
        with open(self.file + "sep.txt") as single:
            single = single.readlines()

            # Check which .sep.txt format is being used
            legacy_format = True
            for line in single:
                if 'dataFormat'.casefold() == line.casefold().split()[0]:
                    legacy_format = False
                    break

            # Use appropriate way of reading .sep.txt file
            if not legacy_format:
                for line in single:
                    if 'timeStamp'.casefold() == line.casefold().split()[0]:
                        string_time_stamp = " ".join(line.split()[2:])
                        time_stamp = datetime.strptime(string_time_stamp, '%Y-%m-%d %H:%M:%S.%f')
                        self.timeStamps.append(time_stamp)
                    elif 'droppedFrame'.casefold() == line.casefold().split()[0]:
                        frame_id = int(re.findall(r'[0-9]+', line)[0])
                        self.droppedFrames.append(frame_id)
                    elif 'width'.casefold() == line.casefold().split()[0]:
                        self.width = int(re.findall(r'[0-9]+', line)[0])
                    elif 'height'.casefold() == line.casefold().split()[0]:
                        self.height = int(re.findall(r'[0-9]+', line)[0])
                    elif 'dataFormat'.casefold() == line.casefold().split()[0]:
                        self._ConvertFormatString(line.split()[1])
                    elif 'pixelFormat'.casefold() in line.casefold().split()[0]:
                        self.pixelFormat = line.split()[1]
                    elif 'totalNumberOfFrames'.casefold() == line.casefold().split()[0]:
                        self.numberOfFrames = int(re.findall(r'[0-9]+', line)[0])
                    elif 'cameraName'.casefold() == line.casefold().split()[0]:
                        self.cameraName = " ".join(line.split()[1:])
                    elif 'recordDate'.casefold() == line.casefold().split()[0]:
                        self.recordDate = " ".join(line.split()[1:])
            else:
                for line in single:
                    if 'Time'.casefold() in line.casefold():
                        string_time_stamp = " ".join(line.split()[3:])
                        time_stamp = datetime.strptime(string_time_stamp, '%Y-%m-%d %H:%M:%S.%f')
                        self.timeStamps.append(time_stamp)
                    elif 'dropped frame'.casefold() in line.casefold():
                        frame_id = int(re.findall(r'[0-9]+', line)[0])
                        self.droppedFrames.append(frame_id)
                    elif 'width'.casefold() in line.casefold():
                        self.width = int(re.findall(r'[0-9]+', line)[0])
                    elif 'height'.casefold() in line.casefold():
                        self.height = int(re.findall(r'[0-9]+', line)[0])
                    elif 'type'.casefold() in line.casefold():
                        self._ConvertFormatString(line.split()[1])
                    elif 'TotalNumberOfFrames'.casefold() in line.casefold():
                        self.numberOfFrames = int(re.findall(r'[0-9]+', line)[0])
                    elif 'DeviceName'.casefold() in line.casefold():
                        self.cameraName = ' '.join(line.split()[1:])

            # If the total number of frames was not found loop again to look for intended amount.
            if self.numberOfFrames is None:
                for line in single:
                    if 'numberOfFrames'.casefold() in line.casefold():
                        self.numberOfFrames = int(re.findall(r'[0-9]+', line)[0])
                        warnings.warn("Failed to find recorded number of frames, using intended amount instead.")

            # Give a warning if there are dropped frames.
            if self.droppedFrames:
                warnings.warn("Dropped frames detected, their indices can be found in droppedFrames.")

    def GetFrame(self, frame_idx=None):
        """Get a (sequence of) frame(s).

        It will return the next frame in the sequence and increment the counter by 1 by default.

        :param frame_idx: None (default), or the index of the frame you want to retrieve, or a tuple indicating a range
            with the indices of the first and last frame you want to retrieve.
        :return: The specified frames in a numpy array (height, width, channels, frames), unused axes will be squeezed.
        :raise IndexError: When the provided indices are out of bounds or the counter has reached the final image.
        :raise TypeError: When an incorrect type is used for `frame_idx`.
        """
        self._CheckFile()
        start_idx, end_idx, n_selected_frames = self._CheckFrameIndices(frame_idx)

        # Retrieve frames from file.
        with open(self.file, 'rb') as file:
            offset = start_idx * self.width * self.height * self._byteMul * self.numberOfChannels
            frames = np.fromfile(file,
                                 self.dataType,
                                 offset=offset,
                                 count=self.width * self.height * self.numberOfChannels * n_selected_frames)
            # Reshape and rearrange.
            try:
                frames = frames.reshape((n_selected_frames, self.height, self.width, self.numberOfChannels))
            except ValueError:  # Occurs when frame selection is out of bounds.
                # Try to return frames that do exist.
                frames = frames.reshape((-1, self.height, self.width, self.numberOfChannels))
                if frames.size == 0:  # Occurs when no frames were retrieved from the file.
                    raise IndexError("Unable to retrieve any of the specified frames. "
                                     "This can occur when numberOfFrames is incorrect.")
                warnings.warn("Not all selected frames were present in the file, returning frames that are available.")
            frames = np.squeeze(frames.transpose((1, 2, 3, 0)))
        return frames


class H5Reader(_GenericReader):
    """Data reader for .h5 files that contain camera data."""
    def __init__(self):
        super().__init__()

    def Open(self, file):
        """Open the data file and load metadata so frames can be retrieved.

        :param file: Path and file name of the .h5 file.
        """
        file = os.path.abspath(file)
        if os.path.isfile(file):
            self.currentFrame = 0
            self.file = file
            self._ReadParameters()
        else:
            self.file = None  # Just making sure in case you reuse the same object.
            raise ValueError("Could not find .h5 file.")

    def _ReadParameters(self):
        """Retrieve camera parameters from .h5 file."""
        with h5py.File(self.file, 'r') as file:
            try:
                self.cameraName = file['camera_data'].attrs['cameraName']
                self.width = int(file['camera_data'].attrs['width'])
                self.height = int(file['camera_data'].attrs['height'])
                self.numberOfFrames = int(file['camera_data'].attrs['numberOfFrames'])
                self.dataType = np.dtype(file['camera_data'].attrs['dataType'].split(".")[1].split("'")[0]).type  # Parse the string to a numpy data type.
                self.pixelFormat = file['camera_data'].attrs['pixelFormat']
                self.numberOfChannels = int(file['camera_data'].attrs['numberOfChannels'])
                self.droppedFrames = file['camera_data'].attrs['droppedFrames'].tolist()
                self.recordDate = file['camera_data'].attrs['recordDate']
                self.timeStamps = \
                    [datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f') for x in file['camera_data'].attrs['timeStamps'].tolist()]
            except KeyError:  # Legacy files
                self.cameraName = file.attrs['cameraName']
                self.width = int(file.attrs['width'])
                self.height = int(file.attrs['height'])
                self.numberOfFrames = int(file.attrs['numberOfFrames'])
                self.dataType = file.attrs['dataType']

    def GetFrame(self, frame_idx=None):
        """Get a (sequence of) frame(s).

        It will return the next frame in the sequence and increment the counter by 1 by default.

        :param frame_idx: None (default), or the index of the frame you want to retrieve, or a tuple indicating a range
            with the indices of the first and last frame you want to retrieve.
        :return: The specified frames in a numpy array (height, width, channels, frames), unused axes will be squeezed.
        :raise IndexError: When the provided indices are out of bounds or the counter has reached the final image.
        :raise TypeError: When an incorrect type is used for `frame_idx`.
        """
        self._CheckFile()
        start_idx, end_idx, n_selected_frames = self._CheckFrameIndices(frame_idx)

        with h5py.File(self.file, 'r') as file:
            if n_selected_frames == 1:
                try:
                    frames = file['camera_data'][..., start_idx]
                except KeyError:  # Legacy files
                    frames = file['data'][..., start_idx]
            else:
                try:
                    frames = file['camera_data'][..., start_idx:end_idx+1]
                except KeyError:  # Legacy files
                    frames = file['data'][..., start_idx:end_idx + 1]
        return frames

    def ExportXML(self, xml_file_name):
        """Retrieve camera XML data from .h5 file and export it to an .xml file."""
        self._CheckFile()
        with h5py.File(self.file, 'r') as h5_file:
            try:
                xml_data = h5_file['camera_XML_Full'][()]
            except ValueError as e:
                raise ValueError("There is no camera XML data in the file.") from e
        with open(xml_file_name, "wb") as xml_file:
            for element in xml_data:
                xml_file.write(element)
