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
import h5py
from DataClass.DataReaders import SepReader, H5Reader


class GenericDataClass():
    # TODO: add chunked data loading/writing for large datasets.
    """Generic data class for reading stored camera data from .h5 and .sep files.

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
        super().__init__()
        self._dataReader = SepReader()  # Using SepReader by default.

    @property
    def currentFrame(self):
        return self._dataReader.currentFrame

    @property
    def dataType(self):
        return self._dataReader.dataType

    @property
    def height(self):
        return self._dataReader.height

    @property
    def width(self):
        return self._dataReader.width

    @property
    def numberOfFrames(self):
        return self._dataReader.numberOfFrames

    @property
    def numberOfChannels(self):
        return self._dataReader.numberOfChannels

    @property
    def cameraName(self):
        return self._dataReader.cameraName

    @property
    def droppedFrames(self):
        return self._dataReader.droppedFrames

    @property
    def timeStamps(self):
        return self._dataReader.timeStamps

    @property
    def recordDate(self):
        return self._dataReader.recordDate

    @property
    def file(self):
        return self._dataReader.file

    def Open(self, file):
        """Select the correct data reader, open the file, and load metadata.

        If a .h5 file was provided with stored camera class parameters these will be loaded automatically.

        :param file: Path and file name of the .h5 or .sep file.
        """
        # Check the filetype.
        filename = os.path.basename(file)
        filetype = filename.split('.')[-1]

        # Initialize corresponding data reader.
        if filetype == 'sep' or filetype == 'Sep':
            self._dataReader = SepReader()
            self._dataReader.Open(file)
        elif filetype == 'h5':
            self._dataReader = H5Reader()
            self._dataReader.Open(file)
            self.LoadClassParams(file)
        else:
            raise ValueError("Incompatible file type.")

    def GetFrame(self, frame_idx=None, raw=False, return_timestamp=False):
        """Get a (sequence of) frame(s).

        It will return the next frame in the sequence and increment the counter by 1 by default.

        :param frame_idx: None (default), or the index of the frame you want to retrieve, or a tuple indicating a range
            with the indices of the first and last frame you want to retrieve.
        :param raw: Whether to return the raw or preprocessed frame.
        :return: The specified frames in a numpy array (height, width, channels, frames), unused axes will be squeezed.
        :raise IndexError: When the provided indices are out of bounds or the counter has reached the final image.
        :raise TypeError: When an incorrect type is used for `frame_idx`.
        """
        self._CheckFile()
        frame = self._dataReader.GetFrame(frame_idx)
        if not raw:
            pass
            #frame = super().PreProcess(frame)
        if return_timestamp:
            return frame, self._dataReader.timeStamps[frame_idx]
        return frame

    def ResetCounter(self):
        """Reset the frame sequence counter.

        The frame sequence counter is used when you use GetFrame() without passing a frame argument.
        """
        self._dataReader.ResetCounter()

    def SaveToH5(self, h5_file_name, xml_file_name=None, mode='w', compression='gzip'):
        """Save data in currently opened file to an HDF5 file.

        :param h5_file_name: Absolute path of target file.
        :param xml_file_name: Absolute path of camera parameter .xml file, optional.
        :param mode: HDF5 file mode, create/truncate by default. Use 'a' for appending data.
        :param compression: Which compression mode to use, use `None` to disable compression.
        """
        frames = self._dataReader.GetFrame((0, self._dataReader.numberOfFrames - 1))
        with h5py.File(h5_file_name, mode) as h5_file:
            # Save Data.
            h5_file.create_dataset('camera_data', data=frames, dtype=self._dataReader.dataType, compression=compression)

            # Save Parameters.
            h5_file['camera_data'].attrs['dataType'] = str(self._dataReader.dataType)
            h5_file['camera_data'].attrs['pixelFormat'] = self._dataReader.pixelFormat
            h5_file['camera_data'].attrs['height'] = self._dataReader.height
            h5_file['camera_data'].attrs['width'] = self._dataReader.width
            h5_file['camera_data'].attrs['numberOfFrames'] = self._dataReader.numberOfFrames
            h5_file['camera_data'].attrs['numberOfChannels'] = self._dataReader.numberOfChannels
            h5_file['camera_data'].attrs['cameraName'] = self._dataReader.cameraName
            h5_file['camera_data'].attrs['recordDate'] = self._dataReader.recordDate
            h5_file['camera_data'].attrs['droppedFrames'] = np.array(self._dataReader.droppedFrames, dtype=np.int32)
            if self._dataReader.timeStamps:
                h5_file['camera_data'].attrs['timeStamps'] = \
                    np.array([x.strftime('%Y-%m-%d %H:%M:%S.%f') for x in self._dataReader.timeStamps], dtype=object)
            else:
                h5_file['camera_data'].attrs['timeStamps'] = []

        # Save camera XML data.
        if xml_file_name is not None:
            xml_file_name = os.path.abspath(xml_file_name)
            AppendXMLToH5(h5_file_name, xml_file_name)

    def ExportXML(self, xml_file_name):
        """Retrieve camera XML data from opened .h5 file and export it to an .xml file.

        :param xml_file_name: Path and file name for .xml file.
        """
        self._CheckFile()
        xml_file_name = os.path.abspath(xml_file_name)
        if isinstance(self._dataReader, H5Reader):
            self._dataReader.ExportXML(xml_file_name)
        else:
            raise ValueError(".sep files do not contain XML data.")

    def _CheckFile(self):
        """Check if a file has been opened."""
        if self._dataReader.file is None:
            raise RuntimeError("You must open a data file before you can manipulate data.")


def AppendXMLToH5(h5_file_name, xml_file_name):
    """Appends camera .xml file data to .h5 file.

    :param h5_file_name: Absolute path of target file.
    :param xml_file_name: Absolute path of camera parameter .xml file.
    """
    h5_file_name = os.path.abspath(h5_file_name)
    if not os.path.isfile(h5_file_name):
        raise ValueError("Provided .h5 file does not exist.")
    xml_file_name = os.path.abspath(xml_file_name)
    if not os.path.isfile(xml_file_name):
        raise ValueError("Provided .xml file does not exist.")
    with h5py.File(h5_file_name, 'a') as h5_file:
        with open(xml_file_name, 'r') as xml_file:
            xml_data = xml_file.readlines()
        feature_names = np.array(xml_data, dtype=object)
        h5_file.create_dataset('camera_XML_Full', data=feature_names)


def SaveArrayToH5(array, h5_file_name, n_frames, n_channels, pixel_format='', camera_name='',
                  dropped_frames=None, time_stamps=None, xml_file_name=None, mode='w', compression='gzip'):
    """Save numpy array to HDF5 file.

    The data should be in the order (height, width, channels, frames).

    :param array: Numpy array containing the image data.
    :param h5_file_name: Absolute path of target file.
    :param n_frames: Number of frames (required for correct data retrieval).
    :param n_channels: Number of channels (required for correct data retrieval).
    :param pixel_format: The camera pixel format.
    :param camera_name: Camera name.
    :param dropped_frames: List containing indices of dropped frames.
    :param time_stamps: List containing datetime time stamps for each frame in the array.
    :param xml_file_name: Absolute path of camera parameter .xml file, optional.
    :param mode: HDF5 file mode, create/truncate by default. Use 'a' for appending data.
    :param compression: Which compression mode to use, use `None` to disable compression.
    """
    h5_file_name = os.path.abspath(h5_file_name)
    with h5py.File(h5_file_name, mode) as h5_file:
        # Save data.
        h5_file.create_dataset('camera_data', data=array, dtype=array.dtype, compression=compression)

        # Save parameters
        h5_file['camera_data'].attrs['dataType'] = str(array.dtype)
        h5_file['camera_data'].attrs['pixelFormat'] = pixel_format
        h5_file['camera_data'].attrs['height'] = array.shape[0]
        h5_file['camera_data'].attrs['width'] = array.shape[1]
        h5_file['camera_data'].attrs['numberOfFrames'] = n_frames
        h5_file['camera_data'].attrs['numberOfChannels'] = n_channels
        h5_file['camera_data'].attrs['cameraName'] = camera_name
        if dropped_frames is not None:
            h5_file['camera_data'].attrs['droppedFrames'] = dropped_frames
        else:
            h5_file['camera_data'].attrs['droppedFrames'] = []
        if time_stamps is not None:
            h5_file['camera_data'].attrs['timeStamps'] = time_stamps
        else:
            h5_file['camera_data'].attrs['timeStamps'] = []

    # Save camera XML data.
    if xml_file_name is not None:
        xml_file_name = os.path.abspath(xml_file_name)
        AppendXMLToH5(h5_file_name, xml_file_name)


def SaveSepToH5(sep_file_name, h5_file_name, xml_file_name=None, mode='w', compression='gzip'):
    """Save data in a .sep file to a .h5 file.

    A new file is created or an existing file is truncated by default, use another mode when needed, e.g., when you
    want to append data to an existing file.

    :param sep_file_name: Absolute filepath of .sep file.
    :param h5_file_name: Absolute path of target file.
    :param xml_file_name: Absolute path of camera parameter .xml file, optional.
    :param mode: HDF5 file mode, create/truncate by default. Use 'a' for appending data.
    :param compression: Which compression mode to use, use `None` to disable compression.
    """
    sep_file_name = os.path.abspath(sep_file_name)
    h5_file_name = os.path.abspath(h5_file_name)
    sep_reader = SepReader()
    sep_reader.Open(sep_file_name)
    frames = sep_reader.GetFrame((0, sep_reader.numberOfFrames - 1))
    with h5py.File(h5_file_name, mode) as h5_file:
        # Save Data.
        h5_file.create_dataset('camera_data', data=frames, dtype=sep_reader.dataType, compression=compression)

        # Save Parameters.
        h5_file['camera_data'].attrs['dataType'] = str(sep_reader.dataType)
        h5_file['camera_data'].attrs['pixelFormat'] = sep_reader.pixelFormat
        h5_file['camera_data'].attrs['height'] = sep_reader.height
        h5_file['camera_data'].attrs['width'] = sep_reader.width
        h5_file['camera_data'].attrs['numberOfFrames'] = sep_reader.numberOfFrames
        h5_file['camera_data'].attrs['numberOfChannels'] = sep_reader.numberOfChannels
        h5_file['camera_data'].attrs['cameraName'] = sep_reader.cameraName
        h5_file['camera_data'].attrs['recordDate'] = sep_reader.recordDate
        h5_file['camera_data'].attrs['droppedFrames'] = np.array(sep_reader.droppedFrames, dtype=np.int32)
        if sep_reader.timeStamps:
            h5_file['camera_data'].attrs['timeStamps'] = \
                np.array([x.strftime('%Y-%m-%d %H:%M:%S.%f') for x in sep_reader.timeStamps], dtype=object)
        else:
            h5_file['camera_data'].attrs['timeStamps'] = []

    # Save camera XML data.
    if xml_file_name is not None:
        xml_file_name = os.path.abspath(xml_file_name)
        AppendXMLToH5(h5_file_name, xml_file_name)
