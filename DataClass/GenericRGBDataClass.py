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

import numpy as np
from CameraModel.Pleora.RGB.GenericRGBCamera import GenericRGBCamera
from hairoad_calib.DataClass.GenericDataClass import GenericDataClass


class GenericRGBDataClass(GenericDataClass, GenericRGBCamera):
    """Data class for Generic RGB cameras"""

    def __init__(self):
        super().__init__()

    def GetFrame(self, frame_idx=None, raw=False, **kwargs):
        # TODO: change Preprocess and clean this up
        """Get a (sequence of) frame(s).

        It will return the next frame in the sequence and increment the counter by 1 by default.

        :param frame_idx: None (default), or the index of the frame you want to retrieve, or a tuple indicating a range
            with the indices of the first and last frame you want to retrieve.
        :param raw: Whether to return the raw or preprocessed frame.
        :return: The specified frames in a numpy array (height, width, channels, frames), unused axes will be squeezed.
        :raise IndexError: When the provided indices are out of bounds or the counter has reached the final image.
        :raise TypeError: When an incorrect type is used for `frame_idx`.
        """
        # Get unprocessed frame
        frame = super().GetFrame(frame_idx)

        # Preprocess frames if raw is False
        if not raw:
            if len(frame.shape) == 2:
                frame = GenericRGBCamera.PreProcess(self, frame)
                return frame
            else:
                frames = []
                for i in range(frame.shape[2]):
                    frames.append(GenericRGBCamera.PreProcess(self, frame[:, :, i]))
                return np.array(frames)
        else:
            return frame
