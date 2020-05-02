from __future__ import annotations
from typing import Optional
import numpy as np
import pymap3d as pm


from .data_provider_interface import IDataProvider
from .common import FrameList, Detection, Position


class Wgs84ToEnuConverter(IDataProvider):
    def __init__(self, frame_list_wgs84: FrameList, observer: Optional[Position]):
        self.__frame_list_wgs84 = frame_list_wgs84
        self.__observer = observer
    # end def

    @property
    def frame_list(self) -> FrameList:
        frame_list_enu: FrameList = FrameList()

        if self.__observer is None:
            self.__observer = self.__frame_list_wgs84.calc_center()

        for frame in self.__frame_list_wgs84:
            frame_list_enu.add_empty_frame()

            for detection in frame:
                # Convert...
                e, n, _ = pm.geodetic2enu(np.asarray(detection.x), np.asarray(detection.y), np.asarray(0),
                                          np.asarray(self.__observer.x), np.asarray(self.__observer.y), np.asarray(0),
                                          ell=None, deg=True)
                frame_list_enu.get_current_frame().add_detection(Detection(float(e), float(n)))
            # end for
        # end for

        return frame_list_enu
    # end def
# end class
