from __future__ import annotations
from typing import Optional
import numpy as np
import pymap3d as pm
from enum import Enum


from simulation_data.data_provider_interface import IDataProvider
from filter_simulator.common import FrameList, Position


class CoordSysConv(Enum):
    NONE = 0
    WGS84 = 1
# end class


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
                frame_list_enu.get_current_frame().add_detection(Position(float(e), float(n)))
            # end for
        # end for

        return frame_list_enu
    # end def
# end class


class EnuToWgs84Converter(IDataProvider):  # XXX diese klasse in io_helper einsetzen und den anderen code hierdurch ersetzen
    def __init__(self, frame_list_enu: FrameList, observer: Optional[Position]):
        self.__frame_list_enu = frame_list_enu
        self.__observer = observer

    # end def

    @property
    def frame_list(self) -> FrameList:
        frame_list_wgs84: FrameList = FrameList()

        if self.__observer is None:
            self.__observer = self.__frame_list_enu.calc_center()

        for frame in self.__frame_list_enu:
            frame_list_wgs84.add_empty_frame()

            for detection in frame:
                # Convert...
                lat, lon, _ = pm.enu2geodetic(np.asarray(detection.x), np.asarray(detection.y), np.asarray(0),
                                              np.asarray(self.__observer.x), np.asarray(self.__observer.y), np.asarray(0),
                                              ell=None, deg=True)
                frame_list_wgs84.get_current_frame().add_detection(Position(float(lat), float(lon)))
            # end for
        # end for

        return frame_list_wgs84
    # end def
# end class
