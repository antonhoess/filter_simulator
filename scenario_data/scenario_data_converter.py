from __future__ import annotations
from typing import Optional
import numpy as np
import pymap3d as pm
from enum import Enum
from abc import ABC, abstractmethod
import copy

from filter_simulator.common import FrameList, Position
from scenario_data.scenario_data import ScenarioData, GroundTruthTrack


class CoordSysConv(Enum):
    ENU = "ENU"
    WGS84 = "WGS84"
# end class


class CoordSysConverter(ABC):
    coordinate_system = None

    @classmethod
    def convert(cls, data: ScenarioData, observer: Optional[Position], in_place=True) -> ScenarioData:

        if not in_place:
            data = copy.deepcopy(data)
        # end if

        data.meta.coordinate_system = cls.coordinate_system.value

        if observer is None:
            observer = data.ds.calc_center()
        # end if

        if data.ds is not None:
            cls.convert_frame(data.ds, observer)

        if data.fas is not None:
            cls.convert_frame(data.fas, observer)

        if data.mds is not None:
            cls.convert_frame(data.mds, observer)

        if data.gtts is not None:
            for track in data.gtts:
                cls.convert_track(track, observer)
            # end for
        # end if

        return data
    # end def

    @classmethod
    @abstractmethod
    def convert_position(cls, position: Position, observer: Position) -> Position:
        raise NotImplementedError
    # end def

    @classmethod
    def convert_frame(cls, frame: FrameList, observer: Optional[Position]) -> FrameList:
        if observer is None:
            observer = frame.calc_center()

        for frame in frame:
            for detection in frame:
                # Convert...
                cls.convert_position(detection, observer)
            # end for
        # end for

        return frame
    # end def

    @classmethod
    def convert_track(cls, track: GroundTruthTrack, observer: Position) -> GroundTruthTrack:

        for point in track.points:
            cls.convert_position(point, observer)
        # end for

        return track
    # end def
# end class


class Wgs84ToEnuConverter(CoordSysConverter):
    coordinate_system = CoordSysConv.ENU

    @classmethod
    def convert_position(cls, position: Position, observer: Position) -> Position:
        # Convert...
        e, n, _ = pm.geodetic2enu(np.asarray(position.x), np.asarray(position.y), np.asarray(0),
                                  np.asarray(observer.x), np.asarray(observer.y), np.asarray(0),
                                  ell=None, deg=True)
        position.x = float(e)
        position.y = float(n)

        return position
    # end def
# end class


class EnuToWgs84Converter(CoordSysConverter):
    coordinate_system = CoordSysConv.WGS84

    @classmethod
    def convert_position(cls, position: Position, observer: Position) -> Position:
        # Convert...
        lat, lon, _ = pm.enu2geodetic(np.asarray(position.x), np.asarray(position.y), np.asarray(0),
                                      np.asarray(observer.x), np.asarray(observer.y), np.asarray(0),
                                      ell=None, deg=True)
        position.x = float(lat)
        position.y = float(lon)

        return position
    # end def
# end class
