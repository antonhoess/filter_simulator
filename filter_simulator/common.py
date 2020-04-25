from __future__ import annotations
from enum import IntEnum
from typing import List, Callable, Optional
import pymap3d as pm
import numpy as np


class Logging(IntEnum):
    NONE = 0
    CRITICAL = 1
    ERROR = 2
    WARNING = 3
    INFO = 4
    DEBUG = 5

    def __init__(self, verbosity: int):
        super().__init__()

        self.verbosity: int = verbosity

    @staticmethod
    def print(message: str):
        print(message)

    def print_verbose(self, verbosity: Logging, message: str):
        if self.verbosity >= verbosity:
            print(message)
# end class


class SimulationDirection(IntEnum):
    FORWARD = 0
    BACKWARD = 1
# end class


class Limits:
    def __init__(self, x_min: float = .0, y_min: float = .0, x_max: float = .0, y_max: float = .0):
        self.x_min: float = x_min
        self.y_min: float = y_min
        self.x_max: float = x_max
        self.y_max: float = y_max
    # end def
# end class


class Position:
    def __init__(self, x: float = .0, y: float = .0) -> None:
        self.x: float = x
        self.y: float = y

    def __str__(self) -> str:
        return "x={}, y={}".format(self.x, self.y)


class Detection(Position):
    def __init__(self, x: float, y: float) -> None:
        super().__init__(x, y)

    def __str__(self) -> str:
        return "x={}, y={}".format(self.x, self.y)


class Frame:
    def __init__(self) -> None:
        self._detections: List[Detection] = []

    def __iter__(self) -> FrameIterator:
        return FrameIterator(self)

    def __getitem__(self, index: int) -> Detection:
        return self._detections[index]

    def __len__(self) -> int:
        return len(self._detections)

    def __str__(self) -> str:
        return "Frame with {} detections".format(len(self))

    def add_detection(self, detection: Detection) -> None:
        self._detections.append(detection)

    def del_last_detection(self) -> None:
        if len(self._detections) > 0:
            del self._detections[-1]
    # end def

    def del_all_detections(self) -> None:
        for _ in reversed(range(len(self))):  # reverse() maybe might make sense at a later point
            del self._detections[-1]
    # end def

    def get_detections(self) -> List[Detection]:
        return self._detections
# end class


class FrameIterator:
    def __init__(self, frame: Frame) -> None:
        self._frame: Frame = frame
        self._index: int = 0

    def __next__(self) -> Detection:
        if self._index < len(self._frame.get_detections()):
            result: Detection = self._frame[self._index]
            self._index += 1

            return result

        # End of iteration
        raise StopIteration
# end class


class FrameListIterator:
    def __init__(self, frame_list) -> None:
        self._frame_list = frame_list
        self._index: int = 0

    def __next__(self) -> Frame:
        if self._index < len(self._frame_list.get_frames()):
            result: Frame = self._frame_list[self._index]
            self._index += 1

            return result

        # End of iteration
        raise StopIteration
# end class


class FrameList:
    def __init__(self) -> None:
        self._frames: List[Frame] = []

    def __iter__(self) -> FrameListIterator:
        return FrameListIterator(self)

    def __getitem__(self, index) -> Frame:
        return self._frames[index]

    def __len__(self) -> int:
        return len(self._frames)

    def __str__(self) -> str:
        return "FrameList with {} frames and {} detections total".format(len(self), self.get_number_of_detections())

    def add_empty_frame(self) -> None:
        self._frames.append(Frame())

    def del_last_frame(self) -> bool:
        if len(self._frames) > 0:
            self.get_current_frame().del_all_detections()
            del self._frames[-1]
            return True
        else:
            return False
        # end if

    def get_frames(self) -> List[Frame]:
        return self._frames

    def get_current_frame(self) -> Optional[Frame]:
        if len(self._frames) > 0:
            return self._frames[-1]
        else:
            return None

    def get_number_of_detections(self) -> int:
        n: int = 0

        for frame in self._frames:
            n += len(frame)
        # end for

        return n

    def foreach_detection(self, cb_detection: Callable, **kwargs) -> None:
        for frame in self._frames:
            for detection in frame:
                cb_detection(detection, **kwargs)
        # end for

    @staticmethod
    def __update_limit_by_detection(detection: Detection, limits: Limits) -> None:
        if limits.x_min is None or detection.x < limits.x_min:
            limits.x_min = detection.x

        if limits.y_min is None or detection.y < limits.y_min:
            limits.y_min = detection.y

        if limits.x_max is None or detection.x > limits.x_max:
            limits.x_max = detection.x

        if limits.y_max is None or detection.y > limits.y_max:
            limits.y_max = detection.y
    # end def

    def calc_limits(self) -> Limits:
        limits: Limits = Limits()

        self.foreach_detection(self.__update_limit_by_detection, limits=limits)

        return limits

    def calc_center(self) -> Position:
        limits = self.calc_limits()

        return Position(x=(limits.x_min + limits.x_max) / 2, y=(limits.y_min + limits.y_max) / 2)
    # end def
# end class


class WGS84ToENUConverter:
    @staticmethod
    def convert(frame_list_wgs84: FrameList, observer: Optional[Position]) -> FrameList:
        frame_list_enu: FrameList = FrameList()

        if observer is None:
            observer = frame_list_wgs84.calc_center()

        for frame in frame_list_wgs84:
            frame_list_enu.add_empty_frame()

            for detection in frame:
                # Convert...
                e, n, _ = pm.geodetic2enu(np.asarray(detection.x), np.asarray(detection.y), np.asarray(0),
                                          np.asarray(observer.x), np.asarray(observer.y), np.asarray(0),
                                          ell=None, deg=True)
                frame_list_enu.get_current_frame().add_detection(Detection(float(e), float(n)))
            # end for
        # end for

        return frame_list_enu
    # end def
# end class
