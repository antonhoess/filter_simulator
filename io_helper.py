from __future__ import annotations
from typing import List, Optional
from abc import ABC, abstractmethod
import os

from common import FrameList, Detection


class InputLineHandler(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def handle_line(self, line) -> None:
        pass


class InputLineHandlerLatLonIdx(InputLineHandler):
    def __init__(self) -> None:
        self.cur_idx: Optional[int] = None
        self.frame_list: FrameList = FrameList()
        super().__init__()

    def handle_line(self, line: str) -> None:
        lat: float = .0
        lon: float = .0
        idx: Optional[int] = 0

        # Split line and read fields
        fields: List[str] = line.split(" ")

        if len(fields) >= 2:
            lat = float(fields[0])
            lon = float(fields[1])
        # end if

        if len(fields) >= 3:
            idx = int(fields[2])
        # end if

        # Check if we need to add a new frame to the frame list
        if idx is None or self.cur_idx is None or idx != self.cur_idx:
            self.cur_idx = idx
            self.frame_list.add_empty_frame()
        # end if

        # Add detection from field values to the frame
        self.frame_list.get_current_frame().add_detection(Detection(lat, lon))

        return
    # end def
# end class


class FileReader:
    def __init__(self, filename: str) -> None:
        self.filename: str = filename

    def read(self, input_line_handler: InputLineHandler) -> None:
        try:
            with open(self.filename, 'r') as file:
                while True:
                    # Get next line from file
                    line: str = file.readline()

                    # if line is empty end of file is reached
                    if not line:
                        break
                    else:
                        input_line_handler.handle_line(line.rstrip(os.linesep))
                    # end if
                # end while
            # end with

        except IOError as e:
            print("Error opening file {}: {}".format(self.filename, e))
    # end def
# end class
