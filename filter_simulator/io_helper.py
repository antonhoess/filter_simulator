from __future__ import annotations
from typing import List, Optional
from abc import ABC, abstractmethod
import os
import numpy as np
import pymap3d as pm

from .common import Frame, FrameList, Position, Detection


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


class FileWriter:
    def __init__(self, filename: str) -> None:
        self.filename: str = filename

    @staticmethod
    def get_files_in_directory(directory: str) -> List[str]:
        # List all files in a directory using scandir()
        files = []

        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_file():
                    files.append(entry.name)
            # end for
        # end while

        return files

    @staticmethod
    def get_next_sequence_filename(directory: str, filename_format: str, n_seq_max: int,
                                   fill_gaps: bool = False) -> Optional[str]:
        existing_files: List[str] = FileWriter.get_files_in_directory(directory)
        fn_ret = None

        if fill_gaps:
            for n in range(n_seq_max + 1):
                fn_tmp = filename_format.format(n)

                if not os.path.exists(fn_tmp):
                    fn_ret = fn_tmp
                    break
            # end for
        else:
            for n in reversed(range(n_seq_max + 1)):
                fn_tmp = filename_format.format(n)

                if fn_tmp in existing_files:
                    if n < n_seq_max:
                        fn_ret = filename_format.format(n + 1)

                    break
            # end for
        # end if

        return fn_ret

    def write(self, frames: FrameList, observer: Position):
        with open(self.filename, "w") as file:
            frame_nr: int = 0

            for frame in frames:
                frame_nr += 1

                for detection in frame:
                    lat, lon, _ = pm.enu2geodetic(np.array(detection.x), np.array(detection.y),
                                                  np.asarray(0), np.asarray(observer.x),
                                                  np.asarray(observer.y), np.asarray(0), ell=None, deg=True)
                    file.write("{} {} {}\n".format(lat, lon, frame_nr))
            # end for
        # end with
