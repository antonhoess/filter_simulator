from __future__ import annotations
from typing import List, Optional
from abc import ABC, abstractmethod
import os
import re

from .common import FrameList, Position
from simulation_data.data_provider_interface import IDataProvider
from simulation_data.data_provider_converter import CoordSysConv, EnuToWgs84Converter
from simulation_data.sim_data import SimulationData, MetaInformation


class InputLineHandler(ABC):
    @abstractmethod
    def handle_line(self, line) -> None:
        pass


class InputLineHandlerLatLonIdx(InputLineHandler, IDataProvider):
    def __init__(self) -> None:
        self.__cur_idx: Optional[int] = None
        self.__frame_list: FrameList = FrameList()
        super().__init__()

    @property
    def sim_data(self) -> SimulationData:
        d = SimulationData()
        d.meta = MetaInformation()
        d.meta.version = "0.1"
        d.meta.number_steps = len(self.__frame_list)

        d.ds = self.__frame_list

        return d
    # end def

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
        if idx is None or self.__cur_idx is None or idx != self.__cur_idx:
            self.__cur_idx = idx
            self.__frame_list.add_empty_frame()
        # end if

        # Add detection from field values to the frame
        self.__frame_list.get_current_frame().add_detection(Position(lat, lon))

        return
    # end def
# end class


class FileReader:
    def __init__(self, filename: str) -> None:
        self.__filename: str = filename

    def read(self, input_line_handler: InputLineHandler) -> None:
        try:
            with open(self.__filename, 'r') as file:
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
            print("Error opening file {}: {}".format(self.__filename, e))
    # end def
# end class


class FileWriter:
    def __init__(self, filename: str) -> None:
        self.__filename: str = filename

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
    def get_next_sequence_filename(directory: str, filename_format: str, filename_search_format: Optional[str], n_seq_max: int,
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
            if filename_search_format is not None:  # Probably more efficient than the version below, especially for many posstible numbers
                max_val = -1

                pattern = re.compile(filename_search_format)

                for file in existing_files:
                    if pattern.match(file):
                        res = pattern.search(file)

                        if res is not None:
                            val = int(res.group(1))

                            if val > max_val:
                                max_val = val
                            # end if
                        # end if
                    # end if
                # end for file

                if max_val < n_seq_max:
                    fn_ret = filename_format.format(max_val + 1)
                # end if
            else:
                found = False
                for n in reversed(range(n_seq_max + 1)):
                    fn_tmp = filename_format.format(n)

                    if fn_tmp in existing_files:
                        found = True

                        if n < n_seq_max:
                            fn_ret = filename_format.format(n + 1)

                        break
                    # end if
                # end for

                if not found:
                    fn_ret = filename_format.format(0)
            # end if
        # end if

        return fn_ret

    def write(self, frames: FrameList, observer: Position, output_coord_system_conversion: CoordSysConv = CoordSysConv.NONE):
        if output_coord_system_conversion == CoordSysConv.NONE:
            pass
        elif output_coord_system_conversion == CoordSysConv.WGS84:
            frames = EnuToWgs84Converter(frames, observer).frame_list
        # end if

        with open(self.__filename, "w") as file:
            frame_nr: int = 0

            for frame in frames:
                frame_nr += 1

                for detection in frame:
                    file.write("{} {} {}\n".format(detection.x, detection.y, frame_nr))
            # end for
        # end with
    # end def
# end class
