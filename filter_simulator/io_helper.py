from __future__ import annotations
from typing import List, Optional,Tuple
import os
import re


class FileHelper:
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
    # end def

    @staticmethod
    def get_filename_formats(filename: str, replace_string: str, len_n_seq_max: int) -> Tuple[str, str]:
        filename_format = filename.replace(replace_string, f"{{0:0{len_n_seq_max}d}}", )
        filename_search_format = "^" + f"(\d{{{len_n_seq_max}}})".join([re.escape(part) for part in filename.split(replace_string)]) + "$"

        return filename_format, filename_search_format
    # end def
    @staticmethod
    def get_next_sequence_filename(directory: str, filename_format: str, filename_search_format: Optional[str], n_seq_max: int,
                                   fill_gaps: bool = False) -> Optional[str]:
        existing_files: List[str] = FileHelper.get_files_in_directory(directory)
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
# end class
