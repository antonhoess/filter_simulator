from enum import IntEnum

from .common import Logging


class WindowMode(IntEnum):
    SIMULATION = 0
    MANUAL_EDITING = 1
# end class


class LimitsMode(IntEnum):
    ALL_DETECTIONS_INIT_ONLY = 0
    ALL_DETECTIONS_FIXED_UPDATE = 1
    ALL_CANVAS_ELEMENTS_DYN_UPDATE = 2
    MANUAL_AREA_INIT_ONLY = 3
    MANUAL_AREA_FIXED_UPDATE = 4
# end class


class WindowModeChecker:
    def __init__(self, default_window_mode: WindowMode, logging: Logging = Logging.NONE) -> None:
        self.__window_mode: WindowMode = default_window_mode
        self.__control_shift_pressed: bool = False
        self.__control_shift_left_click_cnt: int = 0
        self.__logging = logging

    def get_current_mode(self) -> WindowMode:
        return self.__window_mode

    @staticmethod
    def key_is_ctrl_shift(key) -> bool:
        # shift+control or ctrl+shift
        if "shift" in key and ("control" in key or "ctrl" in key):
            return True
        else:
            return False
    # end def

    def check_event(self, action, event) -> None:
        if action == "button_press_event":
            if event.button == 1:  # Left click
                if self.__control_shift_pressed:
                    self.__control_shift_left_click_cnt += 1
                # end if
            # end if
        # end if

        if action == "button_release_event":
            pass

        if action == "key_press_event":
            if self.key_is_ctrl_shift(event.key):
                self.__control_shift_pressed = True
                self.__logging.print_verbose(Logging.DEBUG, "key_press_event: " + event.key)
            # end if
        # end if

        if action == "key_release_event":
            if self.key_is_ctrl_shift(event.key):
                self.__logging.print_verbose(Logging.DEBUG, "key_release_event: " + event.key)

                if self.__control_shift_left_click_cnt >= 3:
                    if self.__window_mode == WindowMode.SIMULATION:
                        self.__window_mode = WindowMode.MANUAL_EDITING
                    else:
                        self.__window_mode = WindowMode.SIMULATION
                    # end if

                    self.__logging.print_verbose(Logging.INFO, "Changed window mode to {}.".
                                                 format(WindowMode(self.__window_mode).name))
                # end if
                self.__control_shift_left_click_cnt = 0
                self.__control_shift_pressed = False
            # end if
        # end if
    # end def
# end class
