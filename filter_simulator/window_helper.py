from enum import IntEnum

from .common import Logging


class WindowMode(IntEnum):
    SIMULATION = 0
    MANUAL_EDITING = 1
# end class


class LimitsMode(IntEnum):
    # Sets the limits to the outer border of the set of all detections only at the beginning and keeps the limits during the program lifecycle at whatever they were changed to
    ALL_DETECTIONS_INIT_ONLY = 0

    # Sets the limits to the outer border of the set of all detections for each refresh - if the limits were manually changed, they'll get reset at the next simulation step
    ALL_DETECTIONS_FIXED_UPDATE = 1

    # Lets matplotlib decide an its own, how to set the limits (e.g. we do nothing) - this should be so that all drawing elements are within the limits
    # and will adapt dynamically as the drawing elements change
    ALL_CANVAS_ELEMENTS_DYN_UPDATE = 2

    # Sets the limits to the fov only at the beginning and keeps the limits during the program lifecycle at whatever they were changed to
    FOV_INIT_ONLY = 3

    # Sets the limits to the fov for each refresh - if the limits were manually changed, they'll get reset at the next simulation step
    FOV_FIXED_UPDATE = 4
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
        if key is not None and "shift" in key and "alt" not in key and ("control" in key or "ctrl" in key):
            return True
        else:
            return False
    # end def

    @staticmethod
    def key_is_ctrl_alt_shift(key) -> bool:
        if key is not None and "shift" in key and "alt" in key and ("control" in key or "ctrl" in key):
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
