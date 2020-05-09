from __future__ import annotations
from typing import Optional, List, Callable
from abc import ABC, abstractmethod
import time
import threading
import numpy as np
import pymap3d as pm
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.backend_bases
from enum import Enum
import re
import datetime

from .common import Logging, SimulationDirection, Limits, Position, Detection, FrameList
from .window_helper import WindowMode, LimitsMode, WindowModeChecker
from .io_helper import FileWriter
from .data_provider_interface import IDataProvider
from .data_provider_converter import CoordSysConv


class SimStepPart(Enum):
    DRAW = 0  # Draw the current scene
    WAIT_FOR_TRIGGER = 1  # Wait for user input to continue with the next step
    LOAD_NEXT_FRAME = 2  # Load the next data (frame)
    USER = 3  # Call user defined function
# end class


class SimStepPartConf:
    def __init__(self):
        self.__step_parts: List[SimStepPart] = []
        self.__user_step_parts: List[Callable] = []
        self.__user_step_part_cur_idx = 0
    # end def

    @property
    def step_parts(self):
        return self.__step_parts

    def add_draw_step(self):
        self.__step_parts.append(SimStepPart.DRAW)
    # end def

    def add_load_next_frame_step(self):
        self.__step_parts.append(SimStepPart.LOAD_NEXT_FRAME)
    # end def

    def add_wait_for_trigger_step(self):
        self.__step_parts.append(SimStepPart.WAIT_FOR_TRIGGER)
    # end def

    def add_user_step(self, user_func: Callable):
        self.__step_parts.append(SimStepPart.USER)
        self.__user_step_parts.append(user_func)
    # end def

    def get_next_user_step(self) -> Callable:
        user_func = self.__user_step_parts[self.__user_step_part_cur_idx]

        if self.__user_step_part_cur_idx < len(self.__user_step_parts) - 1:
            self.__user_step_part_cur_idx += 1
        else:
            self.__user_step_part_cur_idx = 0
        # end if

        return user_func
    # end def
# end class


class FilterSimulator(ABC):
    def __init__(self, data_provider: IDataProvider, output_coord_system_conversion: CoordSysConv, fn_out: str, fn_out_video: Optional[str], auto_step_interval: int,
                 limits: Limits, limits_mode: LimitsMode, observer: Optional[Position], logging: Logging) -> None:
        self.__data_provider = data_provider
        self.__output_coord_system_conversion: CoordSysConv = output_coord_system_conversion
        self.__fn_out: str = fn_out
        self.__fn_out_video: str = fn_out_video
        self.__frames: FrameList = FrameList()
        self.__simulation_direction: SimulationDirection = SimulationDirection.FORWARD
        self.__step: int = -1
        self.__next_part_step: bool = False
        self.__auto_step_interval = auto_step_interval
        self.__ax: Optional[matplotlib.axes.Axes] = None
        self.__fig: Optional[matplotlib.pyplot.figure] = None
        self.__logging: Logging = logging
        self.__observer: Position = observer if observer is not None else Position(0, 0)
        self.__observer_is_set = (observer is not None)
        self.__window_mode_checker: WindowModeChecker = WindowModeChecker(default_window_mode=WindowMode.SIMULATION, logging=logging)
        self.__manual_frames: FrameList = FrameList()
        self.__refresh: threading.Event = threading.Event()
        self.__refresh_finished: threading.Event = threading.Event()
        self.__limits_manual: Limits = limits
        self.__det_borders: Limits = self.__limits_manual
        self.__limits_mode: LimitsMode = limits_mode
        self.__limits_mode_inited: bool = False
        self.__prev_lim: Limits = Limits(0, 0, 0, 0)

        self.__sim_step_part_conf = self._set_sim_loop_step_part_conf()
        self.__fn_out_seq_max: int = 9999
        self.__fn_out_fill_gaps = False

        self.__anim = None
        self.__movie_writer = None
        self.__n_video_frames = 0
        self.__fn_out_video_gen: str = None  # For the generated name

    @property
    def fn_out_seq_max(self) -> int:
        return self.__fn_out_seq_max

    @fn_out_seq_max.setter
    def fn_out_seq_max(self, value: int) -> None:
        self.__fn_out_seq_max = value

    @property
    def fn_out_fill_gaps(self) -> bool:
        return self.__fn_out_fill_gaps

    @fn_out_fill_gaps.setter
    def fn_out_fill_gaps(self, value: bool) -> None:
        self.__fn_out_fill_gaps = value

    @property
    def _step(self) -> int:
        return self.__step

    @property
    def _next_part_step(self) -> bool:
        return self.__next_part_step

    @_next_part_step.setter
    def _next_part_step(self, value: bool) -> None:
        self.__next_part_step = value

    @property
    def _frames(self) -> FrameList:
        return self.__frames

    @property
    def _ax(self) -> Optional[matplotlib.axes.Axes]:
        return self.__ax

    @property
    def _fig(self) -> Optional[matplotlib.pyplot.figure]:
        return self.__fig

    @property
    def _det_borders(self) -> Limits:
        return self.__det_borders

    def __set_next_step(self) -> bool:
        if self.__simulation_direction == SimulationDirection.FORWARD:
            if self.__step < (len(self._frames) - 1):
                self.__step += 1

                return True
            # end if
        else:
            if self.__step > 0:
                self.__step -= 1

                return True
            # end if
        # end if

        return False
    # end def

    def run(self):
        # NB: Mark all threads as daemons so that the process terminates when the GUI thread termines.

        # Processing thread
        t_proc: threading.Thread = threading.Thread(target=self.__processing)
        t_proc.daemon = True
        t_proc.start()

        # Keyboard thread
        t_kbd: threading.Thread = threading.Thread(target=self.__cb_keyboard_wrap)
        t_kbd.daemon = True
        t_kbd.start()

        # Prepare GUI
        self.__fig: plt.Figure = plt.figure()
        self.__fig.canvas.set_window_title("State Space")
        self.__ax = self.__fig.add_subplot(1, 1, 1)

        # self.cid = fig.canvas.mpl_connect('button_press_event', self._cb_button_press_event)
        self.__fig.canvas.mpl_connect("button_press_event", self.__cb_button_press_event)
        self.__fig.canvas.mpl_connect("button_release_event", self.__cb_button_release_event)
        self.__fig.canvas.mpl_connect("key_press_event", self.__cb_key_press_event)
        self.__fig.canvas.mpl_connect("key_release_event", self.__cb_key_release_event)

        # Cyclic update check (but only draws, if there's something new)
        self.__anim: matplotlib.animation.Animation = animation.FuncAnimation(self.__fig, self.__update_window_wrap, interval=100)

        self.__setup_video()

        # Show blocking window which draws the current state and handles mouse clicks
        plt.show()

        # Store video to disk using the grabbed frames
        self.__write_video_to_file(self.__fn_out_video_gen)
    # end def

    def __cb_button_press_event(self, event: matplotlib.backend_bases.MouseEvent):
        self.__window_mode_checker.check_event(action="button_press_event", event=event)
        self.__handle_mpl_event(event)

    def __cb_button_release_event(self, event: matplotlib.backend_bases.MouseEvent):
        self.__window_mode_checker.check_event(action="button_release_event", event=event)

    def __cb_key_press_event(self, event: matplotlib.backend_bases.KeyEvent):
        self.__window_mode_checker.check_event(action="key_press_event", event=event)

    def __cb_key_release_event(self, event: matplotlib.backend_bases.KeyEvent):
        self.__window_mode_checker.check_event(action="key_release_event", event=event)

        fields = event.key.split("+")

        if len(fields) == 2 and ("ctrl" in fields or "control" in fields) and "z" in fields:
            borders = self._frames.calc_limits()
            self._ax.set_xlim(borders.x_min, borders.x_max)
            self._ax.set_ylim(borders.y_min, borders.y_max)
        # end if
    # end def

    def __handle_mpl_event(self, event: matplotlib.backend_bases.MouseEvent):
        # xxx mit überlegen, in welchem modus ich die punkte (also mit mehrerer detektionen pro frame) durch klicken
        # erstellen will - vllt. auch zweilei modi. die geklickten punkte entsprechend eingefärbt darstellen und
        # abspeichern. auch dies erlauben, wenn keine daten geladen wurden, durch angabe von einem default fenster,
        # das ja dann noch gezoomt und verschoben werden kann, um die stelle zu finden, die man möchte. es bräuchte
        # hierzu natürlich auch noch eine observer gps position:
        # man muss auch etwas überspringen können und tracks beenden können
        # hauptfrage: erst track 1, dann track 2, etc. oder erst alle detektionen in frame 1, dann in frame 2, etc.
        # track-weise scheint erst mal unlogisch, da die je erst später erstellt werden, oder doch nicht? es wäre jedoch
        # einfach zu klicken, aber es besteht auch die gefahr, dass die zeiten der verschiedenen tracks auseinander
        # laufen, wenn ich beim einen viel mehr klicks mache, als beim anderen und diese am ende wieder
        # zusammenführen...

        if self.__window_mode_checker.get_current_mode() == WindowMode.SIMULATION:
            # Right mouse button: Navigate forwards / backwards
            #   * Ctrl: Forwards
            #   * Shift: Backwards
            if event.button == 3:  # Right click
                self.__logging.print_verbose(Logging.DEBUG, "Right click")

                if event.key == "control":
                    self.__simulation_direction = SimulationDirection.FORWARD
                    self.__next_part_step = True

                elif event.key == "shift":
                    pass
                    # XXX makes no sense: self.simulation_direction = SimulationDirection.BACKWARD
                    # self.next = True
                # end if
            # end if

        elif self.__window_mode_checker.get_current_mode() == WindowMode.MANUAL_EDITING:
            # Left mouse button: Add
            #   * Ctrl: Points
            #   * Shift: Frame / Track
            # Right mouse button: Remove
            #   * Ctrl: Remove Points
            #   * Shift: Remove Frame / Track
            if event.button == 1:  # Left click
                if event.key == "control":
                    e: float = event.xdata
                    n: float = event.ydata

                    if self.__observer is not None:
                        lat, lon, _ = pm.enu2geodetic(np.array(e), np.array(n), np.asarray(0),
                                                      np.asarray(self.__observer.x), np.asarray(self.__observer.y),
                                                      np.asarray(0), ell=None, deg=True)
                        # print("{} {} {}".format(lat, lon, len(self.manual_points)))

                    # Add initial frame
                    if len(self.__manual_frames) == 0:
                        self.__manual_frames.add_empty_frame()

                    self.__manual_frames.get_current_frame().add_detection(Detection(event.xdata, event.ydata))
                    self.__logging.print_verbose(Logging.INFO, "Add point {:4f}, {:4f} to frame # {}".
                                                 format(event.xdata, event.ydata, len(self.__manual_frames)))

                elif event.key == "shift":
                    self.__manual_frames.add_empty_frame()
                    self.__logging.print_verbose(Logging.INFO, "Add new track (# {})".format(len(self.__manual_frames)))
                # end if

            elif event.button == 3:  # Right click
                if event.key == "control":
                    if self.__manual_frames.get_current_frame() is not None:
                        self.__manual_frames.get_current_frame().del_last_detection()

                elif event.key == "shift":
                    self.__manual_frames.del_last_frame()
                    # end if
                # end if event.key == ...
            # end if event.button == ...

            self.__refresh.set()
        # end if WindowMode.MANUAL_EDITING:

        # Independent on which mode a mouse button was clicked
        # Right mouse button:
        #   * Ctrl+Shift: Store detections
        #   * Ctrl+Alt+Shift: Store video
        if event.button == 3:  # Right click
            if WindowModeChecker.key_is_ctrl_shift(event.key):
                self.__write_points_to_file(self.__frames, self.__output_coord_system_conversion)

            elif WindowModeChecker.key_is_ctrl_alt_shift(event.key):
                self.__write_video_to_file(self.__fn_out_video_gen)

        # end if
    # end def

    def __write_points_to_file(self, frames: FrameList, output_coord_system_conversion: CoordSysConv = CoordSysConv.NONE):
        len_n_seq_max = len(str(self.__fn_out_seq_max))
        filename_format = f"{self.__fn_out}_{{:0{len_n_seq_max}d}}"
        filename_search_format = f"^{re.escape(self.__fn_out)}_(\d{{{len_n_seq_max}}})$"

        fn_out = FileWriter.get_next_sequence_filename(".", filename_format=filename_format, filename_search_format=filename_search_format,
                                                       n_seq_max=self.__fn_out_seq_max, fill_gaps=self.__fn_out_fill_gaps)

        if fn_out is not None:
            self.__logging.print_verbose(Logging.INFO,
                                         "Write points ({} frames with {} detections) to file {}".
                                         format(len(frames),
                                                frames.get_number_of_detections(), fn_out))

            file_writer = FileWriter(fn_out)
            file_writer.write(frames, self.__observer, output_coord_system_conversion)
        else:
            self.__logging.print_verbose(Logging.ERROR, "Could not write a new file based on the filename "
                                                        "{} and a max. sequence number of {}. Try to "
                                                        "remove some old files.".format(self.__fn_out, self.__fn_out_seq_max))
        # end if
    # end def

    def __setup_video(self):
        if self.__fn_out_video:
            self.__fn_out_video_gen = self.__get_video_filename()

            if self.__fn_out_video_gen:
                self.__logging.print_verbose(Logging.INFO, f"Setup new movie writer to file {self.__fn_out_video_gen}")
                self.__movie_writer = matplotlib.animation.FFMpegWriter(codec="h264", fps=1, extra_args=["-r", "25", "-f", "mp4"])  # Set output frame rate with using the extra_args
                self.__movie_writer.setup(fig=self.__fig, outfile=self.__fn_out_video_gen, dpi=200)
            # end if
        # end if
    # end def

    def __get_video_filename(self) -> str:
        len_n_seq_max = len(str(self.__fn_out_seq_max))
        filename_format = f"{self.__fn_out_video}_{{:0{len_n_seq_max}d}}"
        filename_search_format = f"^{re.escape(self.__fn_out_video)}_(\d{{{len_n_seq_max}}})$"

        fn_out = FileWriter.get_next_sequence_filename(".", filename_format=filename_format, filename_search_format=filename_search_format,
                                                       n_seq_max=self.__fn_out_seq_max, fill_gaps=self.__fn_out_fill_gaps)

        return fn_out
    # end def

    def __grab_video_frame(self):
        if self.__movie_writer:
            self.__movie_writer.grab_frame()
            self.__n_video_frames += 1
        # end if
    # end def

    def __write_video_to_file(self, fn_out_video: str):
        if self.__movie_writer:
            if fn_out_video:
                self.__logging.print_verbose(Logging.INFO, "Write video ({} frames) to file {}". format(self.__n_video_frames, fn_out_video))
                self.__movie_writer.finish()

                # Setup for a new video
                self.__setup_video()
                self.__grab_video_frame()  # Take the first shot now, since we'll loose it otherwise, as grabbing a frame gets triggert after moving from the current to the next step
            else:
                self.__logging.print_verbose(Logging.ERROR, "Could not write a new file based on the filename {} and a max. sequence number of {}. Try to "
                                                            "remove some old files.".format(self.__fn_out_video, self.__fn_out_seq_max))
            # end if
        # end if
    # end def

    def __update_window_limits(self) -> None:
        set_det_borders: bool = False
        set_prev_limits: bool = False
        set_manual_limits: bool = False

        if self.__limits_mode == LimitsMode.ALL_DETECTIONS_INIT_ONLY:
            if not self.__limits_mode_inited:
                set_det_borders = True
                self.__limits_mode_inited = True
            else:
                set_prev_limits = True
            # end if

        elif self.__limits_mode == LimitsMode.ALL_DETECTIONS_FIXED_UPDATE:
            set_det_borders = True

        elif self.__limits_mode == LimitsMode.ALL_CANVAS_ELEMENTS_DYN_UPDATE:
            pass

        elif self.__limits_mode == LimitsMode.MANUAL_AREA_INIT_ONLY:
            if not self.__limits_mode_inited:
                set_manual_limits = True
                self.__limits_mode_inited = True
            else:
                set_prev_limits = True
            # end if

        elif self.__limits_mode == LimitsMode.MANUAL_AREA_FIXED_UPDATE:
            set_manual_limits = True
        # end if

        if set_det_borders:
            self._ax.set_xlim(self._det_borders.x_min, self._det_borders.x_max)
            self._ax.set_ylim(self._det_borders.y_min, self._det_borders.y_max)

        elif set_prev_limits:
            self._ax.set_xlim(self.__prev_lim.x_min, self.__prev_lim.x_max)
            self._ax.set_ylim(self.__prev_lim.y_min, self.__prev_lim.y_max)

        elif set_manual_limits:
            self._ax.set_xlim(self.__limits_manual.x_min, self.__limits_manual.x_max)
            self._ax.set_ylim(self.__limits_manual.y_min, self.__limits_manual.y_max)
        # end if
    # end def

    def __processing(self) -> None:
        def time_diff_ms(dt_a: datetime.datetime, dt_b):
            dt_c = dt_b - dt_a

            return dt_c.seconds * 1000 + dt_c.microseconds / 1000

        # end def

        last_auto_step_time = datetime.datetime.now()
        self.__frames = self.__data_provider.frame_list

        if len(self._frames) == 0:
            return

        if not self.__observer_is_set:
            self.__observer = self._frames.calc_center()

        # Get the borders around the points for creating new particles later on
        self.__det_borders = self._frames.calc_limits()

        # Simulation loop
        while True:
            for sp in self.__sim_step_part_conf.step_parts:
                if sp == SimStepPart.DRAW:
                    # Draw - and wait until drawing has finished (do avoid changing the filter's state before it is drawn)
                    self.__refresh.set()
                    self.__refresh_finished.wait()
                    self.__refresh_finished.clear()

                elif sp == SimStepPart.WAIT_FOR_TRIGGER:
                    if self.__auto_step_interval is not None and self.__auto_step_interval >= 0:
                        while time_diff_ms(last_auto_step_time, datetime.datetime.now()) < self.__auto_step_interval:
                            time.sleep(0.1)
                        # end while

                        last_auto_step_time = datetime.datetime.now()
                        continue
                    else:
                        # Wait for Return-Key-Press (console) or mouse click (GUI)
                        while not self.__next_part_step:
                            time.sleep(0.1)

                        self.__next_part_step = False
                    # end if

                elif sp == SimStepPart.LOAD_NEXT_FRAME:
                    # Only continue when the next requested step is valid, e.g. it is within its boundaries
                    err_shown = False

                    while not self.__set_next_step():
                        if not err_shown:
                            err_shown = True
                            self.__logging.print_verbose(Logging.WARNING, "There are no more data frames to load.")
                        # end if

                        time.sleep(0.1)
                    # end while

                    self.__logging.print_verbose(Logging.INFO, "Step {}".format(self.__step))

                else:  # sp == SimulationStepPart.USER
                    self.__sim_step_part_conf.get_next_user_step()()
                # end if
            # end for
        # end while
    # end def

    def __update_window_wrap(self, _frame: Optional[int] = None) -> None:
        # This should block all subsequent calls to update_windows, but should be no problem
        timed_out = not self.__refresh.wait(50. / 1000)

        self.__refresh.clear()

        if self._ax is None or timed_out:
            return

        # Store current limits for resetting it the next time
        # (after drawing the elements, which might unwantedly change the limits)
        self.__prev_lim.x_min, self.__prev_lim.x_max = self._ax.get_xlim()
        self.__prev_lim.y_min, self.__prev_lim.y_max = self._ax.get_ylim()

        self._ax.clear()

        # Filter dependent function
        self._update_window()

        # Manually set points
        for frame in self.__manual_frames:
            self._ax.scatter([det.x for det in frame], [det.y for det in frame], s=20, marker="x")
            self._ax.plot([det.x for det in frame], [det.y for det in frame], color="black", linewidth=.5,
                          linestyle="--")
        # end for

        # Visualization settings (need to be set every time since they aren't permanent)
        self.__update_window_limits()

        self._ax.set_aspect('equal', 'datalim')
        self._ax.grid(False)

        self._ax.set_xlabel('east [m]')
        self._ax.set_ylabel('north [m]')

        # Grab current frame for creating the output video
        self.__grab_video_frame()

        self.__refresh_finished.set()
    # end def

    def __cb_keyboard_wrap(self) -> None:
        while True:
            cmd: str = input()

            try:
                self._cb_keyboard(cmd)

            except Exception as e:
                self.__logging.print_verbose(Logging.ERROR, "Invalid command. Exception: {}".format(e))
            # end try
        # end while
    # end def

    @abstractmethod
    def _set_sim_loop_step_part_conf(self) -> SimStepPartConf:
        pass

    @abstractmethod
    def _update_window(self) -> None:
        pass

    def _cb_keyboard(self, cmd: str) -> None:  # Can be overloaded
        if cmd == "":
            self._next_part_step = True

        elif cmd == "+":
            pass  # XXX

        elif cmd.startswith("-"):
            pass
            # XXX idx: int = int(cmd[1:])
        # end if
    # end def
# end class
