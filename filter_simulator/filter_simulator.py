from __future__ import annotations
from typing import Optional
from abc import ABC, abstractmethod
import os
import time
import threading
import numpy as np
import pymap3d as pm
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.backend_bases

from .common import Logging, SimulationDirection, Limits, Position, Detection, FrameList, WGS84ToENUConverter
from .window_helper import WindowMode, LimitsMode, WindowModeChecker
from .io_helper import FileReader, FileWriter, InputLineHandlerLatLonIdx


class FilterSimulator(ABC):
    def __init__(self, fn_in: str, fn_out: str, limits: Limits, observer: Optional[Position], logging: Logging) -> None:
        self.__fn_in: str = fn_in
        self.__fn_out: str = fn_out
        self.__step: int = -1
        self.__next: bool = False
        self.__frames: FrameList = FrameList()
        self.__simulation_direction: SimulationDirection = SimulationDirection.FORWARD
        self.__ax: Optional[matplotlib.axes.Axes] = None
        self.__logging: Logging = logging
        self.__observer: Position = observer if observer is not None else Position(0, 0)
        self.__observer_is_set = (observer is not None)
        self.__window_mode_checker: WindowModeChecker = WindowModeChecker(default_window_mode=WindowMode.SIMULATION,
                                                                          logging=logging)
        self.__manual_frames: FrameList = FrameList()
        self.__refresh: threading.Event = threading.Event()
        self.__refresh_finished: threading.Event = threading.Event()
        self.__limits_manual: Limits = limits
        self.__det_borders: Limits = self.__limits_manual
        self.__limits_mode: LimitsMode = LimitsMode.ALL_DETECTIONS_INIT_ONLY
        self.__limits_mode_inited: bool = False
        self.__prev_lim: Limits = Limits(0, 0, 0, 0)

    @property
    def _step(self) -> int:
        return self.__step

    @property
    def _next(self) -> bool:
        return self.__next

    @_next.setter
    def _next(self, value: bool) -> None:
        self.__next = value

    @property
    def _frames(self) -> FrameList:
        return self.__frames

    @property
    def _ax(self) -> Optional[matplotlib.axes.Axes]:
        return self.__ax

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

    def __wait_for_valid_next_step(self) -> None:
        while True:
            # Wait for Return-Key-Press (console) of mouse click (GUI)
            while not self.__next:
                time.sleep(0.1)

            self.__next = False

            # Only continue when the next requested step is valid, e.g. it is within its boundaries
            if self.__set_next_step():
                break
        # end while

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
        fig: plt.Figure = plt.figure()
        fig.canvas.set_window_title("State Space")
        self.__ax = fig.add_subplot(1, 1, 1)

        # self.cid = fig.canvas.mpl_connect('button_press_event', self._cb_button_press_event)
        fig.canvas.mpl_connect("button_press_event", self.__cb_button_press_event)
        fig.canvas.mpl_connect("button_release_event", self.__cb_button_release_event)
        fig.canvas.mpl_connect("key_press_event", self.__cb_key_press_event)
        fig.canvas.mpl_connect("key_release_event", self.__cb_key_release_event)

        # Cyclic update check (but only draws, if there's something new)
        _anim: matplotlib.animation.Animation = animation.FuncAnimation(fig, self.__update_window_wrap, interval=100)

        # Show blocking window which draws the current state and handles mouse clicks
        plt.show()
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
                    self.__next = True

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

                elif WindowModeChecker.key_is_ctrl_shift(event.key):
                    n_seq_max = 99
                    fn_out = FileWriter.get_next_sequence_filename(".", filename_format=(self.__fn_out + "_{:02d}"),
                                                                   n_seq_max=n_seq_max, fill_gaps=False)

                    if fn_out is not None:
                        self.__logging.print_verbose(Logging.INFO,
                                                     "Write manual points ({} frames with {} detections) to file {}".
                                                     format(len(self.__manual_frames),
                                                            self.__manual_frames.get_number_of_detections(), fn_out))

                        file_writer = FileWriter(fn_out)
                        file_writer.write(self.__manual_frames, self.__observer)
                    else:
                        self.__logging.print_verbose(Logging.ERROR, "Could not write a new file based on the filename"
                                                                    "{} and a max. sequence number of {}. Try to"
                                                                    "remove some old files.".format(self.__fn_out,
                                                                                                    n_seq_max))

            # end if event.button ...

            self.__refresh.set()
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
        if not os.path.isfile(self.__fn_in):
            return

        # Read all measurements from file
        file_reader: FileReader = FileReader(self.__fn_in)
        line_handler: InputLineHandlerLatLonIdx = InputLineHandlerLatLonIdx()
        file_reader.read(line_handler)
        self.__frames = line_handler.frame_list

        if len(self._frames) == 0:
            return

        if not self.__observer_is_set:
            self.__observer = self._frames.calc_center()

        # Convert from WGS84 to ENU, with its origin at the center of all points
        self.__frames = WGS84ToENUConverter.convert(frame_list_wgs84=self._frames, observer=self.__observer)

        # Get the borders around the points for creating new particles later on
        self.__det_borders = self._frames.calc_limits()

        # Simulation loop
        while True:
            # Filter dependent function
            self._sim_loop_before_step_and_drawing()

            # Draw
            self.__refresh.set()

            # Wait until drawing has finished (do avoid changing the filter's state before it is drawn)
            self.__refresh_finished.wait()
            self.__refresh_finished.clear()

            # Wait for a valid next step
            self.__wait_for_valid_next_step()
            self.__logging.print_verbose(Logging.INFO, "Step {}".format(self.__step))

            # Filter dependent function
            self._sim_loop_after_step_and_drawing()
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

        # Visualization settings (need to be set every time since they don't are permanent)
        self.__update_window_limits()

        self._ax.set_aspect('equal', 'datalim')
        self._ax.grid(False)

        self._ax.set_xlabel('east [m]')
        self._ax.set_ylabel('north [m]')

        self.__refresh_finished.set()
    # end def

    def __cb_keyboard_wrap(self) -> None:
        while True:
            cmd: str = input()

            try:
                self._cb_keyboard(cmd)

            except Exception as e:
                print("Invalid command. Exception: {}".format(e))
            # end try
        # end while
    # end def

    @abstractmethod
    def _sim_loop_before_step_and_drawing(self) -> None:
        pass

    @abstractmethod
    def _sim_loop_after_step_and_drawing(self) -> None:
        pass

    @abstractmethod
    def _update_window(self) -> None:
        pass

    @abstractmethod
    def _cb_keyboard(self, cmd: str) -> None:
        pass
# end class
