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
from .io_helper import FileReader, InputLineHandlerLatLonIdx


class FilterSimulator(ABC):
    def __init__(self, fn_in: str, fn_out: str, verbosity: Logging, observer: Position, limits: Limits) -> None:
        self.fn_in: str = fn_in
        self.fn_out: str = fn_out
        self.step: int = -1
        self.next: bool = False
        self.frames: FrameList = FrameList()
        self.simulation_direction: SimulationDirection = SimulationDirection.FORWARD
        self.ax: Optional[matplotlib.axes.Axes] = None
        self.logging: Logging = verbosity
        self.observer: Position = observer
        self.window_mode_checker: WindowModeChecker = WindowModeChecker(default_window_mode=WindowMode.SIMULATION,
                                                                        verbosity=verbosity)
        self.manual_frames: FrameList = FrameList()
        self.refresh: threading.Event = threading.Event()
        self.refresh_finished: threading.Event = threading.Event()
        self.limits_manual: Limits = limits
        self.det_borders: Limits = self.limits_manual
        self.limits_mode: LimitsMode = LimitsMode.ALL_DETECTIONS_INIT_ONLY
        self.limits_mode_inited: bool = False
        self.prev_lim: Limits = Limits(0, 0, 0, 0)

    def set_next_step(self) -> bool:
        if self.simulation_direction == SimulationDirection.FORWARD:
            if self.step < (len(self.frames) - 1):
                self.step += 1
                return True
            # end if
        else:
            if self.step > 0:
                self.step -= 1
                return True
            # end if
        # end if

        return False

    def wait_for_valid_next_step(self) -> None:
        while True:
            # Wait for Return-Key-Press (console) of mouse click (GUI)
            while not self.next:
                time.sleep(0.1)

            self.next = False

            # Only continue when the next requested step is valid, e.g. it is within its boundaries
            if self.set_next_step():
                break
        # end while

    def run(self):
        # Processing thread
        t_proc: threading.Thread = threading.Thread(target=self._processing)
        t_proc.start()

        # Keyboard thread
        t_kbd: threading.Thread = threading.Thread(target=self._cb_keyboard)
        t_kbd.daemon = True
        t_kbd.start()

        # Prepare GUI
        fig: plt.Figure = plt.figure()
        fig.canvas.set_window_title("State Space")
        self.ax = fig.add_subplot(1, 1, 1)

        # self.cid = fig.canvas.mpl_connect('button_press_event', self._cb_button_press_event)
        fig.canvas.mpl_connect("button_press_event", self._cb_button_press_event)
        fig.canvas.mpl_connect("button_release_event", self._cb_button_release_event)
        fig.canvas.mpl_connect("key_press_event", self._cb_key_press_event)
        fig.canvas.mpl_connect("key_release_event", self._cb_key_release_event)

        # Cyclic update check (but only draws, if there's something new)
        _anim: matplotlib.animation.Animation = animation.FuncAnimation(fig, self._update_window, interval=100)

        # Show blocking window which draws the current state and handles mouse clicks
        plt.show()
    # end def

    def _cb_button_press_event(self, event: matplotlib.backend_bases.MouseEvent):
        self.window_mode_checker.check_event(action="button_press_event", event=event)
        self.handle_mpl_event(event)

    def _cb_button_release_event(self, event: matplotlib.backend_bases.MouseEvent):
        self.window_mode_checker.check_event(action="button_release_event", event=event)

    def _cb_key_press_event(self, event: matplotlib.backend_bases.KeyEvent):
        self.window_mode_checker.check_event(action="key_press_event", event=event)

    def _cb_key_release_event(self, event: matplotlib.backend_bases.KeyEvent):
        self.window_mode_checker.check_event(action="key_release_event", event=event)

    def handle_mpl_event(self, event: matplotlib.backend_bases.MouseEvent):
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

        if self.window_mode_checker.get_current_mode() == WindowMode.SIMULATION:
            # Right mouse button: Navigate forwards / backwards
            #   * Ctrl: Forwards
            #   * Shift: Backwards
            if event.button == 3:  # Right click
                self.logging.print_verbose(Logging.DEBUG, "Right click")

                if event.key == "control":
                    self.simulation_direction = SimulationDirection.FORWARD
                    self.next = True

                elif event.key == "shift":
                    pass
                    # XXX makes no sense: self.simulation_direction = SimulationDirection.BACKWARD
                    # self.next = True
                # end if
            # end if

        elif self.window_mode_checker.get_current_mode() == WindowMode.MANUAL_EDITING:
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
                    lat, lon, _ = pm.enu2geodetic(np.array(e), np.array(n), np.asarray(0), np.asarray(self.observer.x),
                                                  np.asarray(self.observer.y), np.asarray(0), ell=None, deg=True)

                    # print("{} {} {}".format(lat, lon, len(self.manual_points)))
                    # Add initial frame
                    if len(self.manual_frames) == 0:
                        self.manual_frames.add_empty_frame()

                    self.manual_frames.get_current_frame().add_detection(Detection(event.xdata, event.ydata))
                    self.logging.print_verbose(Logging.INFO, "Add point {:4f}, {:4f} to frame # {}".
                                               format(event.xdata, event.ydata, len(self.manual_frames)))

                elif event.key == "shift":
                    self.manual_frames.add_empty_frame()
                    self.logging.print_verbose(Logging.INFO, "Add new track (# {})".format(len(self.manual_frames)))
                # end if

            elif event.button == 3:  # Right click
                if event.key == "control":
                    if self.manual_frames.get_current_frame() is not None:
                        self.manual_frames.get_current_frame().del_last_detection()

                elif event.key == "shift":
                    self.manual_frames.del_last_frame()

                elif WindowModeChecker.key_is_ctrl_shift(event.key):
                    fn_out: str = self.fn_out

                    for i in range(100):
                        fn_out = "{}_{:02d}".format(self.fn_out, i)

                        if not os.path.exists(fn_out):
                            break
                    # end for

                    self.logging.print_verbose(Logging.INFO,
                                               "Write manual points ({} frames with {} detections) to file {}".
                                               format(len(self.manual_frames),
                                                      self.manual_frames.get_number_of_detections(), fn_out))

                    with open(fn_out, "w") as file:
                        frame_nr: int = 0

                        for frame in self.manual_frames:
                            frame_nr += 1

                            for detection in frame:
                                lat, lon, _ = pm.enu2geodetic(np.array(detection.x), np.array(detection.y),
                                                              np.asarray(0), np.asarray(self.observer.x),
                                                              np.asarray(self.observer.y), np.asarray(0), ell=None,
                                                              deg=True)
                                file.write("{} {} {}\n".format(lat, lon, frame_nr))
                        # end for
                    # end with
            # end if

            self.refresh.set()
        # end if
    # end def

    def update_window_limits(self) -> None:
        set_det_borders: bool = False
        set_prev_limits: bool = False
        set_manual_limits: bool = False

        if self.limits_mode == LimitsMode.ALL_DETECTIONS_INIT_ONLY:
            if not self.limits_mode_inited:
                set_det_borders = True
                self.limits_mode_inited = True
            else:
                set_prev_limits = True
            # end if

        elif self.limits_mode == LimitsMode.ALL_DETECTIONS_FIXED_UPDATE:
            set_det_borders = True

        elif self.limits_mode == LimitsMode.ALL_CANVAS_ELEMENTS_DYN_UPDATE:
            pass

        elif self.limits_mode == LimitsMode.MANUAL_AREA_INIT_ONLY:
            if not self.limits_mode_inited:
                set_manual_limits = True
                self.limits_mode_inited = True
            else:
                set_prev_limits = True
            # end if

        elif self.limits_mode == LimitsMode.MANUAL_AREA_FIXED_UPDATE:
            set_manual_limits = True
        # end if

        if set_det_borders:
            self.ax.set_xlim(self.det_borders.x_min, self.det_borders.x_max)
            self.ax.set_ylim(self.det_borders.y_min, self.det_borders.y_max)

        elif set_prev_limits:
            self.ax.set_xlim(self.prev_lim.x_min, self.prev_lim.x_max)
            self.ax.set_ylim(self.prev_lim.y_min, self.prev_lim.y_max)

        elif set_manual_limits:
            self.ax.set_xlim(self.limits_manual.x_min, self.limits_manual.x_max)
            self.ax.set_ylim(self.limits_manual.y_min, self.limits_manual.y_max)
        # end if
    # end def

    def _processing(self) -> None:
        if not os.path.isfile(self.fn_in):
            return

        # 1. Read all measurements from file
        file_reader: FileReader = FileReader(self.fn_in)
        line_handler: InputLineHandlerLatLonIdx = InputLineHandlerLatLonIdx()
        file_reader.read(line_handler)
        self.frames = line_handler.frame_list

        if len(self.frames) == 0:
            return

        # Convert from WGS84 to ENU, with its origin at the center of all points
        if self.observer is None:
            self.observer = self.frames.calc_center()

        self.frames = WGS84ToENUConverter.convert(frame_list_wgs84=self.frames, observer=self.observer)

        # Get the borders around the points for creating new particles later on
        self.det_borders = self.frames.calc_limits()

        # Filter depenand function
        self.processing()
    # end def

    def _update_window(self, _frame: Optional[int] = None) -> None:
        # This should block all subsequent calls to update_windows, but should be no problem
        self.refresh.wait(50. / 1000)
        self.refresh.clear()

        if self.ax is None:
            return

        # Store current limits for resetting it the next time
        # (after drawing the elements, which might unwantedly change the limits)
        self.prev_lim.x_min, self.prev_lim.x_max = self.ax.get_xlim()
        self.prev_lim.y_min, self.prev_lim.y_max = self.ax.get_ylim()

        self.ax.clear()

        # Filter depenand function
        self.update_window()

        # Manually set points
        for frame in self.manual_frames:
            self.ax.scatter([det.x for det in frame], [det.y for det in frame], s=20, marker="x")
            self.ax.plot([det.x for det in frame], [det.y for det in frame], color="black", linewidth=.5,
                         linestyle="--")
        # end for

        # Visualization settings (need to be set every time since they don't are permanent)
        self.update_window_limits()

        self.ax.set_aspect('equal', 'datalim')
        self.ax.grid(False)

        self.ax.set_xlabel('east [m]')
        self.ax.set_ylabel('north [m]')

        self.refresh_finished.set()
    # end def

    def _cb_keyboard(self) -> None:
        while True:
            cmd: str = input()

            try:

                self.cb_keyboard(cmd)

            except Exception as e:
                print("Invalid command. Exception: {}".format(e))
            # end try
        # end while
    # end def

    @abstractmethod
    def processing(self) -> None:
        pass

    @abstractmethod
    def update_window(self) -> None:
        pass

    @abstractmethod
    def cb_keyboard(self, cmd: str) -> None:
        pass
# end class
