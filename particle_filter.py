#!/usr/bin/env python3


import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import random
import math
import bisect
import pymap3d as pm
import numpy as np
import threading
import time
from datetime import datetime
import seaborn as sns
from sklearn.cluster import MeanShift
import sys
import getopt
from enum import IntEnum
from abc import ABC, abstractmethod
from typing import Optional
import os


class Logging(IntEnum):
    NONE = 0
    CRITICAL = 1
    ERROR = 2
    WARNING = 3
    INFO = 4
    DEBUG = 5
# end class


class WindowMode(IntEnum):
    SIMULATION = 0
    MANUAL_EDITING = 1
# end class


class SimulationDirection(IntEnum):
    FORWARD = 0
    BACKWARD = 1
# end class


class LimitsMode(IntEnum):
    ALL_DETECTIONS_INIT_ONLY = 0
    ALL_DETECTIONS_FIXED_UPDATE = 1
    ALL_CANVAS_ELEMENTS_DYN_UPDATE = 2
    MANUAL_AREA_INIT_ONLY = 3
    MANUAL_AREA_FIXED_UPDATE = 4
# end class


class Limits:
    def __init__(self, x_min=None, y_min=None, x_max=None, y_max=None):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
    # end def
# end class


class WindowModeChecker:
    def __init__(self, default_window_mode: WindowMode, verbosity: Logging = Logging.NONE):
        self.window_mode = default_window_mode
        self.control_shift_pressed = False
        self.control_shift_left_click_cnt = 0
        self.verbosity: Logging = verbosity

    def get_current_mode(self):
        return self.window_mode

    @staticmethod
    def key_is_ctrl_shift(key):
        # shift+control or ctrl+shift
        if "shift" in key and ("control" in key or "ctrl" in key):
            return True
        else:
            return False
    # end def

    def check_event(self, action, event):
        if action == "button_press_event":
            if event.button == 1:  # Left click
                if self.control_shift_pressed:
                    self.control_shift_left_click_cnt += 1
                # end if
            # end if
        # end if

        if action == "button_release_event":
            pass

        if action == "key_press_event":
            if self.key_is_ctrl_shift(event.key):
                self.control_shift_pressed = True
                self.print_verbose(Logging.DEBUG, "key_press_event: " + event.key)
            # end if
        # end if

        if action == "key_release_event":
            if self.key_is_ctrl_shift(event.key):
                self.print_verbose(Logging.DEBUG, "key_release_event: " + event.key)

                if self.control_shift_left_click_cnt >= 3:
                    if self.window_mode == WindowMode.SIMULATION:
                        self.window_mode = WindowMode.MANUAL_EDITING
                    else:
                        self.window_mode = WindowMode.SIMULATION
                    # end if

                    self.print_verbose(Logging.DEBUG, "Changed window mode to {}.".format(WindowMode(self.window_mode).name))
                # end if
                self.control_shift_left_click_cnt = 0
                self.control_shift_pressed = False
            # end if
        # end if
    # end def
# end class


class WeightedDistribution:
    def __init__(self, state):
        accum = 0.0
        self.state = [p for p in state if p.w > 0]
        self.distribution = []
        for x in self.state:
            accum += x.w
            self.distribution.append(accum)

    def pick(self):
        try:
            # Due to numeric problems, the weight don't sum up to 1.0 after normalization,
            # so we can't pick from a uniform distribution in range [0, 1]
            return self.state[bisect.bisect_left(self.distribution, random.uniform(0, self.distribution[-1]))]
        except IndexError:
            # Happens when all particles are improbable w=0
            return None


class Particle:
    def __init__(self, x, y, w=1., noisy=False):
        if noisy:
            x, y = Simulator.add_some_noise(x, y)

        self.x = x
        self.y = y
        self.w = w

    def __repr__(self):
        return "(%f, %f, w=%f)" % (self.x, self.y, self.w)

    @property
    def xy(self):
        return self.x, self.y

    @classmethod
    def create_random(cls, count, limits: Limits):
        return [cls(random.uniform(limits.x_min, limits.x_max), random.uniform(limits.y_min, limits.y_max)) for _ in range(0, count)]

    def move_by(self, x, y):
        self.x += x
        self.y += y


class Obj(Particle):
    speed = 0.2

    def __init__(self):
        super(Obj, self).__init__(None, None)

    def read_sensor(self, maze):
        """
        Poor robot, it's sensors are noisy and pretty strange,
        it only can measure the distance to the nearest beacon(!)
        and is not very accurate at that too!
        """
        return Simulator.add_little_noise(super(Robot, self).read_sensor(maze))[0]

    def move(self, maze):
        """
        Move the robot. Note that the movement is stochastic too.
        """
        while True:
            self.step_count += 1
            if self.advance_by(self.speed, noisy=True, checker=lambda r, dx, dy: maze.is_free(r.x + dx, r.y + dy)):
                break
            # Bumped into something or too long in same direction,
            # chose random new direction
            self.chose_random_direction()


class Position:
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y

    def __str__(self):
        return "x={}, y={}".format(self.x, self.y)


class Detection(Position):
    def __init__(self, x, y):
        super().__init__(x, y)

    def __str__(self):
        return "x={}, y={}".format(self.x, self.y)


class FrameIterator:
    def __init__(self, frame):
        self._frame = frame
        self._index = 0

    def __next__(self):
        if self._index < len(self._frame.get_detections()):
            result = self._frame[self._index]
            self._index += 1

            return result

        # End of iteration
        raise StopIteration
# end class


class Frame:
    def __init__(self):
        self._detections = []

    def __iter__(self):
        return FrameIterator(self)

    def __getitem__(self, index):
        return self._detections[index]

    def __len__(self):
        return len(self._detections)

    def __str__(self):
        return "Frame with {} detections".format(len(self))

    def add_detection(self, detection: Detection):
        self._detections.append(detection)

    def del_last_detection(self):
        if len(self._detections) > 0:
            del self._detections[-1]
    # end def

    def del_all_detections(self):
        for _ in reversed(range(len(self))):  # reverse() maybe might make sense at a later point
            del self._detections[-1]
    # end def

    def get_detections(self):
        return self._detections


class FrameListIterator:
    def __init__(self, frame_list):
        self._frame_list = frame_list
        self._index = 0

    def __next__(self):
        if self._index < len(self._frame_list.get_frames()):
            result = self._frame_list[self._index]
            self._index += 1

            return result

        # End of iteration
        raise StopIteration
# end class


class FrameList:
    def __init__(self):
        self._frames = []

    def __iter__(self):
        return FrameListIterator(self)

    def __getitem__(self, index):
        return self._frames[index]

    def __len__(self):
        return len(self._frames)

    def __str__(self):
        return "FrameList with {} frames and {} detections total".format(len(self), self.get_number_of_detections())

    def add_empty_frame(self):
        self._frames.append(Frame())

    def del_last_frame(self):
        if len(self._frames) > 0:
            self.get_current_frame().del_all_detections()
            del self._frames[-1]
        # end if

    def get_frames(self):
        return self._frames

    def get_current_frame(self) -> Frame:
        if len(self._frames) > 0:
            return self._frames[-1]
        else:
            return None

    def get_number_of_detections(self):
        n = 0

        for frame in self._frames:
            n += len(frame)
        # end for

        return n

    def foreach_detection(self, cb_detection, **kwargs):
        for frame in self._frames:
            for detection in frame:
                cb_detection(detection, **kwargs)
        # end for

    @staticmethod
    def _update_limit_by_detection(detection, limits: Limits):
        if limits.x_min is None or detection.x < limits.x_min:
            limits.x_min = detection.x

        if limits.y_min is None or detection.y < limits.y_min:
            limits.y_min = detection.y

        if limits.x_max is None or detection.x > limits.x_max:
            limits.x_max = detection.x

        if limits.y_max is None or detection.y > limits.y_max:
            limits.y_max = detection.y
    # end def

    def calc_limits(self):
        limits = Limits()

        self.foreach_detection(self._update_limit_by_detection, limits=limits)

        return limits

    def calc_center(self):
        limits = self.calc_limits()

        return Position(x=(limits.x_min + limits.x_max) / 2, y=(limits.y_min + limits.y_max) / 2)


class InputLineHandler(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def handle_line(self, line):
        pass


class InputLineHandlerLatLonIdx(InputLineHandler):
    def __init__(self):
        self.cur_idx = None
        self.frame_list = FrameList()
        super().__init__()

    def handle_line(self, line):
        lat = None
        lon = None
        idx = None

        # Split line and read fields
        fields = line.split(" ")

        if len(fields) >= 2:
            lat = float(fields[0])
            lon = float(fields[1])
        # end if

        if len(fields) >= 3:
            idx = float(fields[2])
        # end if

        # Check if we need to add a new frame to the frame list
        if idx is None or self.cur_idx is None or idx != self.cur_idx:
            self.cur_idx = idx
            self.frame_list.add_empty_frame()

        # Add detection from field values to the frame
        self.frame_list.get_current_frame().add_detection(Detection(lat, lon))

        return
    # end def
# end class


class FileReader:
    def __init__(self, filename):
        self.filename = filename

    def read(self, input_line_handler: InputLineHandler):
        try:
            with open(self.filename, 'r') as file:
                while True:
                    # Get next line from file
                    line = file.readline()

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


class WGS84ToENUConverter:
    @staticmethod
    def convert(frame_list_wgs84: FrameList, observer: Optional[Position]):
        frame_list_enu = FrameList()

        if observer is None:
            observer = frame_list_wgs84.calc_center()

        for frame in frame_list_wgs84:
            frame_list_enu.add_empty_frame()

            for detection in frame:
                # Convert...
                e, n, _ = pm.geodetic2enu(np.asarray(detection.x), np.asarray(detection.y), np.asarray(0),
                                          np.asarray(observer.x), np.asarray(observer.y), np.asarray(0),
                                          ell=None, deg=True)
                frame_list_enu.get_current_frame().add_detection(Detection(e, n))
            # end for
        # end for

        return frame_list_enu
    # end def
# end class


class Simulator:
    def __init__(self, fn_in, fn_out, limits, n_part, s_gauss, speed, verbosity, observer):
        self.fn_in = fn_in
        self.fn_out = fn_out
        self.n_part = n_part
        self.s_gauss = s_gauss
        self.speed = speed
        self.frames = FrameList()
        self.cur_frame = None

        self.refresh = threading.Event()
        self.refresh_finished = threading.Event()
        self.particles = []
        self.step = -1
        self.limits_manual = limits
        self.det_borders = self.limits_manual
        self.limits_mode: LimitsMode = LimitsMode.ALL_DETECTIONS_INIT_ONLY
        self.limits_mode_inited = False
        self.x_lim = (0, 0)
        self.y_lim = (0, 0)
        self.m_x = None
        self.m_y = None
        self.m_confident = False
        self.ax = None
        self.next = False
        self.observer = observer
        self.simulation_direction = SimulationDirection.FORWARD
        self.window_mode_checker = WindowModeChecker(default_window_mode=WindowMode.SIMULATION, verbosity=verbosity)
        self.manual_frames = FrameList()
        self.verbosity = verbosity

    def _cb_keyboard(self):
        while True:
            cmd = input()

            try:
                if cmd == "":
                    self.next = True

                elif cmd == "+":
                    pass #XXX

                elif cmd.startswith("-"):
                    idx = int(cmd[1:])

            except Exception as e:
                print("Invalid command. Exception: {}".format(e))

    def set_next_step(self):
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

    def wait_for_valid_next_step(self):
        while True:
            # Wait for Return-Key-Press (console) of mouse click (GUI)
            while not self.next:
                time.sleep(0.1)

            self.next = False

            # Only continue when the next requested step is valid, e.g. it is within its boundaries
            if self.set_next_step():
                break
        # end while

    def print_verbose(self, verbosity, message):
        if self.verbosity >= verbosity:
            print(message)

    def processing(self):
        if not os.path.isfile(self.fn_in):
            return

        # 1. Read all measurements from file
        file_reader = FileReader(self.fn_in)
        line_handler = InputLineHandlerLatLonIdx()
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

        # 2. Generate many particles
        self.particles = Particle.create_random(self.n_part, self.det_borders)

        # 3. Generate a robot
        # robbie = Robot(world)

        # 4. Simulation loop
        while True:
            # Calculate mean shift
            clustering = True
            if clustering:
                cluster_samples = np.array([[p.x, p.y] for p in self.particles])
                clust = MeanShift(bandwidth=10).fit(cluster_samples)
                self.cluster_centers_ = clust.cluster_centers_
                self.print_verbose(Logging.DEBUG, clust.labels_)
                self.print_verbose(Logging.DEBUG, clust.cluster_centers_)
            # end if

            # 4.3 Compute weighted mean of particles (gray circle)
            self.m_x, self.m_y, self.m_confident = self.compute_mean_point()

            # Wait until drawing has finished (do avoid changing e.g. particles
            # before they are drawn in their current position)

            self.refresh.set()

            self.refresh_finished.wait()
            self.refresh_finished.clear()

            # Wait for a valid next step
            self.wait_for_valid_next_step()
            self.print_verbose(Logging.INFO, "Step {}".format(self.step))

            # Set current frame
            self.cur_frame = self.frames[self.step]

            # 4.2 Update particle weight according to how good every particle matches
            #     Robbie's sensor reading
            for p in self.particles:
                w_total = .0

                for det in self.cur_frame:
                    # get distance of particle to nearest beacon
                    d_x = p.x - det.x
                    d_y = p.y - det.y
                    p_d = math.sqrt(d_x * d_x + d_y * d_y)
                    w_total += self.w_gauss(p_d, self.s_gauss)
                # end for

                n_vec = len(self.cur_frame)

                if n_vec > 0:
                    w_total /= n_vec

                p.w = w_total
            # end for

            # 4.5 Resampling follows here:
            resampling = True
            if resampling:
                new_particles = []

                # 4.5.1 Normalise weights
                nu = sum(p.w for p in self.particles)
                self.print_verbose(Logging.DEBUG, "nu = {}".format(nu))

                if nu:
                    for p in self.particles:
                        p.w = p.w / nu
                # end if

                # 4.5.2 create a weighted distribution, for fast picking
                dist = WeightedDistribution(self.particles)
                self.print_verbose(Logging.INFO, "# particles: {}".format(len(self.particles)))

                cnt = 0
                for _ in range(len(self.particles)):
                    p = dist.pick()
                    if p is None:  # No pick b/c all totally improbable
                        new_particle = Particle.create_random(1, self.det_borders)[0]
                        cnt += 1
                    else:
                        new_particle = Particle(p.x, p.y, 1.)
                    # end if

                    x, y = self.add_noise(.1, 0, 0)
                    new_particle.move_by(x, y)

                    new_particles.append(new_particle)
                # end for

                self.print_verbose(Logging.INFO, "# particles newly created: {}".format(cnt))
                self.particles = new_particles
            # end if

            # 4.5.3 Move Robbie in world (randomly)
            #    old_heading = robbie.h
            #    robbie.move(world)
            for p in self.particles:
                x, y = self.add_noise(10, 0, 0)
                p.move_by(x, y)
            # end for

            # 4.5.4 Move all particles according to belief of movement
            d_x = .0
            d_y = .0
            for p in self.particles:

                # Add all vectors
                for det in self.cur_frame:
                    d_x += (det.x - p.x)
                    d_y += (det.y - p.y)
                # end for

                # Calculate resulting vector
                n_vec = len(self.cur_frame)
                if n_vec > 0:
                    d_x = d_x / n_vec
                    d_y = d_y / n_vec
                # end if

                p_d = math.sqrt(d_x * d_x + d_y * d_y)
                angle = math.atan2(d_y, d_x)
                # p_d = 1.0  # XXX
                # p.x += self.speed * p_d * math.cos(angle)
                # p.y += self.speed * p_d * math.sin(angle)
                p.move_by(min(self.speed, p_d) * math.cos(angle), min(self.speed, p_d) * math.sin(angle))
            # end for
        # end while

    def run(self):
        # Processing thread
        t_proc = threading.Thread(target=self.processing)
        t_proc.start()

        # Keyboard thread
        t_kbd = threading.Thread(target=self._cb_keyboard)
        t_kbd.daemon = True
        t_kbd.start()

        # Prepare GUI
        fig = plt.figure()
        fig.canvas.set_window_title("State Space")
        self.ax = fig.add_subplot(1, 1, 1)

        # self.cid = fig.canvas.mpl_connect('button_press_event', self._cb_button_press_event)
        fig.canvas.mpl_connect("button_press_event", self._cb_button_press_event)
        fig.canvas.mpl_connect("button_release_event", self._cb_button_release_event)
        fig.canvas.mpl_connect("key_press_event", self._cb_key_press_event)
        fig.canvas.mpl_connect("key_release_event", self._cb_key_release_event)

        # Cyclic update check (but only draws, if there's something new)
        _anim = animation.FuncAnimation(fig, self.update_window, interval=100)

        # Show blocking window which draws the current state and handles mouse clicks
        plt.show()
    # end def

    def _cb_button_press_event(self, event):
        self.window_mode_checker.check_event(action="button_press_event", event=event)
        self.handle_mpl_event(event)

    def _cb_button_release_event(self, event):
        self.window_mode_checker.check_event(action="button_release_event", event=event)

    def _cb_key_press_event(self, event):
        self.window_mode_checker.check_event(action="key_press_event", event=event)

    def _cb_key_release_event(self, event):
        self.window_mode_checker.check_event(action="key_release_event", event=event)

    def handle_mpl_event(self, event):
        # xxx mit überlegen, in welchem modus ich die punkte (also mit mehrerer detektionen pro frame) durch klicken
        # erstellen will - vllt. auch zweilei modi. die geklickten punkte entsprechend eingefärbt darstellen und
        # abspeichern. auch dies erlauben, wenn keine daten geladen wurden, durch angabe von einem default fenster,
        # das ja dann noch gezoomt und verschoben werden kann, um die stelle zu finden, die man möchte. es bräuchte
        # hierzu natürlich auch noch eine observer gps position:
        # man muss auch etwas überspringen können und tracks beenden können
        # hauptfrage: erst track 1, dann track 2, etc. oder erst alle detektionen in frame 1, dann in frame 2, etc.
        # track-weise scheint erst mal unlogisch, da die je erst später erstellt werden, oder doch nicht? es wäre jedoch
        # einfach zu klicken, aber es besteht auch die gefahr, dass die zeiten der verschiedenen tracks auseinander
        # laufen, wenn ich beim einen viel mehr klicks mache, als beim anderen und diese am ende wieder zusammenführen...

        if self.window_mode_checker.get_current_mode() == WindowMode.SIMULATION:
            # Right mouse button: Navigate forwards / backwards
            #   * Ctrl: Forwards
            #   * Shift: Backwards
            if event.button == 3:  # Right click
                self.print_verbose(Logging.DEBUG, "Right click")

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
                    e = event.xdata
                    n = event.ydata
                    lat, lon, _ = pm.enu2geodetic(e, n, np.asarray(0), np.asarray(self.observer.x),
                                                  np.asarray(self.observer.y), np.asarray(0), ell=None, deg=True)

                    # print("{} {} {}".format(lat, lon, len(self.manual_points)))
                    # Add initial frame
                    if len(self.manual_frames) == 0:
                        self.manual_frames.add_empty_frame()

                    self.manual_frames.get_current_frame().add_detection(Detection(event.xdata, event.ydata))
                    self.print_verbose(Logging.INFO, "Add point {:4f}, {:4f} to frame # {}".format(event.xdata, event.ydata, len(self.manual_frames)))

                elif event.key == "shift":
                    self.manual_frames.add_empty_frame()
                    self.print_verbose(Logging.INFO, "Add new track (# {})".format(len(self.manual_frames)))
                # end if

            elif event.button == 3:  # Right click
                if event.key == "control":
                    if self.manual_frames.get_current_frame() is not None:
                        self.manual_frames.get_current_frame().del_last_detection()

                elif event.key == "shift":
                    self.manual_frames.del_last_frame()

                elif WindowModeChecker.key_is_ctrl_shift(event.key):
                    fn_out = self.fn_out

                    for i in range(100):
                        fn_out = "{}_{:02d}".format(self.fn_out, i)

                        if not os.path.exists(fn_out):
                            break
                    # end for

                    self.print_verbose(Logging.INFO, "Write manual points ({} frames with {} detections) to file {}".format(len(self.manual_frames), self.manual_frames.get_number_of_detections(), fn_out))

                    with open(fn_out, "w") as file:
                        frame_nr = 0
                        for frame in self.manual_frames:
                            frame_nr += 1

                            for detection in frame:
                                lat, lon, _ = pm.enu2geodetic(detection.x, detection.y, np.asarray(0), np.asarray(self.observer.x),
                                                              np.asarray(self.observer.y), np.asarray(0), ell=None,
                                                              deg=True)
                                file.write("{} {} {}\n".format(lat, lon, frame_nr))
                        # end for
                    # end with
            # end if

            self.refresh.set()
        # end if

    def calc_density(self, x, y):
        accum = 0.

        for p in self.particles:
            d_x = p.x - x
            d_y = p.y - y
            p_d = 1. / np.sqrt(d_x * d_x + d_y * d_y)
            accum += p_d
        # end for

        return accum

    def calc_density_map(self, grid_res=100):
        x = np.linspace(self.det_borders.x_min, self.det_borders.x_max, grid_res)
        y = np.linspace(self.det_borders.y_min, self.det_borders.y_max, grid_res)

        X, Y = np.meshgrid(x, y)
        Z = self.calc_density(X, Y)

        return X, Y, Z

    def update_window_limits(self):
        set_det_borders = False
        set_prev_limits = False
        set_manual_limits = False

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
            self.ax.set_xlim([self.det_borders.x_min, self.det_borders.x_max])
            self.ax.set_ylim([self.det_borders.y_min, self.det_borders.y_max])

        elif set_prev_limits:
            self.ax.set_xlim(self.x_lim)
            self.ax.set_ylim(self.y_lim)

        elif set_manual_limits:
            self.ax.set_xlim([self.limits_manual.x_min, self.limits_manual.x_max])
            self.ax.set_ylim([self.limits_manual.y_min, self.limits_manual.y_max])
        # end if

    def update_window(self, _frame=None):
        self.refresh.wait(50. / 1000)  # This should block all subsequent calls to update_windows, but should be no problem
        self.refresh.clear()

        if self.ax is None:
            return

        # Store current limits for resetting it the next time
        # (after drawing the elements, which might unwantedly change the limits)
        self.x_lim = self.ax.get_xlim()
        self.y_lim = self.ax.get_ylim()

        self.ax.clear()

        # Draw density map
        draw_kde = True
        if not draw_kde:
            X, Y, Z = self.calc_density_map(grid_res=100)
            self.ax.contourf(X, Y, Z, 20, cmap='Blues')
        else:
            X = [p.x for p in self.particles]
            Y = [p.y for p in self.particles]
            sns.kdeplot(X, Y, shade=True, ax=self.ax)
        # end if

        # All detections - each frame's detections in a different color
        for frame in self.frames:
            self.ax.scatter([det.x for det in frame], [det.y for det in frame], edgecolor="green", marker="o")
        # end for

        # Weighted mean
        self.ax.scatter([self.m_x], [self.m_y], s=200, c="gray" if self.m_confident else "pink", edgecolor="black",
                        marker="o")

        # Particles
        self.ax.scatter([p.x for p in self.particles], [p.y for p in self.particles], s=5, edgecolor="blue", marker="o")

        # Mean shift centers XXX
        if hasattr(self, 'cluster_centers_'):
            self.ax.scatter([cc[0] for cc in self.cluster_centers_],
                            [cc[1] for cc in self.cluster_centers_], s=25, edgecolor="orange", marker="x")

        if self.cur_frame is not None:
            # Current detections
            det_pos_x = [det.x for det in self.cur_frame]
            det_pos_y = [det.y for det in self.cur_frame]
            self.ax.scatter(det_pos_x, det_pos_y, s=100, c="red", marker="x")

            # Importance weight Gaussian-kernel covariance ellipse
            ell_radius_x = self.s_gauss
            ell_radius_y = self.s_gauss

            for det in self.cur_frame:
                ellipse = Ellipse((det.x, det.y), width=ell_radius_x * 2,
                                  height=ell_radius_y * 2, facecolor='none', edgecolor="black", linewidth=.5)
                self.ax.add_patch(ellipse)
            # end for
        # end if

        # Manually set points
        for frame in self.manual_frames:
            self.ax.scatter([det.x for det in frame], [det.y for det in frame], s=20, marker="x")
            self.ax.plot([det.x for det in frame], [det.y for det in frame], color="black", linewidth=.5, linestyle="--")
        # end for

        # Visualization settings (need to be set every time since they don't are permanent)
        self.update_window_limits()

        self.ax.set_aspect('equal', 'datalim')
        self.ax.grid(False)

        self.ax.set_xlabel('east [m]')
        self.ax.set_ylabel('north [m]')

        self.refresh_finished.set()
    # end def

    @staticmethod
    def add_noise(level, *coords):
        return [x + random.uniform(-level, level) for x in coords]

    @staticmethod
    def add_little_noise(*coords):
        return Simulator.add_noise(0.02, *coords)

    @staticmethod
    def add_some_noise(*coords):
        return Simulator.add_noise(0.1, *coords)

    # This is just a gaussian kernel I pulled out of my hat, to transform
    # values near to robbie's measurement => 1, further away => 0
    @staticmethod
    def w_gauss(x, sigma):
        g = math.e ** -(x * x / (2 * sigma * sigma))

        return g

    def compute_mean_point(self):
        """
        Compute the mean for all particles that have a reasonably good weight.
        This is not part of the particle filter algorithm but rather an
        addition to show the "best belief" for current position.
        """

        m_x = 0
        m_y = 0
        m_count = 0

        for p in self.particles:
            m_count += p.w
            m_x += p.x * p.w
            m_y += p.y * p.w

        if m_count == 0:
            return -1, -1, False

        m_x /= m_count
        m_y /= m_count

        # Now compute how good that mean is -- check how many particles
        # actually are in the immediate vicinity
        m_count = 0

        for p in self.particles:
            d_x = p.x - m_x
            d_y = p.y - m_y
            p_d = math.sqrt(d_x * d_x + d_y * d_y)

            if p_d < 3:
                m_count += 1

        return m_x, m_y, m_count > len(self.particles) * 0.95


def main(argv):
    # Library settings
    sns.set(color_codes=True)

    # Initialize random generator
    random.seed(datetime.now())

    # Read command line arguments
    def usage():
        return "{} <PARAMETERS>\n".format(os.path.basename(argv[0])) + \
               "\n" + \
               "-g <GAUSS_SIGMA>:\n" + \
               "    Sigma of Gaussian importance weight kernel.\n" + \
               "\n" + \
               "-h: This help.\n" + \
               "\n" + \
               "-i <INPUT_FILE>:\n" + \
               "    Input file to parse with coordinates in WGS 84 system.\n" + \
               "\n" + \
               "-l <LIMITS>:\n" + \
               "    Fixed limits for the canvas in format 'X_MIN;Y_MIN;X_MAX;Y_MAX'.\n" + \
               "\n" + \
               "-n <N_PARTICLES>:\n" + \
               "    Number of particles.\n" + \
               "\n" + \
               "-o <OUTPUT_FILE>:\n" \
               "    Output file to write manually set coordinates converted to WGS84\n" + \
               "\n" + \
               "-p <OBSERVER_POSITION>:\n" \
               "    Position of the observer in WGS84. Can be used instead of the center of the detections or in case of only manually creating detections, which needed to be transformed back to WGS84.\n" + \
               "\n" + \
               "-s <SPEED>:\n" \
               "    Speed of the object.\n" + \
               "\n" + \
               "-v <VERBOSITY>:\n" \
               "    Verbosity level. 0 = Silent [Default], >0 = decreasing verbosity.\n"
    # end def

    inputfile = ""
    outputfile = "out.lst"
    limits = Limits(-10, -10, 10, 10)
    n_particles = 100
    sigma = 20.
    speed = 1.
    verbosity = Logging.INFO
    observer = None

    try:
        opts, args = getopt.getopt(argv[1:], "g:hi:l:n:o:p:s:v")
    except getopt.GetoptError as e:
        print("Reading parameters caused error {}".format(e))
        print(usage())
        sys.exit(2)
    # end try

    for opt, arg in opts:
        if opt == "-g":
            sigma = float(arg)

        elif opt == '-h':
            print(usage())
            sys.exit()

        elif opt == "-i":
            inputfile = arg

        elif opt == "-l":
            fields = arg.split(";")
            if len(fields) == 4:
                limits = Limits(float(fields[0]), float(fields[1]), float(fields[2]), float(fields[3]))

        elif opt == "-n":
            n_particles = int(arg)

        elif opt == "-o":
            outputfile = arg

        elif opt == "-p":
            fields = arg.split(";")
            if len(fields) >= 2:
                observer = Position(float(fields[0]), float(fields[1]))

        elif opt == "-v":
            verbosity = int(arg)
    # end for

    sim = Simulator(fn_in=inputfile, fn_out=outputfile, limits=limits, n_part=n_particles, s_gauss=sigma, speed=speed,
                    verbosity=verbosity, observer=observer)
    sim.run()


if __name__ == "__main__":
    main(sys.argv)
