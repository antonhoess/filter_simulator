#!/usr/bin/env python


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


class Limits:
    def __init__(self, x_min=None, y_min=None, x_max=None, y_max=None):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
    # end def
# end class


class WindowModeChecker:
    def __init__(self, default_window_mode: WindowMode, verbosity: Logging = 0):
        self.window_mode = default_window_mode
        self.control_shift_pressed = False
        self.control_shift_left_click_cnt = 0
        self.verbosity: Logging = verbosity

    def get_current_mode(self):
        return self.window_mode

    @staticmethod
    def key_is_ctrl_shift(key):
        # shift+control #ctrl+shift
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

                if self.verbosity >= Logging.DEBUG:
                    print("key_press_event: " + event.key)
            # end if
        # end if

        if action == "key_release_event":
            if self.key_is_ctrl_shift(event.key):
                if self.verbosity >= Logging.DEBUG:
                    print("key_release_event: " + event.key)

                if self.control_shift_left_click_cnt >= 3:
                    if self.window_mode == WindowMode.SIMULATION:
                        self.window_mode = WindowMode.MANUAL_EDITING
                    else:
                        self.window_mode = WindowMode.SIMULATION
                    # end if
                    if self.verbosity >= Logging.DEBUG:
                        print("Changed window mode to {}.".format(WindowMode(self.window_mode).name))
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


class Detection(Position):
    def __init__(self, x, y):
        super().__init__(x, y)


class Frame:
    def __init__(self):
        self.detections = []

    def add_detection(self, detection: Detection):
        self.detections.append(detection)

    def del_last_detection(self):
        if len(self.detections) > 0:
            del self.detections[-1]
    # end def

    def del_all_detections(self):
        for _ in reversed(range(len(self.detections))):  # reverse() maybe might make sense at a later point
            del self.detections[-1]
    # end def


class FrameList:
    def __init__(self):
        self.frames = []

    def add_empty_frame(self):
        self.frames.append(Frame())

    def del_last_frame(self):
        if len(self.frames) > 0:
            self.get_current_frame().del_all_detections()
            del self.frames[-1]
        # end if

    def get_current_frame(self) -> Frame:
        if len(self.frames) > 0:
            return self.frames[-1]
        else:
            return None

    def get_number_of_detections(self):
        n = 0

        for frame in self.frames:
            n += len(frame.detections)
        # end for

        return n

    def foreach_detection(self, cb_detection, **kwargs):
        for frame in self.frames:
            for detection in frame.detections:
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

        for frame in frame_list_wgs84.frames:
            frame_list_enu.add_empty_frame()

            for detection in frame.detections:
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
    def __init__(self, fn_in, fn_out, n_part, s_gauss, speed, verbosity, observer):
        self.fn_in = fn_in
        self.fn_out = fn_out
        self.n_part = n_part
        self.s_gauss = s_gauss
        self.speed = speed
        self.coords_x = []
        self.coords_y = []
        self.coords_idx = []
        self.frame_list = FrameList()

        self.refresh = True
        self.particles = []
        self.step = 0
        self.part_borders = Limits(-10, -10, 10, 10)
        self.m_x = None
        self.m_y = None
        self.m_confident = False
        self.ax = None
        self.next = False
        self.observer = observer
        self.simulation_direction = SimulationDirection.FORWARD
        self.window_mode_checker = WindowModeChecker(default_window_mode=WindowMode.SIMULATION, verbosity=verbosity)
        self.manual_points = FrameList()
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

    def processing(self):
        if not os.path.isfile(self.fn_in):
            return

        # 1. Read all measurements from file
        file_reader = FileReader(self.fn_in)
        line_handler = InputLineHandlerLatLonIdx()
        file_reader.read(line_handler)
        self.frame_list = line_handler.frame_list

        if len(self.frame_list.frames) == 0:
            return

        # Convert from WGS84 to ENU, with its origin at the center of all points
        if self.observer is None:
            self.observer = self.frame_list.calc_center()

        self.frame_list = WGS84ToENUConverter.convert(frame_list_wgs84=self.frame_list, observer=self.observer)

        # Add coordinates to coords-array just for compatibility XXX
        def add_detection_to_coords(detection: Detection):
            self.coords_x.append(detection.x)
            self.coords_y.append(detection.y)
        # end def

        self.frame_list.foreach_detection(add_detection_to_coords)

        # Get the borders around the points for creating new particles later on
        self.part_borders = self.frame_list.calc_limits()

        # 2. Generate many particles
        self.particles = Particle.create_random(self.n_part, self.part_borders)

        # 3. Generate a robot
        # robbie = Robot(world)

        # 4. Simulation loop
        while True:
            if self.verbosity >= Logging.INFO:
                print("Step {}".format(self.step + 1))

            # Wait for Return-Key-Press (console) of mouse click (GUI)
            while not self.next:
                time.sleep(0.1)

            self.next = False

            # 4.1 Read Robbie's sensor:
            #     i.e., get the distance r_d to the nearest beacon
            #    r_d = robbie.read_sensor(world)
            #self.obj.x = self.coords_x[self.step]
            #self.obj.y = self.coords_y[self.step]
            if self.simulation_direction == SimulationDirection.FORWARD:
                if self.step < (len(self.frame_list.frames) - 1):
                    self.step += 1
            else:
                if self.step > 0:
                    self.step -= 1
            # end if

            # 4.2 Update particle weight according to how good every particle matches
            #     Robbie's sensor reading
            for p in self.particles:
                # get distance of particle to nearest beacon
                d_x = p.x - self.coords_x[self.step]
                d_y = p.y - self.coords_y[self.step]
                p_d = math.sqrt(d_x * d_x + d_y * d_y)
                p.w = self.w_gauss(p_d, self.s_gauss)  # XXX

            # 4.3 Compute weighted mean of particles (gray circle)
            self.m_x, self.m_y, self.m_confident = self.compute_mean_point()

            # Mean shift XXX
            X = np.array([[p.x, p.y] for p in self.particles])

            clustering = True
            if clustering:
                clust = MeanShift(bandwidth=10).fit(X)
                self.cluster_centers_ = clust.cluster_centers_
                if self.verbosity >= Logging.DEBUG:
                    print(clust.labels_)
                    print(clust.cluster_centers_)
                # end if
            # end if

            # 4.4 show particles, show mean point, show Robbie
            # Wait until drawing has finished (do avoid changing e.g. particles
            # before they are drawn in their current position)
            self.refresh = True

            while self.refresh:
                time.sleep(0.1)

            # 4.5 Resampling follows here:
            resampling = True
            if resampling:
                new_particles = []

                # 4.5.1 Normalise weights
                nu = sum(p.w for p in self.particles)
                if self.verbosity >= Logging.DEBUG:
                    print("nu = {}".format(nu))

                if nu:
                    for p in self.particles:
                        p.w = p.w / nu
                # end if

                # 4.5.2 create a weighted distribution, for fast picking
                dist = WeightedDistribution(self.particles)

                if self.verbosity >= Logging.INFO:
                    print("# particles: {}".format(len(self.particles)))

                cnt = 0
                for _ in range(len(self.particles)):
                    p = dist.pick()
                    if p is None:  # No pick b/c all totally improbable
                        new_particle = Particle.create_random(1, self.part_borders)[0]
                        cnt += 1
                    else:
                        new_particle = p
                        new_particle.w = 1
                    # end if

                    x, y = self.add_noise(1, 0, 0)
                    new_particle.move_by(x, y)

                    new_particles.append(new_particle)
                # end for

                if self.verbosity >= Logging.INFO:
                    print("# particles newly created: {}".format(cnt))

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
            for p in self.particles:
                d_x = p.x - self.coords_x[self.step]
                d_y = p.y - self.coords_y[self.step]
                p_d = math.sqrt(d_x * d_x + d_y * d_y)
                angle = math.atan2(self.coords_y[self.step] - p.y, self.coords_x[self.step] - p.x)
                # p_d = 1.0  # XXX
                # p.x += self.speed * p_d * math.cos(angle)
                # p.y += self.speed * p_d * math.sin(angle)
                p.move_by(min(self.speed, p_d) * math.cos(angle), min(self.speed, p_d) * math.sin(angle))
            # end for

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
                if self.verbosity >= Logging.DEBUG:
                    print("Right click")

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
                    if len(self.manual_points.frames) == 0:
                        self.manual_points.add_empty_frame()

                    self.manual_points.get_current_frame().add_detection(Detection(event.xdata, event.ydata))

                    if self.verbosity >= Logging.INFO:
                        print("Add point {:4f}, {:4f} to frame # {}".format(event.xdata, event.ydata, len(self.manual_points.frames)))

                elif event.key == "shift":
                    self.manual_points.add_empty_frame()

                    if self.verbosity >= Logging.INFO:
                        print("Add new track (# {})".format(len(self.manual_points.frames)))
                # end if

            elif event.button == 3:  # Right click
                if event.key == "control":
                    if self.manual_points.get_current_frame() is not None:
                        self.manual_points.get_current_frame().del_last_detection()

                elif event.key == "shift":
                    self.manual_points.del_last_frame()

                elif WindowModeChecker.key_is_ctrl_shift(event.key):
                    fn_out = self.fn_out

                    for i in range(100):
                        fn_out = "{}_{:02d}".format(self.fn_out, i)

                        if not os.path.exists(fn_out):
                            break
                    # end for

                    if self.verbosity >= Logging.INFO:
                        print("Write manual points ({} frames with {} detections) to file {}".format(len(self.manual_points.frames), self.manual_points.get_number_of_detections(), fn_out))

                    with open(fn_out, "w") as file:
                        frame_nr = 0
                        for frame in self.manual_points.frames:
                            frame_nr += 1

                            for detection in frame.detections:
                                lat, lon, _ = pm.enu2geodetic(detection.x, detection.y, np.asarray(0), np.asarray(self.observer.x),
                                                              np.asarray(self.observer.y), np.asarray(0), ell=None,
                                                              deg=True)
                                file.write("{} {} {}\n".format(lat, lon, frame_nr))
                        # end for
                    # end with
            # end if

            self.refresh = True
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
        x = np.linspace(self.part_borders[0], self.part_borders[2], grid_res)
        y = np.linspace(self.part_borders[1], self.part_borders[3], grid_res)

        X, Y = np.meshgrid(x, y)
        Z = self.calc_density(X, Y)

        return X, Y, Z

    def update_window(self, _frame=None):
        if not self.refresh or self.ax is None:
            return

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

        # All detections
        for frame in self.frame_list.frames:
            self.ax.scatter([det.x for det in frame.detections], [det.y for det in frame.detections], edgecolor="green", marker="o")
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

        if len(self.frame_list.frames) > self.step:
            # Current detections
            det_pos_x = [d.x for d in self.frame_list.frames[self.step].detections]
            det_pos_y = [d.y for d in self.frame_list.frames[self.step].detections]
            self.ax.scatter(det_pos_x, det_pos_y, s=100, c="red", marker="x")

            # Importance weight Gaussian-kernel covariance ellipse
            ell_radius_x = self.s_gauss
            ell_radius_y = self.s_gauss
            ellipse = Ellipse((self.coords_x[self.step], self.coords_y[self.step]), width=ell_radius_x * 2,
                              height=ell_radius_y * 2, facecolor='none', edgecolor="black")
            self.ax.add_patch(ellipse)
        # end if

        # Manually set points
        for frame in self.manual_points.frames:
            self.ax.scatter([det.x for det in frame.detections], [det.y for det in frame.detections], s=20, marker="x")
            self.ax.plot([det.x for det in frame.detections], [det.y for det in frame.detections], color="black", linewidth=.5, linestyle="--")
        # end for

        # Visualization settings (need to be set every time since they don't are permanent)
        self.ax.set_xlim([self.part_borders.x_min, self.part_borders.x_max])
        self.ax.set_ylim([self.part_borders.y_min, self.part_borders.y_max])
        self.ax.set_aspect('equal', 'datalim')
        self.ax.grid(False)

        self.ax.set_xlabel('east [m]')
        self.ax.set_ylabel('north [m]')

        self.refresh = False
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
        return "{} -g <GAUSS_SIGMA> -h -i <INPUT_FILE> -n <N_PARTICLES> -o <OUTPUT_FILE> -p <OBSERVER_POSITION> -s <SPEED> -v <VERBOSITY>\n".format(argv[0]) + \
               "-g: Sigma of Gaussian importance weight kernel.\n" + \
               "-h: This help.\n" + \
               "-i: Input file to parse with coordinates in WGS 84 system." + \
               "-n: Number of particles.\n" + \
               "-o: Output file to write manually set coordinates converted to WGS84\n" + \
               "-p: Position of the observer in WGS84. Can be used instead of the center of the detections or in case of only manually creating detections, which needed to be transformed back to WGS84.\n" + \
               "-s: Speed of the object.\n" + \
               "-v: Verbosity level. 0 = Silent [Default], >0 = increasing verbosity.\n"
    # end def

    inputfile = ""
    outputfile = "out.lst"
    n_particles = 100
    sigma = 20.
    speed = 1.
    verbosity = Logging.INFO
    observer = None

    try:
        opts, args = getopt.getopt(argv[1:], "g:hi:n:o:p:s:v")
    except getopt.GetoptError as e:
        print("Reading parameters caused error {}".format(e))
        print(usage())
        sys.exit(2)
    # end try

    for opt, arg in opts:
        if opt == "-g":
            sigma = arg
        elif opt == '-h':
            print(usage())
            sys.exit()
        elif opt == "-i":
            inputfile = arg
        elif opt == "-n":
            n_particles = arg
        elif opt == "-o":
            outputfile = arg
        elif opt == "-p":
            fields = arg.split(";")
            if len(fields) >= 2:
                observer = Position(float(fields[0]), float(fields[1]))
        elif opt == "-v":
            verbosity = arg
    # end for

    sim = Simulator(fn_in=inputfile, fn_out=outputfile, n_part=n_particles, s_gauss=sigma, speed=speed,
                    verbosity=verbosity, observer=observer)
    sim.run()


if __name__ == "__main__":
    main(sys.argv)
