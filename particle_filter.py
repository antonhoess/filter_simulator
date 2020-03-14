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


class Logging(IntEnum):
    NONE = 0
    CRITICAL = 1
    ERROR = 2
    WARNING = 3
    INFO = 4
    DEBUG = 5


class Limits:
    def __init__(self):
        self.x_min = None
        self.y_min = None
        self.x_max = None
        self.y_max = None
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

    def add_detection(self, detection):
        self.detections.append(detection)


class FrameList:
    def __init__(self):
        self.frames = []

    def append_empty_frame(self):
        self.frames.append(Frame())

    def get_current_frame(self) -> Frame:
        return self.frames[-1]

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
            self.frame_list.append_empty_frame()

        # Add detection from field values to the frame
        self.frame_list.get_current_frame().add_detection(Detection(lat, lon))

        return
    # end def
# end class


class FileReader:
    def __init__(self, filename):
        self.filename = filename

    def read(self, input_line_handler: InputLineHandler):
        with open(self.filename, 'r') as file:
            while True:
                # Get next line from file
                line = file.readline()

                # if line is empty end of file is reached
                if not line:
                    break
                else:
                    input_line_handler.handle_line(line)
                # end if
            # end while
        # end with
    # end def
# end class


class WGS84ToENUConverter:
    @staticmethod
    def convert(frame_list_wgs84: FrameList, observer: Optional[Position]):
        frame_list_enu = FrameList()

        if observer is None:
            observer = frame_list_wgs84.calc_center()

        for frame in frame_list_wgs84.frames:
            frame_list_enu.append_empty_frame()

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
    def __init__(self, fn_in, fn_out, n_part, s_gauss, speed, verbosity):
        self.fn_in = fn_in
        self.fn_out = fn_out
        self.n_part = n_part
        self.s_gauss = s_gauss
        self.speed = speed
        self.coords_x = []
        self.coords_y = []
        self.coords_idx = []
        self.frame_list = None

        self.refresh = True
        self.particles = []
        self.step = 0
        self.part_borders: Optional[Limits] = None
        self.m_x = None
        self.m_y = None
        self.m_confident = False
        self.ax = None
        self.obj = None
        self.next = False
        self.cid = None
        self.observer = None
        self.manual_points = []
        self.verbosity = verbosity

        self.manual_points.append([])

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
        # 1. Read all measurements from file
        file_reader = FileReader(self.fn_in)
        line_handler = InputLineHandlerLatLonIdx()
        file_reader.read(line_handler)
        self.frame_list = line_handler.frame_list

        # Convert from WGS84 to ENU, with its origin at the center of all points
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
        self.obj = Obj()

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
            self.obj.x = self.coords_x[self.step]
            self.obj.y = self.coords_y[self.step]
            self.step += 1

            # 4.2 Update particle weight according to how good every particle matches
            #     Robbie's sensor reading
            for p in self.particles:
                # get distance of particle to nearest beacon
                d_x = p.x - self.obj.x
                d_y = p.y - self.obj.y
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
                d_x = p.x - self.obj.x
                d_y = p.y - self.obj.y
                p_d = math.sqrt(d_x * d_x + d_y * d_y)
                angle = math.atan2(self.obj.y - p.y, self.obj.x - p.x)
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
        fig.canvas.set_window_title('State Space')
        self.ax = fig.add_subplot(1, 1, 1)

        self.cid = fig.canvas.mpl_connect('button_press_event', self._cb_button_press_event)

        # Cyclic update check (but only draws, if there's something new)
        _anim = animation.FuncAnimation(fig, self.update_window, interval=100)

        # Show blocking window which draws the current state and handles mouse clicks
        plt.show()

    def _cb_button_press_event(self, event):
        if event.button == 1 and event.key == "control":  # Ctrl-Left click
            if self.verbosity >= Logging.INFO:
                print("Add new track")

            self.manual_points.append([])

        elif event.button == 1 and event.key == "shift":  # Shift-Left click
            if self.verbosity >= Logging.INFO:
                print("Add point {:4f}, {:4f} to track # {}".format(event.xdata, event.ydata, len(self.manual_points)))

            e = event.xdata
            n = event.ydata
            lat, lon, _ = pm.enu2geodetic(e, n, np.asarray(0), np.asarray(self.observer.x), np.asarray(self.observer.y),
                                          np.asarray(0), ell=None, deg=True)

            print("{} {} {}".format(lat, lon, len(self.manual_points)))
            self.manual_points[-1].append((event.xdata, event.ydata))
            self.refresh = True

        elif event.button == 3:  # Right click
            if self.verbosity >= Logging.DEBUG:
                print("Right click")

            self.next = True
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
        if not self.refresh or self.ax is None or self.obj is None:
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
        self.ax.scatter(self.coords_x, self.coords_y, c="green", edgecolor="darkgreen", marker="o")

        # Weighted mean
        self.ax.scatter([self.m_x], [self.m_y], s=200, c="gray" if self.m_confident else "pink", edgecolor="black",
                        marker="o")

        # Particles
        self.ax.scatter([p.x for p in self.particles], [p.y for p in self.particles], s=5, edgecolor="blue", marker="o")

        # Mean shift centers XXX
        if hasattr(self, 'cluster_centers_'):
            self.ax.scatter([cc[0] for cc in self.cluster_centers_],
                            [cc[1] for cc in self.cluster_centers_], s=25, edgecolor="orange", marker="x")

        # Current detection
        self.ax.scatter([self.obj.x], [self.obj.y], s=100, c="red", marker="x")

        # Importance weight Gaussian-kernel covariance ellipse
        ell_radius_x = self.s_gauss
        ell_radius_y = self.s_gauss
        ellipse = Ellipse((self.obj.x, self.obj.y), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor='none',
                          edgecolor="black")
        self.ax.add_patch(ellipse)

        # Manual set points
        for track in self.manual_points:
            self.ax.scatter([p[0] for p in track], [p[1] for p in track], s=20, marker="x")
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
        return "{} -g <GAUSS_SIGMA> -h -i <INPUT_FILE> -n <N_PARTICLES> -o <OUTPUT_FILE> -s <SPEED> -v <VERBOSITY>\n".format(argv[0]) + \
               "-g: Sigma of Gaussian importance weight kernel.\n" + \
               "-h: This help.\n" + \
               "-i: Input file to parse with coordinates in WGS 84 system." + \
               "-n: Number of particles.\n" + \
               "-o: Output file to write manually set coordinates converted to WGS 84\n" + \
               "-s: Speed of the object.\n" + \
               "-v: Verbosity level. 0 = Silent [Default], >0 = increasing verbosity.\n"
    # end def

    inputfile = ""
    outputfile = "out.lst"
    n_particles = 100
    sigma = 20.
    speed = 1.
    verbosity = Logging.INFO

    try:
        opts, args = getopt.getopt(argv[1:], "g:hi:n:o:s:v")
    except getopt.GetoptError:
        print(usage())
        sys.exit(2)
    # end try

    for opt, arg in opts:
        if opt == '-h':
            print(usage())
            sys.exit()
        elif opt == "-i":
            inputfile = arg
        elif opt == "-n":
            n_particles = arg
        elif opt == "-o":
            outputfile = arg
        elif opt == "-g":
            sigma = arg
        elif opt == "-v":
            verbosity = arg
    # end for

    sim = Simulator(fn_in=inputfile, fn_out=outputfile, n_part=n_particles, s_gauss=sigma, speed=speed,
                    verbosity=verbosity)
    sim.run()


if __name__ == "__main__":
    main(sys.argv)
