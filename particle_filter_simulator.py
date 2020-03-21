#!/usr/bin/env python3

from __future__ import annotations
from typing import List, Tuple, Optional
import os
import sys
import getopt
import math
import bisect
import random
import numpy as np
from datetime import datetime
from sklearn.cluster import MeanShift
from matplotlib.patches import Ellipse
import seaborn as sns

from filter_simulator.common import Logging, Limits, Position, Frame
from filter_simulator.filter_simulator import FilterSimulator


class WeightedDistribution:
    def __init__(self, state) -> None:
        accum: float = .0
        self.state: List[Particle] = [p for p in state if p.w > 0]
        self.distribution: List[float] = []

        for x in self.state:
            accum += x.w
            self.distribution.append(accum)

    def pick(self) -> Optional[Particle]:
        try:
            # Due to numeric problems, the weight don't sum up to 1.0 after normalization,
            # so we can't pick from a uniform distribution in range [0, 1]
            return self.state[bisect.bisect_left(self.distribution, random.uniform(0, self.distribution[-1]))]
        except IndexError:
            # Happens when all particles are improbable w=0
            return None


class Particle:
    def __init__(self, x: float, y: float, w: float = 1., noisy: bool = False) -> None:
        if noisy:
            x, y = ParticleFilterSimulator.add_some_noise(x, y)

        self.x: float = x
        self.y: float = y
        self.w: float = w

    def __repr__(self) -> str:
        return "(%f, %f, w=%f)" % (self.x, self.y, self.w)

    @property
    def xy(self) -> Tuple[float, float]:
        return self.x, self.y

    @classmethod
    def create_random(cls, count: int, limits: Limits) -> List[Particle]:
        return [cls(random.uniform(limits.x_min, limits.x_max), random.uniform(limits.y_min, limits.y_max))
                for _ in range(0, count)]

    def move_by(self, x, y) -> None:
        self.x += x
        self.y += y


class ParticleFilterSimulator(FilterSimulator):
    def __init__(self, fn_in: str, fn_out: str, limits: Limits, n_part: int, s_gauss: float, speed: float,
                 verbosity: Logging, observer: Position):
        super().__init__(fn_in, fn_out, verbosity, observer, limits)

        self.n_part: int = n_part
        self.s_gauss: float = s_gauss
        self.speed: float = speed
        self.cur_frame: Optional[Frame] = None
        self.particles: List[Particle] = []
        self.mean: Optional[Position] = None
        self.m_confident: bool = False
        self.cluster_centers: Optional[np.ndarray] = None

    def processing(self) -> None:
        # 2. Generate many particles
        self.particles = Particle.create_random(self.n_part, self.det_borders)

        # 4. Simulation loop
        while True:
            # Calculate mean shift
            clustering: bool = True
            if clustering:
                cluster_samples: np.array = np.array([[p.x, p.y] for p in self.particles])
                clust: MeanShift = MeanShift(bandwidth=10).fit(cluster_samples)
                self.cluster_centers: np.ndarray = clust.cluster_centers_
                self.logging.print_verbose(Logging.DEBUG, clust.labels_)
                self.logging.print_verbose(Logging.DEBUG, clust.cluster_centers_)
            # end if

            # 4.3 Compute weighted mean of particles (gray circle)
            self.mean, self.m_confident = self.compute_mean_point()

            # Wait until drawing has finished (do avoid changing e.g. particles
            # before they are drawn in their current position)

            self.refresh.set()

            self.refresh_finished.wait()
            self.refresh_finished.clear()

            # Wait for a valid next step
            self.wait_for_valid_next_step()
            self.logging.print_verbose(Logging.INFO, "Step {}".format(self.step))

            # Set current frame
            self.cur_frame = self.frames[self.step]

            # 4.2 Update particle weight according to how good every particle matches
            #     Robbie's sensor reading
            for p in self.particles:
                w_total: float = .0

                for det in self.cur_frame:
                    # get distance of particle to nearest beacon
                    d_x: float = p.x - det.x
                    d_y: float = p.y - det.y
                    p_d: float = math.sqrt(d_x * d_x + d_y * d_y)
                    w_total += self.w_gauss(p_d, self.s_gauss)
                # end for

                n_vec: int = len(self.cur_frame)

                if n_vec > 0:
                    w_total /= n_vec

                p.w = w_total
            # end for

            # 4.5 Resampling follows here:
            resampling: bool = True
            if resampling:
                new_particles: List[Particle] = []

                # 4.5.1 Normalise weights
                nu: float = sum(p.w for p in self.particles)
                self.logging.print_verbose(Logging.DEBUG, "nu = {}".format(nu))

                if nu:
                    for p in self.particles:
                        p.w = p.w / nu
                # end if

                # 4.5.2 create a weighted distribution, for fast picking
                dist: WeightedDistribution = WeightedDistribution(self.particles)
                self.logging.print_verbose(Logging.INFO, "# particles: {}".format(len(self.particles)))

                cnt: int = 0
                for _ in range(len(self.particles)):
                    p: Particle = dist.pick()

                    if p is None:  # No pick b/c all totally improbable
                        new_particle: Particle = Particle.create_random(1, self.det_borders)[0]
                        cnt += 1
                    else:
                        new_particle = Particle(p.x, p.y, 1.)
                    # end if

                    x, y = self.add_noise(.1, 0, 0)
                    new_particle.move_by(x, y)

                    new_particles.append(new_particle)
                # end for

                self.logging.print_verbose(Logging.INFO, "# particles newly created: {}".format(cnt))
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
            d_x: float = .0
            d_y: float = .0
            for p in self.particles:

                # Add all vectors
                for det in self.cur_frame:
                    d_x += (det.x - p.x)
                    d_y += (det.y - p.y)
                # end for

                # Calculate resulting vector
                n_vec: int = len(self.cur_frame)
                if n_vec > 0:
                    d_x = d_x / n_vec
                    d_y = d_y / n_vec
                # end if

                p_d: float = math.sqrt(d_x * d_x + d_y * d_y)
                angle: float = math.atan2(d_y, d_x)
                # p_d = 1.0  # XXX
                # p.x += self.speed * p_d * math.cos(angle)
                # p.y += self.speed * p_d * math.sin(angle)
                p.move_by(min(self.speed, p_d) * math.cos(angle), min(self.speed, p_d) * math.sin(angle))
            # end for
        # end while
    # end def

    def calc_density(self, x: float, y: float) -> float:
        accum: float = 0.

        for p in self.particles:
            d_x: float = p.x - x
            d_y: float = p.y - y
            p_d: float = 1. / np.sqrt(d_x * d_x + d_y * d_y)
            accum += p_d
        # end for

        return accum

    def calc_density_map(self, grid_res: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_: np.ndarray = np.linspace(self.det_borders.x_min, self.det_borders.x_max, grid_res)
        y_: np.ndarray = np.linspace(self.det_borders.y_min, self.det_borders.y_max, grid_res)

        x, y = np.meshgrid(x_, y_)
        z: np.ndarray = np.array(self.calc_density(x, y))

        return x, y, z

    def update_window(self) -> None:
        # Draw density map
        draw_kde: bool = True
        if not draw_kde:
            x, y, z = self.calc_density_map(grid_res=100)
            self.ax.contourf(x, y, z, 20, cmap='Blues')
        else:
            x = [p.x for p in self.particles]
            y = [p.y for p in self.particles]
            sns.kdeplot(x, y, shade=True, ax=self.ax)
        # end if

        # All detections - each frame's detections in a different color
        for frame in self.frames:
            self.ax.scatter([det.x for det in frame], [det.y for det in frame], edgecolor="green", marker="o")
        # end for

        # Weighted mean
        self.ax.scatter([self.mean.x], [self.mean.y], s=200,
                        c="gray" if self.m_confident else "pink", edgecolor="black", marker="o")

        # Particles
        self.ax.scatter([p.x for p in self.particles], [p.y for p in self.particles], s=5, edgecolor="blue", marker="o")

        # Mean shift centers
        if hasattr(self, 'cluster_centers_'):
            self.ax.scatter([cc[0] for cc in self.cluster_centers],
                            [cc[1] for cc in self.cluster_centers], s=25, edgecolor="orange", marker="x")

        if self.cur_frame is not None:
            # Current detections
            det_pos_x: List[float] = [det.x for det in self.cur_frame]
            det_pos_y: List[float] = [det.y for det in self.cur_frame]
            self.ax.scatter(det_pos_x, det_pos_y, s=100, c="red", marker="x")

            # Importance weight Gaussian-kernel covariance ellipse
            ell_radius_x: float = self.s_gauss
            ell_radius_y: float = self.s_gauss

            for det in self.cur_frame:
                ellipse: Ellipse = Ellipse((det.x, det.y), width=ell_radius_x * 2,
                                           height=ell_radius_y * 2, facecolor='none', edgecolor="black", linewidth=.5)
                self.ax.add_patch(ellipse)
            # end for
        # end if
    # end def

    def cb_keyboard(self, cmd: str) -> None:
        if cmd == "":
            self.next = True

        elif cmd == "+":
            pass  # XXX

        elif cmd.startswith("-"):
            pass
            # XXX idx: int = int(cmd[1:])
        # end if
    # end def

    @staticmethod
    def add_noise(level: float, *coords) -> List[float]:
        return [x + random.uniform(-level, level) for x in coords]

    @staticmethod
    def add_some_noise(*coords) -> List[float]:
        return ParticleFilterSimulator.add_noise(0.1, *coords)

    # This is just a gaussian kernel I pulled out of my hat, to transform
    # values near to robbie's measurement => 1, further away => 0
    @staticmethod
    def w_gauss(x: float, sigma: float) -> float:
        g = math.e ** -(x * x / (2 * sigma * sigma))

        return g

    def compute_mean_point(self) -> Tuple[Position, int]:
        """
        Compute the mean for all particles that have a reasonably good weight.
        This is not part of the particle filter algorithm but rather an
        addition to show the "best belief" for current position.
        """

        m_x: float = 0
        m_y: float = 0
        m_count: int = 0

        for p in self.particles:
            m_count += p.w
            m_x += p.x * p.w
            m_y += p.y * p.w

        if m_count == 0:
            return Position(-1, -1), False

        m_x /= m_count
        m_y /= m_count

        # Now compute how good that mean is - check how many particles actually are in the immediate vicinity
        m_count = 0

        for p in self.particles:
            d_x = p.x - m_x
            d_y = p.y - m_y
            p_d = math.sqrt(d_x * d_x + d_y * d_y)

            if p_d < 3:  # xxx param
                m_count += 1

        return Position(m_x, m_y), m_count > len(self.particles) * 0.95  # xxx param
    # end def
# end class


def main(argv: List[str]):
    # Library settings
    sns.set(color_codes=True)

    # Initialize random generator
    random.seed(datetime.now())

    # Read command line arguments
    def usage() -> str:
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
               "    Position of the observer in WGS84. Can be used instead of the center of the detections or in case" \
               "of only manually creating detections, which needed to be transformed back to WGS84.\n" + \
               "\n" + \
               "-s <SPEED>:\n" \
               "    Speed of the object.\n" + \
               "\n" + \
               "-v <VERBOSITY>:\n" \
               "    Verbosity level. 0 = Silent [Default], >0 = decreasing verbosity.\n"
    # end def

    inputfile: str = ""
    outputfile: str = "out.lst"
    limits: Limits = Limits(-10, -10, 10, 10)
    n_particles: int = 100
    sigma: float = 20.
    speed: float = 1.
    verbosity: Logging = Logging.INFO
    observer: Optional[Position] = None

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
            fields: List[str] = arg.split(";")
            if len(fields) == 4:
                limits = Limits(float(fields[0]), float(fields[1]), float(fields[2]), float(fields[3]))

        elif opt == "-n":
            n_particles = int(arg)

        elif opt == "-o":
            outputfile = arg

        elif opt == "-p":
            fields: List[str] = arg.split(";")
            if len(fields) >= 2:
                observer = Position(float(fields[0]), float(fields[1]))

        elif opt == "-v":
            verbosity = Logging(arg)
    # end for

    sim: ParticleFilterSimulator = ParticleFilterSimulator(fn_in=inputfile, fn_out=outputfile, limits=limits,
                                                           n_part=n_particles, s_gauss=sigma,
                                                           speed=speed, verbosity=verbosity, observer=observer)
    sim.run()


if __name__ == "__main__":
    main(sys.argv)
