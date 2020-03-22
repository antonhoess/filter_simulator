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
import copy

from filter_simulator.common import Logging, Limits, Position, Frame
from filter_simulator.filter_simulator import FilterSimulator


class WeightedDistribution:
    def __init__(self, state: List[Particle]) -> None:
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
    def __init__(self, x: float, y: float, w: float = 1.) -> None:
        self.x: float = x
        self.y: float = y
        self.vx: float = .0
        self.vy: float = .0
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
    def __init__(self, fn_in: str, fn_out: str, limits: Limits, n_part: int, s_gauss: float, noise: float, speed: float,
                 verbosity: Logging, observer: Position):
        super().__init__(fn_in, fn_out, verbosity, observer, limits)

        self.n_part: int = n_part
        self.s_gauss: float = s_gauss
        self.noise: float = noise
        self.speed: float = speed
        self.cur_frame: Optional[Frame] = None
        self.particles: List[Particle] = []
        self.m_confident: bool = False
        self.cluster_centers: Optional[np.ndarray] = None
        self.ms_bandwidth: float = .1  # XXX Param?
        self.use_speed: bool = True  # XXX Param? # If we use speed to update the particle's position, the particle might fly out of the curve, since their value of the gaussian kernel for calculating the importance weight will be less
        self.resampling: bool = True

    def processing(self) -> None:
        # Generate many particles
        self.particles = Particle.create_random(self.n_part, self.det_borders)

        # Simulation loop
        while True:
            # Calculate mean shift
            clustering: bool = True
            if clustering:
                cluster_samples: np.array = np.array([[p.x, p.y] for p in self.particles])
                clust: MeanShift = MeanShift(bandwidth=self.ms_bandwidth).fit(cluster_samples)
                self.cluster_centers: np.ndarray = clust.cluster_centers_
                self.logging.print_verbose(Logging.DEBUG, clust.labels_)
                self.logging.print_verbose(Logging.DEBUG, clust.cluster_centers_)
            # end if

            # Draw
            self.refresh.set()

            # Wait until drawing has finished (do avoid changing e.g. particles
            # before they are drawn in their current position)
            self.refresh_finished.wait()
            self.refresh_finished.clear()

            # Wait for a valid next step
            self.wait_for_valid_next_step()
            self.logging.print_verbose(Logging.INFO, "Step {}".format(self.step))

            # Set current frame
            self.cur_frame = self.frames[self.step]

            # 4.2 Update particle weight according to how good every particle matches the nearest detection
            for p in self.particles:
                # Use the detection position nearest to the current particle
                p_d_min: Optional[float] = None

                for det in self.cur_frame:
                    d_x: float = p.x - det.x
                    d_y: float = p.y - det.y
                    p_d: float = math.sqrt(d_x * d_x + d_y * d_y)

                    if p_d_min is None or p_d < p_d_min:
                        p_d_min = p_d
                    # end if
                # end for

                p.w = self.w_gauss(p_d_min, self.s_gauss)
            # end for

            # Resampling follows here:

            if self.resampling:
                new_particles: List[Particle] = []

                # Normalize weights
                nu: float = sum(p.w for p in self.particles)
                self.logging.print_verbose(Logging.DEBUG, "nu = {}".format(nu))

                if nu > 0:
                    for p in self.particles:
                        p.w = p.w / nu
                # end if

                # Create a weighted distribution, for fast picking
                dist: WeightedDistribution = WeightedDistribution(self.particles)
                self.logging.print_verbose(Logging.INFO, "# particles: {}".format(len(self.particles)))

                cnt: int = 0
                for _ in range(len(self.particles)):
                    p: Particle = dist.pick()

                    if p is None:  # No pick b/c all totally improbable
                        new_particle: Particle = Particle.create_random(1, self.det_borders)[0]
                        cnt += 1
                    else:
                        new_particle = copy.deepcopy(p)
                        new_particle.w = 1.
                    # end if

                    new_particle.move_by(*self.create_gaussian_noise(self.noise, 0, 0))
                    new_particles.append(new_particle)
                # end for

                self.logging.print_verbose(Logging.INFO, "# particles newly created: {}".format(cnt))
                self.particles = new_particles
            # end if

            # Move all particles according to belief of movement
            for p in self.particles:
                p_x = p.x
                p_y = p.y

                idx: Optional[int] = None
                p_d_min: float = .0

                # Determine detection nearest to the current particle
                for _idx, det in enumerate(self.cur_frame):
                    d_x: float = (det.x - p.x)
                    d_y: float = (det.y - p.y)
                    p_d: float = math.sqrt(d_x * d_x + d_y * d_y)

                    if idx is None or p_d < p_d_min:
                        idx = _idx
                        p_d_min = p_d
                    # end if
                # end for

                # Calc distance and andle to nearest detection
                det = self.cur_frame[idx]
                d_x: float = (det.x - p.x)
                d_y: float = (det.y - p.y)

                p_d: float = math.sqrt(d_x * d_x + d_y * d_y)
                angle: float = math.atan2(d_y, d_x)

                # Use a convex combination of...
                # .. the particles speed
                d_x = self.speed * p_d * math.cos(angle)
                d_y = self.speed * p_d * math.sin(angle)

                # .. and the 'speed', the particle progresses towards the new detection
                if self.use_speed:
                    d_x += (1 - self.speed) * p.vx
                    d_y += (1 - self.speed) * p.vy
                # end if

                # Move particle towards nearest detection
                p.move_by(d_x, d_y)

                # Calc particle speed for next time step
                p.vx = p.x - p_x
                p.vy = p.y - p_y

                # Add some noise for more stability
                p.move_by(*self.create_gaussian_noise(self.noise, 0, 0))
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
            self.ax.plot([det.x for det in frame], [det.y for det in frame], color="black", linewidth=.5,
                         linestyle="--")
        # end for

        # Mean shift centers
        if self.step >= 0:
            self.ax.scatter([cc[0] for cc in self.cluster_centers],
                            [cc[1] for cc in self.cluster_centers],
                            s=200, c="gray" if self.m_confident else "pink", edgecolor="black", marker="o")

        # Particles
        self.ax.scatter([p.x for p in self.particles], [p.y for p in self.particles], s=5, edgecolor="blue", marker="o")

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
    def create_noise(level: float, *coords) -> List[float]:
        return [x + random.uniform(-level, level) for x in coords]

    @staticmethod
    def create_gaussian_noise(level: float, *coords) -> List[float]:
        return [x + np.random.normal(.0, level) for x in coords]

    # Gaussian kernel to transform values near the particle => 1, further away => 0
    @staticmethod
    def w_gauss(x: float, sigma: float) -> float:
        g = math.e ** -(x * x / (2 * sigma * sigma))

        return g


def main(argv: List[str]):
    # Library settings
    sns.set(color_codes=True)

    # Initialize random generator
    random.seed(datetime.now())

    # Read command line arguments
    def usage() -> str:
        return "{} <PARAMETERS>\n".format(os.path.basename(argv[0])) + \
               "\n" + \
               "-g, --sigma_gauss_kernel=GAUSS_SIGMA:\n" + \
               "    Set sigma of Gaussian importance weight kernel to GAUSS_SIGMA.\n" + \
               "\n" + \
               "-h, --help: Prints this help.\n" + \
               "\n" + \
               "-i, --input=INPUT_FILE:\n" + \
               "    Parse detections with coordinates in WGS84 from INPUT_FILE.\n" + \
               "\n" + \
               "-l, --limits=LIMITS:\n" + \
               "    Sets the limits for the canvas to LIMITS. Its format is 'X_MIN;Y_MIN;X_MAX;Y_MAX'.\n" + \
               "\n" + \
               "-n, --number_of_particles=N_PARTICLES:\n" + \
               "    Sets the particle filter's number of particles to N_PARTICLES.\n" + \
               "\n" + \
               "-o, --output=OUTPUT_FILE:\n" \
               "    Sets the output file to store the manually set coordinates converted to WGS84 to OUTPUT_FILE.\n" + \
               "\n" + \
               "-p, --observer_position=OBSERVER_POSITION:\n" \
               "    Sets the position of the observer in WGS84 to OBSERVER_POSITION. Can be used instead of the center" \
               "of the detections or in case of only manually creating detections, which needed to be transformed back" \
               "to WGS84. Its format is 'X_POS;Y_POS'.\n" + \
               "\n" + \
               "-r, --particle_movement_noise=NOISE:\n" \
               "    Sets the particle's movement noise to NOISE.\n" + \
               "\n" + \
               "-s, --speed=SPEED:\n" \
               "    Sets the speed the particles move towards their nearest detection to SPEED.\n" + \
               "\n" + \
               "-v, --verbosity_level=VERBOSITY:\n" \
               "    Sets the programs verbosity level to VERBOSITY. 0 = Silent [Default], >0 = decreasing verbosity.\n"
    # end def

    inputfile: str = ""
    outputfile: str = "out.lst"
    limits: Limits = Limits(-10, -10, 10, 10)
    n_particles: int = 100
    sigma: float = 20.
    noise: float = .1
    speed: float = 1.
    verbosity: Logging = Logging.INFO
    observer: Optional[Position] = None

    try:
        opts, args = getopt.getopt(argv[1:], "g:hi:l:n:o:p:r:s:v:",
                                   ["sigma_gauss_kernel=", "--help", "input=", "limits=", "number_of_particles=",
                                    "output=", "observer_position=", "particle_movement_noise=", "speed=",
                                    "verbosity_level="])
    except getopt.GetoptError as e:
        print("Reading parameters caused error {}".format(e))
        print(usage())
        sys.exit(2)
    # end try

    for opt, arg in opts:
        if opt in ("-g", "--sigma_gauss_kernel"):
            sigma = float(arg)

        elif opt in ("-h", "--help"):
            print(usage())
            sys.exit()

        elif opt in ("-i", "--input"):
            inputfile = arg

        elif opt in ("-l", "--limits"):
            fields: List[str] = arg.split(";")
            if len(fields) == 4:
                limits = Limits(float(fields[0]), float(fields[1]), float(fields[2]), float(fields[3]))

        elif opt in ("-n", "--number_of_particles"):
            n_particles = int(arg)

        elif opt == ("-o", "--output"):
            outputfile = arg

        elif opt in ("-p", "--observer_position"):
            fields: List[str] = arg.split(";")
            if len(fields) >= 2:
                observer = Position(float(fields[0]), float(fields[1]))

        elif opt in ("-r", "--particle_movement_noise"):
            noise = float(arg)

        elif opt in ("-s", "--speed"):
            speed = max(min(float(arg), 1.), .0)

        elif opt in ("-v", "--verbosity_level"):
            verbosity = Logging(int(arg))
    # end for

    sim: ParticleFilterSimulator = ParticleFilterSimulator(fn_in=inputfile, fn_out=outputfile, limits=limits,
                                                           n_part=n_particles, s_gauss=sigma, speed=speed,
                                                           noise=noise, verbosity=verbosity, observer=observer)
    sim.run()


if __name__ == "__main__":
    main(sys.argv)
