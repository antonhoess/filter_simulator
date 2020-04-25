#!/usr/bin/env python3

from __future__ import annotations
from typing import List, Tuple, Optional
import os
import sys
import getopt
import random
import numpy as np
from datetime import datetime
from sklearn.cluster import MeanShift
from matplotlib.patches import Ellipse
import seaborn as sns

from filter_simulator.common import Logging, Limits, Position
from filter_simulator.filter_simulator import FilterSimulator
from particle_filter import ParticleFilter


class ParticleFilterSimulator(FilterSimulator, ParticleFilter):
    def __init__(self, fn_in: str, fn_out: str, limits: Limits, observer: Position, logging: Logging, n_part: int, s_gauss: float, noise: float, speed: float):
        FilterSimulator.__init__(self, fn_in, fn_out, limits, observer, logging)
        ParticleFilter.__init__(self, n_part, s_gauss, noise, speed, limits, logging)

        self.__m_confident: bool = False
        self.__cluster_centers: Optional[np.ndarray] = None
        self.__ms_bandwidth: float = .1  # XXX Param?
        self.__logging: Logging = logging

    def _sim_loop_before_step_and_drawing(self):
        # Calculate mean shift
        clustering: bool = True
        if clustering:
            cluster_samples: np.array = np.array([[p.x, p.y] for p in self._particles])
            clust: MeanShift = MeanShift(bandwidth=self.__ms_bandwidth).fit(cluster_samples)
            self.__cluster_centers: np.ndarray = clust.cluster_centers_
            self.__logging.print_verbose(Logging.DEBUG, clust.labels_)
            self.__logging.print_verbose(Logging.DEBUG, clust.cluster_centers_)
        # end if
    # end def

    def _sim_loop_after_step_and_drawing(self):
        # Set current frame
        self._cur_frame = self._frames[self._step]

        # Update
        self._update()

        # Resample
        self._resample()

        # Predict
        self._predict()
    # end def

    def _calc_density(self, x: np.ndarray, y: np.ndarray) -> float:
        accum: float = 0.

        for p in self._particles:
            d_x: float = p.x - x
            d_y: float = p.y - y
            p_d: float = 1. / np.sqrt(d_x * d_x + d_y * d_y)
            accum += p_d
        # end for

        return accum

    def _calc_density_map(self, grid_res: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_: np.ndarray = np.linspace(self._det_borders.x_min, self._det_borders.x_max, grid_res)
        y_: np.ndarray = np.linspace(self._det_borders.y_min, self._det_borders.y_max, grid_res)

        x, y = np.meshgrid(x_, y_)
        z: np.ndarray = np.array(self._calc_density(x, y))

        return x, y, z

    def _update_window(self) -> None:
        print(".", end="")  # XXX
        # Draw density map
        draw_kde: bool = True  # XXX parameter
        if not draw_kde:
            x, y, z = self._calc_density_map(grid_res=100)
            self._ax.contourf(x, y, z, 20, cmap='Blues')
        else:
            x = [p.x for p in self._particles]
            y = [p.y for p in self._particles]
            sns.kdeplot(x, y, shade=True, ax=self._ax)
        # end if

        # All detections - each frame's detections in a different color
        for frame in self._frames:
            self._ax.scatter([det.x for det in frame], [det.y for det in frame], edgecolor="green", marker="o")
            self._ax.plot([det.x for det in frame], [det.y for det in frame], color="black", linewidth=.5,
                          linestyle="--")
        # end for

        # Mean shift centers
        if self._step >= 0:
            self._ax.scatter([cc[0] for cc in self.__cluster_centers],
                             [cc[1] for cc in self.__cluster_centers],
                             s=200, c="gray" if self.__m_confident else "pink", edgecolor="black", marker="o")

        # Particles
        self._ax.scatter([p.x for p in self._particles], [p.y for p in self._particles], s=5, edgecolor="blue",
                         marker="o")

        if self._cur_frame is not None:
            # Current detections
            det_pos_x: List[float] = [det.x for det in self._cur_frame]
            det_pos_y: List[float] = [det.y for det in self._cur_frame]
            self._ax.scatter(det_pos_x, det_pos_y, s=100, c="red", marker="x")

            # Importance weight Gaussian-kernel covariance ellipse
            ell_radius_x: float = self._s_gauss
            ell_radius_y: float = self._s_gauss

            for det in self._cur_frame:
                ellipse: Ellipse = Ellipse((det.x, det.y), width=ell_radius_x * 2,
                                           height=ell_radius_y * 2, facecolor='none', edgecolor="black", linewidth=.5)
                self._ax.add_patch(ellipse)
            # end for
        # end if
    # end def

    def _cb_keyboard(self, cmd: str) -> None:
        if cmd == "":
            self._next = True

        elif cmd == "+":
            pass  # XXX

        elif cmd.startswith("-"):
            pass
            # XXX idx: int = int(cmd[1:])
        # end if
    # end def


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
               "    Sets the position of the observer in WGS84 to OBSERVER_POSITION." \
               "Can be used instead of the center of the detections or in case of only manually creating detections," \
               "which needed to be transformed back" \
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
                                   ["sigma_gauss_kernel=", "help", "input=", "limits=", "number_of_particles=",
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

    sim: ParticleFilterSimulator = ParticleFilterSimulator(fn_in=inputfile, fn_out=outputfile, limits=limits, observer=observer, logging=verbosity,
                                                           n_part=n_particles, s_gauss=sigma, speed=speed, noise=noise)
    sim.run()


if __name__ == "__main__":
    main(sys.argv)
