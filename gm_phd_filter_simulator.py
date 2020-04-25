#!/usr/bin/env python3

from __future__ import annotations
from typing import List, Tuple, Optional
import os
import sys
import getopt
import random
import numpy as np
from datetime import datetime
from matplotlib.patches import Ellipse
import seaborn as sns

from filter_simulator.common import Logging, Limits, Position
from filter_simulator.filter_simulator import FilterSimulator
from gm_phd_filter import GmPhdFilter, GmComponent


class GmPhdFilterSimulator(FilterSimulator, GmPhdFilter):
    def __init__(self, fn_in: str, fn_out: str, limits: Limits, observer: Position, logging: Logging,
                 birth_gmm: List[GmComponent], p_survival: float, p_detection: float,
                 f: np.ndarray, q: np.ndarray, h: np.ndarray, r: np.ndarray, clutter: float,
                 trunc_thresh: float, merge_thresh: float, max_components: int):
        FilterSimulator.__init__(self, fn_in, fn_out, limits, observer, logging)
        GmPhdFilter.__init__(self, birth_gmm=birth_gmm, survival=p_survival, detection=p_detection, f=f, q=q, h=h, r=r, clutter=clutter, logging=logging)
        self.__trunc_thresh = trunc_thresh
        self.__merge_thresh = merge_thresh
        self.__max_components = max_components
        self.__logging: Logging = logging

    def _sim_loop_before_step_and_drawing(self):
        pass
        # XXX vllt. wie im partikelfilter manches vorher ausrechnen, da es sonst bereits für den nächsten Schritt upgedatet wird
    # end def

    def _sim_loop_after_step_and_drawing(self):
        # Set current frame
        self._cur_frame = self._frames[self._step]

        # Update
        self._update([np.array([det.x, det.y]) for det in self._cur_frame])

        # Prune
        self._prune(trunc_thresh=self.__trunc_thresh, merge_thresh=self.__merge_thresh, max_components=self.__max_components)

        # Predict als eigenen Step herausarbeiten, sofern was überhaupt geht - ist dies überhaupt sinnvoll?
    # end def

    def _calc_density(self, x: np.ndarray, y: np.ndarray) -> float:
        # XXX ähnlich zu eval_grid_2d bzw. daraus entnommen
        points = np.stack((x, y), axis=-1).reshape(-1, 2)

        vals = self._gmm.eval_list(points)  # XXX , which_dims)

        return np.array(vals).reshape(x.shape)

    def _calc_density_map(self, grid_res: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_: np.ndarray = np.linspace(self._det_borders.x_min, self._det_borders.x_max, grid_res)
        y_: np.ndarray = np.linspace(self._det_borders.y_min, self._det_borders.y_max, grid_res)

        x, y = np.meshgrid(x_, y_)
        z: np.ndarray = np.array(self._calc_density(x, y))

        return x, y, z

    @staticmethod
    def get_cov_ellipse(comp: GmComponent, n_std: float, **kwargs):
        return GmPhdFilterSimulator.get_cov_ellipse2(comp.cov, comp.loc, n_std, **kwargs)
    # end def

    @staticmethod
    def get_cov_ellipse2(cov, centre, nstd, **kwargs):
        """ Return a matplotlib Ellipse patch representing the covariance matrix
        cov centred at centre and scaled by the factor nstd. """
        # Find and sort eigenvalues and eigenvectors into descending order
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]

        # The anti-clockwise angle to rotate our ellipse by
        vx, vy = eigvecs[:, 0][0], eigvecs[:, 0][1]
        theta = np.arctan2(vy, vx)

        # Width and height of ellipse to draw
        width, height = 2 * nstd * np.sqrt(eigvals)
        return Ellipse(xy=centre, width=width, height=height, angle=float(np.degrees(theta)), **kwargs)

    # end def

    def _update_window(self) -> None:
        # Draw density map
        draw_kde: bool = True
        if not draw_kde:
            x, y, z = self._calc_density_map(grid_res=100)
            self._ax.contourf(x, y, z, 20, cmap='Blues')
        else:
            if len(self._gmm) > 0:
                samples = self._gmm.samples(1000)
                x = [s[0] for s in samples]
                y = [s[1] for s in samples]
                sns.kdeplot(x, y, shade=True, ax=self._ax)
            # end if
        # end if

        # All detections - each frame's detections in a different color
        for frame in self._frames:
            self._ax.scatter([det.x for det in frame], [det.y for det in frame], edgecolor="green", marker="o")
            self._ax.plot([det.x for det in frame], [det.y for det in frame], color="black", linewidth=.5,
                          linestyle="--")
        # end for

        # Estimated states
        est_items = self._extract_states(bias=1.)  # XXX params bias + use_integral
        self._ax.scatter([est_item[0] for est_item in est_items], [est_item[1] for est_item in est_items], s=200, c="gray", edgecolor="black", marker="o")

        # # Particles
        # self._ax.scatter([p.x for p in self._particles], [p.y for p in self._particles], s=5, edgecolor="blue",
        #                  marker="o")

        if self._cur_frame is not None:
            # Current detections
            det_pos_x: List[float] = [det.x for det in self._cur_frame]
            det_pos_y: List[float] = [det.y for det in self._cur_frame]
            self._ax.scatter(det_pos_x, det_pos_y, s=100, c="red", marker="x")

            # GM-PHD components covariance ellipses
            for comp in self._gmm:
                ell = self.get_cov_ellipse(comp, 1., facecolor='none', edgecolor="black", linewidth=.5)
                self._ax.add_patch(ell)
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

    # XXX Adapt parameters - first adopt the other code to find out, which parameters there are
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

    # XXX
    birth_comps = list()
    birth_comps.append(GmComponent(0.1, [0, 0], np.eye(2) * 2. ** 2))

    sim: GmPhdFilterSimulator = GmPhdFilterSimulator(fn_in=inputfile, fn_out=outputfile, limits=limits, observer=observer, logging=verbosity,
                                                     birth_gmm=birth_comps, p_survival=0.9, p_detection=0.9,
                                                     f=np.eye(2), q=np.eye(2) * 0., h=np.eye(2), r=np.eye(2) * .1, clutter=0.000002,
                                                     trunc_thresh=1e-6, merge_thresh=0.01, max_components=10)

    sim.run()


if __name__ == "__main__":
    main(sys.argv)
