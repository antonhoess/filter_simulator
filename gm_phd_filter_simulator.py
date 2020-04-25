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
from enum import Enum

from filter_simulator.common import Logging, Limits, Position
from filter_simulator.filter_simulator import FilterSimulator
from gm_phd_filter import GmPhdFilter, GmComponent, Gmm


class DensityDrawStyle(Enum):
    DRAW_KDE = 0
    DRAW_EVAL = 1
    DRAW_HEATMAP = 2
# end class


class GmPhdFilterSimulator(FilterSimulator, GmPhdFilter):
    def __init__(self, fn_in: str, fn_out: str, limits: Limits, observer: Position, logging: Logging,
                 birth_gmm: List[GmComponent], p_survival: float, p_detection: float,
                 f: np.ndarray, q: np.ndarray, h: np.ndarray, r: np.ndarray, clutter: float,
                 trunc_thresh: float, merge_thresh: float, max_components: int,
                 ext_states_bias: float, ext_states_use_integral: bool,
                 density_draw_style: DensityDrawStyle, n_samples_heatmap: int, n_bins_heatmap: int):
        FilterSimulator.__init__(self, fn_in, fn_out, limits, observer, logging)
        GmPhdFilter.__init__(self, birth_gmm=birth_gmm, survival=p_survival, detection=p_detection, f=f, q=q, h=h, r=r, clutter=clutter, logging=logging)
        self.__trunc_thresh = trunc_thresh
        self.__merge_thresh = merge_thresh
        self.__max_components = max_components
        self.__ext_states_bias = ext_states_bias
        self.__ext_states_use_integral = ext_states_use_integral
        self.__density_draw_style = density_draw_style
        self.__n_samples_heatmap = n_samples_heatmap
        self.__n_bins_heatmap = n_bins_heatmap
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

        # XXX Predict als eigenen Step herausarbeiten, sofern was überhaupt geht - ist dies überhaupt sinnvoll?
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
        if self.__density_draw_style == DensityDrawStyle.DRAW_KDE:
            if len(self._gmm) > 0:
                samples = self._gmm.samples(1000)
                x = [s[0] for s in samples]
                y = [s[1] for s in samples]
                sns.kdeplot(x, y, shade=True, ax=self._ax)
            # end if

        elif self.__density_draw_style == DensityDrawStyle.DRAW_EVAL:
            x, y, z = self._calc_density_map(grid_res=100)
            self._ax.contourf(x, y, z, 20, cmap='Blues')

        else:  # DensityDrawStyle.DRAW_HEATMAP
            n_samples_heatmap = self.__n_samples_heatmap
            n_bins_heatmap = self.__n_bins_heatmap
            samples = self._gmm.samples(n_samples_heatmap)
            det_borders = self._det_borders
            self._ax.hist2d([s[0] for s in samples], [s[1] for s in samples], bins=n_bins_heatmap,
                            range=[[det_borders.x_min, det_borders.x_max], [det_borders.y_min, det_borders.y_max]], density=False, cmap='plasma')
        # end if

        # All detections - each frame's detections in a different color
        for frame in self._frames:
            self._ax.scatter([det.x for det in frame], [det.y for det in frame], edgecolor="green", marker="o")
            self._ax.plot([det.x for det in frame], [det.y for det in frame], color="black", linewidth=.5,
                          linestyle="--")
        # end for

        # Estimated states
        est_items = self._extract_states(bias=self.__ext_states_bias, use_integral=self.__ext_states_use_integral)
        self._ax.scatter([est_item[0] for est_item in est_items], [est_item[1] for est_item in est_items], s=200, c="gray", edgecolor="black", marker="o")

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

            # GM-PHD components means
            self._ax.scatter([comp.loc[0] for comp in self._gmm], [comp.loc[1] for comp in self._gmm], s=5, edgecolor="blue", marker="o")
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
               "-h, --help: Prints this help.\n" + \
               "\n" + \
               "-i, --input=INPUT_FILE:\n" + \
               "    Parse detections with coordinates in WGS84 from INPUT_FILE.\n" + \
               "\n" + \
               "-o, --output=OUTPUT_FILE:\n" \
               "    Sets the output file to store the manually set coordinates converted to WGS84 to OUTPUT_FILE.\n" + \
               "\n" + \
               "-l, --limits=LIMITS:\n" + \
               "    Sets the limits for the canvas to LIMITS. Its format is 'Limits(X_MIN, Y_MIN, X_MAX Y_MAX)'.\n" + \
               "    Example: Limits(-10, -10, 10, 10)\n" + \
               "\n" + \
               "-v, --verbosity_level=VERBOSITY:\n" \
               "    Sets the programs verbosity level to VERBOSITY. 0 = Silent [Default], >0 = decreasing verbosity: 1 = CRITICAL, 2 = ERROR, 3 = WARNING, 4 = INFO, 5 = DEBUG.\n" \
               "\n" + \
               "-p, --observer_position=OBSERVER_POSITION:\n" \
               "    Sets the position of the observer in WGS84 to OBSERVER_POSITION. " \
               "Can be used instead of the center of the detections or in case of only manually creating detections, " \
               "which needed to be transformed back to WGS84. Its format is 'X_POS;Y_POS'.\n" + \
               "\n" + \
               "    --birth_gmm=BIRTH_GMM:\n" + \
               "    List ([]) of GmComponent which defines the birth-GMM. Format for a single GmComponent: GmComponent(weight, mean, covariance_matrix).\n" + \
               "    Example: [GmComponent(0.1, [0, 0], np.array([[5, 2], [2, 5]]))]\n" + \
               "\n" + \
               "    --p_survival=P_SURVIVAL:\n" + \
               "    Sets the survival probability for the PHD from time step k to k+1.\n" + \
               "\n" + \
               "    --p_detection=P_DETECTION:\n" + \
               "    Sets the (sensor's) detection probability for the measurements.\n" + \
               "\n" + \
               "    --mat_f=F:\n" + \
               "    Sets the transition matrix for the PHD.\n" + \
               "    Example: np.eye(2)\n" + \
               "\n" + \
               "    --mat_q=Q:\n" + \
               "    Sets the process noise covariance matrix.\n" + \
               "    Example: np.eye(2) * 0.\n" + \
               "\n" + \
               "    --mat_h=H:\n" + \
               "    Sets the measurement model.\n" + \
               "    Example: np.eye(2)\n" + \
               "\n" + \
               "    --mat_r=R:\n" + \
               "    Sets the measurement noise covariance matrix.\n" + \
               "    Example: np.eye(2) * .1\n" + \
               "\n" + \
               "    --clutter=CLUTTER:\n" + \
               "    Sets the amount of clutter.\n" + \
               "    Example: 2e-6\n" + \
               "\n" + \
               "    --trunc_thresh=TRUNC_THRESH:\n" + \
               "    Sets the truncation threshold for the prunging step. GM components with weights lower than this value get directly removed.\n" + \
               "    Example: 1e-6\n" + \
               "\n" + \
               "    --merge_thresh=MERGE_THRESH:\n" + \
               "    Sets the merge threshold for the prunging step. GM components with a Mahalanobis distance lower than this value get merged.\n" + \
               "    Example: 1e-2\n" + \
               "\n" + \
               "    --max_components=MAX_COMPONENTS:\n" + \
               "    Sets the max. number of Gm components used for the GMM representing the current PHD.\n" + \
               "    Example: 100\n" + \
               "\n" + \
               "    --ext_states_bias=EXT_STATES_BIAS:\n" + \
               "    Sets the bias for extracting the current states. It works as a factor for the GM component's weights and is used, " \
               "in case the weights are too small to reach a value higher than 0.5, which in needed to get extracted as a state.\n" + \
               "    Example: 1.\n" + \
               "\n" + \
               "    --ext_states_use_integral:\n" + \
               "    Defines if the integral approach for extracting the current states should be used.\n" + \
               "\n" + \
               "    --density_draw_style=DENSITY_DRAW_STYLE:\n" + \
               "    Sets the drawing style to visualizing the density/intensity map. Possible values are: DensityDrawStyle.DRAW_KDE (kernel density estimator), " \
               "DensityDrawStyle.DRAW_EVAL (evaluate the correct value for each cell in a grid) and DensityDrawStyle.DRAW_HEATMAP (heatmap made of sampled points from the PHD).\n" + \
               "    Example: DensityDrawStyle.DRAW_HEATMAP" + \
               "\n" + \
               "\n" + \
               "GUI\n" + \
               "    Mouse and keyboard events on the plotting window (GUI).\n" \
               "\n" + \
               "    There are two operating modes:\n" \
               "    * SIMULATION [Default]\n" \
               "    * MANUAL_EDITING\n" \
               "\n" + \
               "    To switch between these two modes, one needs to click (at least) three times with the LEFT mouse button while holding the CTRL and SHIFT buttons pressed without interruption. " \
               "Release the keyboard buttons to complete the mode switch.\n" \
               "\n" + \
               "    SIMULATION mode\n" \
               "        In the SIMULATION mode there are following commands:\n" + \
               "        * CTRL + RIGHT CLICK: Navigate forwards (load measurement data of the next time step).\n" + \
               "\n" + \
               "    MANUAL_EDITING mode\n" \
               "        In the MANUAL_EDITING mode there are following commands:\n" + \
               "        * CTRL + LEFT CLICK: Add point to current (time) frame.\n" + \
               "        * SHIFT + LEFT CLICK: Add frame.\n" + \
               "        * CTRL + RIGHT CLICK: Remove last set point.\n" + \
               "        * SHIFT + RIGHT CLICK: Remove last frame.\n" + \
               ""
    # end def

    inputfile: str = ""
    outputfile: str = "out.lst"
    limits: Limits = Limits(-10, -10, 10, 10)
    verbosity: Logging = Logging.INFO
    observer: Optional[Position] = None

    birth_gmm: Gmm = Gmm([GmComponent(0.1, [0, 0], np.eye(2) * 10. ** 2)])
    p_survival: float = 0.9
    p_detection: float = 0.9
    f: np.ndarray = np.eye(2)
    q: np.ndarray = np.eye(2) * 0.
    h: np.ndarray = np.eye(2)
    r: np.ndarray = np.eye(2) * .1
    clutter: float = 2e-6
    trunc_thresh: float = 1e-6
    merge_thresh: float = 0.01
    max_components: int = 10
    ext_states_bias: float = 1.
    ext_states_use_integral: bool = False
    density_draw_style: DensityDrawStyle = DensityDrawStyle.DRAW_HEATMAP
    n_samples_heatmap: int = 10000
    n_bins_heatmap: int = 100

    try:
        opts, args = getopt.getopt(argv[1:], "hi:l:o:p:v:", ["help", "input=", "limits=", "output=", "observer_position=", "verbosity_level=",
                                                             "birth_gmm=", "p_survival=", "p_detection=", "mat_f=", "mat_q=", "mat_h=", "mat_r=", "clutter=",
                                                             "trunc_thresh=", "merge_thresh=", "max_components=",
                                                             "ext_states_bias=", "ext_states_use_integral", "density_draw_style=", "n_samples_heatmap=", "n_bins_heatmap="])

    except getopt.GetoptError as e:
        print("Reading parameters caused error {}".format(e))
        print(usage())
        sys.exit(2)
    # end try

    for opt, arg in opts:
        err = False

        if opt in ("-h", "--help"):
            print(usage())
            sys.exit()

        elif opt in ("-i", "--input"):
            inputfile = arg

        elif opt in ("-l", "--limits"):
            fields: List[str] = arg.split(";")
            if len(fields) == 4:
                limits = Limits(float(fields[0]), float(fields[1]), float(fields[2]), float(fields[3]))

        elif opt == ("-o", "--output"):
            outputfile = arg

        elif opt in ("-p", "--observer_position"):
            fields: List[str] = arg.split(";")
            if len(fields) >= 2:
                observer = Position(float(fields[0]), float(fields[1]))

        elif opt in ("-v", "--verbosity_level"):
            verbosity = Logging(int(arg))

        elif opt == "--birth_gmm":
            birth_gmm = eval(arg)

            if isinstance(birth_gmm, list):
                for comp in birth_gmm:
                    if not isinstance(comp, GmComponent):
                        err = True
                        break
                    # end if
                # end for
            else:
                err = True
            # end if

        elif opt == "--p_survival":
            p_survival = float(arg)

        elif opt == "--p_detection":
            p_detection = float(arg)

        elif opt == "--mat_f":
            f = eval(arg)

            if not isinstance(f, np.ndarray):
                err = True
            # end if

        elif opt == "--mat_q":
            q = eval(arg)

            if not isinstance(q, np.ndarray):
                err = True
            # end if

        elif opt == "--mat_h":
            h = eval(arg)

            if not isinstance(h, np.ndarray):
                err = True
            # end if

        elif opt == "--mat_r":
            r = eval(arg)

            if not isinstance(r, np.ndarray):
                err = True
            # end if

        elif opt == "--clutter":
            clutter = float(arg)

        elif opt == "--trunc_thresh":
            trunc_thresh = float(arg)

        elif opt == "--merge_thresh":
            merge_thresh = float(arg)

        elif opt == "--max_components":
            max_components = int(arg)

        elif opt == "--ext_states_bias":
            ext_states_bias = float(arg)

        elif opt == "--ext_states_use_integral":
            ext_states_use_integral = True

        elif opt == "--density_draw_style":
            density_draw_style = eval(arg)

            if not isinstance(density_draw_style, DensityDrawStyle):
                err = True
            # end if

        elif opt == "--n_samples_heatmap":
            n_samples_heatmap = int(arg)

        elif opt == "--n_bins_heatmap":
            n_bins_heatmap = int(arg)
        # end if

        if err:
            print(f"Reading parameter \'{opt}\' caused an error. Argument not provided in correct format.")
            sys.exit(2)
        # end if
    # end for

    sim: GmPhdFilterSimulator = GmPhdFilterSimulator(fn_in=inputfile, fn_out=outputfile, limits=limits, observer=observer, logging=verbosity,
                                                     birth_gmm=birth_gmm, p_survival=p_survival, p_detection=p_detection,
                                                     f=f, q=q, h=h, r=r, clutter=clutter,
                                                     trunc_thresh=trunc_thresh, merge_thresh=merge_thresh, max_components=max_components,
                                                     ext_states_bias=ext_states_bias, ext_states_use_integral=ext_states_use_integral,
                                                     density_draw_style=density_draw_style, n_samples_heatmap=n_samples_heatmap, n_bins_heatmap=n_bins_heatmap)
    sim.run()


if __name__ == "__main__":
    main(sys.argv)
