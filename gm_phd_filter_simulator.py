#!/usr/bin/env python3

from __future__ import annotations
from typing import Sequence, List, Tuple, Optional
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
from filter_simulator.io_helper import FileReader, InputLineHandlerLatLonIdx
from filter_simulator.filter_simulator import FilterSimulator, SimStepPartConf
from filter_simulator.data_provider_interface import IDataProvider
from filter_simulator.data_provider_converter import Wgs84ToEnuConverter
from filter_simulator.dyn_matrix import TransitionModel, PcwConstWhiteAccelModelNd
from gm_phd_filter import GmPhdFilter, GmComponent, Gmm
from phd_filter_data_provider import PhdFilterDataProvider


class DrawLayer(Enum):
    DENSITY_MAP = 0
    ALL_DET = 1
    ALL_DET_CONN = 2
    GMM_COV_ELL = 3
    GMM_COV_MEAN = 4
    EST_STATE = 5
    CUR_DET = 6
# end class


class DensityDrawStyle(Enum):
    KDE = 0
    EVAL = 1
    HEATMAP = 2
# end class


class DataProviderType(Enum):
    FILE_READER = 0
    SIMULATOR = 1
# end class


class CoordSysConv(Enum):
    NONE = 0
    WGS84 = 1
# end class


class GmPhdFilterSimulator(FilterSimulator, GmPhdFilter):
    def __init__(self, data_provider: IDataProvider, fn_out: str, limits: Limits, observer: Position, logging: Logging,
                 birth_gmm: List[GmComponent], p_survival: float, p_detection: float,
                 f: np.ndarray, q: np.ndarray, h: np.ndarray, r: np.ndarray, clutter: float,
                 trunc_thresh: float, merge_thresh: float, max_components: int,
                 ext_states_bias: float, ext_states_use_integral: bool,
                 density_draw_style: DensityDrawStyle, n_samples_density_map: int, n_bins_density_map: int,
                 draw_layers: Optional[List[DrawLayer]]):
        FilterSimulator.__init__(self, data_provider, fn_out, limits, observer, logging)
        GmPhdFilter.__init__(self, birth_gmm=birth_gmm, survival=p_survival, detection=p_detection, f=f, q=q, h=h, r=r, clutter=clutter, logging=logging)

        self.__trunc_thresh = trunc_thresh
        self.__merge_thresh = merge_thresh
        self.__max_components = max_components
        self.__ext_states_bias = ext_states_bias
        self.__ext_states_use_integral = ext_states_use_integral
        self.__density_draw_style = density_draw_style
        self.__n_samples_density_map = n_samples_density_map
        self.__n_bins_density_map = n_bins_density_map
        self.__logging: Logging = logging
        self.__draw_layers: Optional[List[DrawLayer]] = draw_layers if draw_layers is not None else [ly for ly in DrawLayer]

    def _set_sim_loop_step_part_conf(self):
        # Configure the processing steps
        sim_step_part_conf = SimStepPartConf()

        sim_step_part_conf.add_user_step(self.__sim_loop_predict_and_update)
        # sim_step_part_conf.add_draw_step()
        # sim_step_part_conf.add_wait_for_trigger_step()
        sim_step_part_conf.add_user_step(self.__sim_loop_prune)
        sim_step_part_conf.add_draw_step()
        sim_step_part_conf.add_wait_for_trigger_step()
        sim_step_part_conf.add_load_next_frame_step()

        return sim_step_part_conf
    # end def

    def __sim_loop_predict_and_update(self):
        if self._step < 0:
            self._predict_and_update([])

        else:
            # Set current frame
            self._cur_frame = self._frames[self._step]

            # Predict and update
            self._predict_and_update([np.array([det.x, det.y]) for det in self._cur_frame])
        # end if
    # end def

    def __sim_loop_prune(self):
        if self._step >= 0:
            # Prune
            self._prune(trunc_thresh=self.__trunc_thresh, merge_thresh=self.__merge_thresh, max_components=self.__max_components)
        # end if
    # end def

    def __calc_density(self, x: np.ndarray, y: np.ndarray) -> float:
        # Code taken from eval_grid_2d()
        points = np.stack((x, y), axis=-1).reshape(-1, 2)

        vals = self._gmm.eval_list(points, which_dims=(0, 1))

        return np.array(vals).reshape(x.shape)

    def __calc_density_map(self, grid_res: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_: np.ndarray = np.linspace(self._det_borders.x_min, self._det_borders.x_max, grid_res)
        y_: np.ndarray = np.linspace(self._det_borders.y_min, self._det_borders.y_max, grid_res)

        x, y = np.meshgrid(x_, y_)
        z: np.ndarray = np.array(self.__calc_density(x, y))

        return x, y, z

    @staticmethod
    def __get_cov_ellipse_from_comp(comp: GmComponent, n_std: float, which_dims: Sequence[int] = (0, 1), **kwargs):
        which_dims = list(which_dims)
        comp = comp.get_with_reduced_dims(which_dims)
        return GmPhdFilterSimulator.__get_cov_ellipse(comp.cov, comp.loc, n_std, **kwargs)
    # end def

    @staticmethod
    def __get_cov_ellipse(cov, centre, n_std, **kwargs):
        """ Return a matplotlib Ellipse patch representing the covariance matrix
        cov centred at centre and scaled by the factor n_std. """
        # Find and sort eigenvalues and eigenvectors into descending order
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        order = eig_vals.argsort()[::-1]
        eig_vals, eig_vecs = eig_vals[order], eig_vecs[:, order]

        # The anti-clockwise angle to rotate our ellipse by
        vx, vy = eig_vecs[:, 0][0], eig_vecs[:, 0][1]
        theta = np.arctan2(vy, vx)

        # Width and height of ellipse to draw
        width, height = 2 * n_std * np.sqrt(eig_vals)
        return Ellipse(xy=centre, width=width, height=height, angle=float(np.degrees(theta)), **kwargs)

    # end def

    def _update_window(self) -> None:

        for l, ly in enumerate(self.__draw_layers):
            zorder = l

            if ly == DrawLayer.DENSITY_MAP:
                # cmap = "Greys"
                # cmap = "plasma"
                cmap = "Blues"
                # Draw density map
                if self.__density_draw_style == DensityDrawStyle.KDE:
                    if len(self._gmm) > 0:
                        samples = self._gmm.samples(self.__n_samples_density_map)
                        x = [s[0] for s in samples]
                        y = [s[1] for s in samples]
                        sns.kdeplot(x, y, shade=True, ax=self._ax, shade_lowest=False, cmap=cmap, zorder=zorder)
                    # end if

                elif self.__density_draw_style == DensityDrawStyle.EVAL:
                    x, y, z = self.__calc_density_map(grid_res=self.__n_bins_density_map)
                    self._ax.contourf(x, y, z, 20, cmap=cmap)

                else:  # DensityDrawStyle.DRAW_HEATMAP
                    samples = self._gmm.samples(self.__n_samples_density_map)
                    det_borders = self._det_borders
                    self._ax.hist2d([s[0] for s in samples], [s[1] for s in samples], bins=self.__n_bins_density_map,
                                    range=[[det_borders.x_min, det_borders.x_max], [det_borders.y_min, det_borders.y_max]], density=False, cmap=cmap, zorder=zorder)
                # end if

            elif ly == DrawLayer.ALL_DET:
                # All detections - each frame's detections in a different color
                for frame in self._frames:
                    self._ax.scatter([det.x for det in frame], [det.y for det in frame], edgecolor="green", marker="o", zorder=zorder)
                # end for

            elif ly == DrawLayer.ALL_DET_CONN:
                # Connections between all detections - only makes sense, if they are manually created or created in a very ordered way, otherwise it's just chaos
                for frame in self._frames:
                    self._ax.plot([det.x for det in frame], [det.y for det in frame], color="black", linewidth=.5, linestyle="--", zorder=zorder)
                # end for

            elif ly == DrawLayer.GMM_COV_ELL:
                if self._cur_frame is not None:
                    # GM-PHD components covariance ellipses
                    for comp in self._gmm:
                        ell = self.__get_cov_ellipse_from_comp(comp, 1., facecolor='none', edgecolor="black", linewidth=.5, zorder=zorder)
                        self._ax.add_patch(ell)
                    # end for
                # end if

            elif ly == DrawLayer.GMM_COV_MEAN:
                if self._cur_frame is not None:
                    # GM-PHD components means
                    self._ax.scatter([comp.loc[0] for comp in self._gmm], [comp.loc[1] for comp in self._gmm], s=5, edgecolor="blue", marker="o", zorder=zorder)
                # end if

            elif ly == DrawLayer.EST_STATE:
                # Estimated states
                est_items = self._extract_states(bias=self.__ext_states_bias, use_integral=self.__ext_states_use_integral)
                self._ax.scatter([est_item[0] for est_item in est_items], [est_item[1] for est_item in est_items], s=200, c="gray", edgecolor="black", marker="o", zorder=zorder)

            elif ly == DrawLayer.CUR_DET:
                if self._cur_frame is not None:
                    # Current detections
                    det_pos_x: List[float] = [det.x for det in self._cur_frame]
                    det_pos_y: List[float] = [det.y for det in self._cur_frame]
                    self._ax.scatter(det_pos_x, det_pos_y, s=100, c="red", marker="x", zorder=zorder)
                # end if
            # end if
        # end for
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
               "    --data_provider=DATA_PROVIDER:\n" + \
               "       Sets the data provider type that defines the data source. Possible values are: DataProviderType.FILE_READER (reads lines from file defined in --input_file), " \
               "DataProviderType.SIMULATOR (simulates the PHD behaviour defined by the paraemters given in section SIMULATOR).\n" + \
               "    Example: DensityDrawStyle.DRAW_HEATMAP" + \
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
               "    Sets the drawing style to visualizing the density/intensity map. Possible values are: DensityDrawStyle.KDE (kernel density estimator), " \
               "DensityDrawStyle.EVAL (evaluate the correct value for each cell in a grid) and DensityDrawStyle.HEATMAP (heatmap made of sampled points from the PHD).\n" + \
               "    Example: DensityDrawStyle.DRAW_HEATMAP" + \
               "\n" + \
               "    --n_samples_density_map=N_SAMPLES_DENSITY_MAP:\n" + \
               "    Sets the number samples to draw from the PHD for drawing the density map.\n" + \
               "    Example: 10000\n" + \
               "\n" + \
               "    --n_bins_density_map=N_BINS_DENSITY_MAP:\n" + \
               "    Sets the number bins for drawing the PHD density map.\n" + \
               "    Example: 100\n" + \
               "\n" + \
               "    --draw_layers=DRAW_LAYERS:\n" + \
               "    Sets the list of drawing layers. Allows to draw only the required layers and in the desired order. As default all layers are drawn in a fixed order.\n" + \
               "    Example 1: [DrawLayer.DENSITY_MAP, DrawLayer.EST_STATE]\n" \
               "    Example 2: [layer for layer in DrawLayer if not layer == DrawLayer.GMM_COV_ELL and not layer == DrawLayer.GMM_COV_MEAN]\n" + \
               "\n" + \
               "\n" + \
               "FILE READER\n" + \
               "    Reads detections from file.\n" \
               "\n" + \
               "-i, --input=INPUT_FILE:\n" + \
               "    Parse detections with coordinates from INPUT_FILE.\n" + \
               "\n" + \
               "    --coord_system_conversion=COORD_SYSTEM_CONVERSION:\n" + \
               "    Defines the conversion of the parsed coordinates into another coordinate system.\n" + \
               "\n" + \
               "\n" + \
               "SIMULATOR\n" + \
               "    Calculates detections from simulation.\n" \
               "XXX\n" \
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
               "\n" + \
               ""
    # end def

    data_provider_type: DataProviderType = DataProviderType.FILE_READER

    outputfile: str = "out.lst"
    limits: Limits = Limits(-10, -10, 10, 10)
    verbosity: Logging = Logging.INFO
    observer: Optional[Position] = None

    birth_gmm: Gmm = Gmm([GmComponent(0.1, [0, 0], np.eye(2) * 10. ** 2)])
    p_survival: float = 0.9
    p_detection: float = 0.9
    model = TransitionModel.INDIVIDUAL  # XXX in parameter aufnehmen
    dt = 1.  # XXX auch als param aufnehmen
    f: np.ndarray = np.eye(2)
    q: np.ndarray = np.eye(2) * 0.
    h: np.ndarray = np.eye(2)
    r: np.ndarray = np.eye(2) * .1
    sigma_vel_x = .5  # XXX in parameter aufnehmen
    sigma_vel_y = .5  # XXX in parameter aufnehmen
    clutter: float = 2e-6
    trunc_thresh: float = 1e-6
    merge_thresh: float = 0.01
    max_components: int = 10
    ext_states_bias: float = 1.
    ext_states_use_integral: bool = False
    density_draw_style: DensityDrawStyle = DensityDrawStyle.HEATMAP
    n_samples_density_map: int = 10000
    n_bins_density_map: int = 100
    draw_layers: Optional[List[DrawLayer]] = None

    inputfile: str = ""
    coord_system_conversion: CoordSysConv = CoordSysConv.NONE

    try:
        opts, args = getopt.getopt(argv[1:], "hi:l:o:p:v:", ["help", "data_provider=", "limits=", "output=", "observer_position=", "verbosity_level=",
                                                             "birth_gmm=", "p_survival=", "p_detection=", "mat_f=", "mat_q=", "mat_h=", "mat_r=", "clutter=",
                                                             "trunc_thresh=", "merge_thresh=", "max_components=",
                                                             "ext_states_bias=", "ext_states_use_integral", "density_draw_style=", "n_samples_density_map=", "n_bins_density_map=", "draw_layers=",
                                                             "input=", "coord_system_conversion="])

    except getopt.GetoptError as e:
        print("Reading parameters caused error {}".format(e))
        print(usage())
        sys.exit(2)
    # end try

    for opt, arg in opts:
        err: bool = False
        err_msg: Optional[str] = None

        if opt in ("-h", "--help"):
            print(usage())
            sys.exit()

        elif opt == "--data_provider":
            try:
                data_provider_type = eval(arg)
            except Exception as e:
                err_msg = str(e)
            # end try

            if not isinstance(data_provider_type, DataProviderType):
                err = True
            # end if

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
            try:
                birth_gmm = eval(arg)
            except Exception as e:
                err_msg = str(e)
            # end try

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
            try:
                f = eval(arg)
            except Exception as e:
                err_msg = str(e)
            # end try

            if not isinstance(f, np.ndarray):
                err = True
            # end if

        elif opt == "--mat_q":
            try:
                q = eval(arg)
            except Exception as e:
                err_msg = str(e)
            # end try

            if not isinstance(q, np.ndarray):
                err = True
            # end if

        elif opt == "--mat_h":
            try:
                h = eval(arg)
            except Exception as e:
                err_msg = str(e)
            # end try

            if not isinstance(h, np.ndarray):
                err = True
            # end if

        elif opt == "--mat_r":
            try:
                r = eval(arg)
            except Exception as e:
                err_msg = str(e)
            # end try

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
            try:
                density_draw_style = eval(arg)
            except Exception as e:
                err_msg = str(e)
            # end try

            if not isinstance(density_draw_style, DensityDrawStyle):
                err = True
            # end if

        elif opt == "--n_samples_density_map":
            n_samples_density_map = int(arg)

        elif opt == "--n_bins_density_map":
            n_bins_density_map = int(arg)

        elif opt == "--draw_layers":
            try:
                draw_layers = eval(arg)
            except Exception as e:
                err_msg = str(e)
            # end try

            if isinstance(draw_layers, list):
                for layer in draw_layers:
                    if not isinstance(layer, DrawLayer):
                        err = True
                        break
                    # end if
                # end for
            # end if

        elif opt == "--input":
            inputfile = arg

        elif opt == "--coord_system_conversion":
            try:
                coord_system_conversion = eval(arg)
            except Exception as e:
                err_msg = str(e)
            # end try

            if not isinstance(coord_system_conversion, CoordSysConv):
                err = True
            # end if
        # end if

        if err or err_msg:
            print(f"Reading parameter \'{opt}\' caused an error. Argument not provided in correct format.")

            if err_msg is not None:
                print(f"Evaluation error: {err_msg}.")
            # end if
            sys.exit(2)
        # end if
    # end for

    # Evaluate dynamic matrices
    model = TransitionModel.PCW_CONST_WHITE_ACC_MODEL_2xND  # XXX

    if model == TransitionModel.PCW_CONST_WHITE_ACC_MODEL_2xND:
        dt = 1.0
        m = PcwConstWhiteAccelModelNd(dim=2, sigma=(0.4, 0.3))  # XXX

        f = m.eval_f(dt)
        q = m.eval_q(dt)
    # end if

    # Get data from a data provider
    if data_provider_type == DataProviderType.FILE_READER:
        # Read all measurements from file
        file_reader: FileReader = FileReader(inputfile)
        line_handler: InputLineHandlerLatLonIdx = InputLineHandlerLatLonIdx()
        file_reader.read(line_handler)
        data_provider = file_reader

    else:  # data_provider_type == DataProviderType.SIMULATOR
        # XXX replace fixed paraemter values
        data_provider = PhdFilterDataProvider(f=f, q=q, dt=dt, t_max=50, n_birth=1, var_birth=1, n_fa=10, var_fa=10, limits=limits, p_survival=p_survival, p_detection=p_detection, sigma_vel_x=sigma_vel_x, sigma_vel_y=sigma_vel_y)
    # end if

    # Convert data from certain coordinate systems to ENU, which is used internally
    if coord_system_conversion == CoordSysConv.WGS84:
        data_provider = Wgs84ToEnuConverter(data_provider.frame_list, observer)
    # end if
    sim: GmPhdFilterSimulator = GmPhdFilterSimulator(data_provider=data_provider, fn_out=outputfile, limits=limits, observer=observer, logging=verbosity,
                                                     birth_gmm=birth_gmm, p_survival=p_survival, p_detection=p_detection,
                                                     f=f, q=q, h=h, r=r, clutter=clutter,
                                                     trunc_thresh=trunc_thresh, merge_thresh=merge_thresh, max_components=max_components,
                                                     ext_states_bias=ext_states_bias, ext_states_use_integral=ext_states_use_integral,
                                                     density_draw_style=density_draw_style, n_samples_density_map=n_samples_density_map, n_bins_density_map=n_bins_density_map,
                                                     draw_layers=draw_layers)

    sim.run()


if __name__ == "__main__":
    main(sys.argv)
