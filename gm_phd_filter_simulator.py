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
from filter_simulator.data_provider_converter import CoordSysConv, Wgs84ToEnuConverter
from filter_simulator.dyn_matrix import TransitionModel, PcwConstWhiteAccelModelNd
from gm_phd_filter import GmPhdFilter, GmComponent, Gmm
from phd_filter_data_provider import PhdFilterDataProvider
from filter_simulator.window_helper import LimitsMode


class DrawLayer(Enum):
    DENSITY_MAP = 0
    ALL_DET = 1
    ALL_DET_CONN = 2
    GMM_COV_ELL = 3
    GMM_COV_MEAN = 4
    EST_STATE = 5
    ALL_EST_STATE = 6
    CUR_DET = 7
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


class GmPhdFilterSimulator(FilterSimulator, GmPhdFilter):
    def __init__(self, data_provider: IDataProvider, output_coord_system_conversion: CoordSysConv,
                 fn_out: str, fn_out_video: Optional[str], limits: Limits, limits_mode: LimitsMode, observer: Position, logging: Logging,
                 birth_gmm: List[GmComponent], p_survival: float, p_detection: float,
                 f: np.ndarray, q: np.ndarray, h: np.ndarray, r: np.ndarray, clutter: float,
                 trunc_thresh: float, merge_thresh: float, max_components: int,
                 ext_states_bias: float, ext_states_use_integral: bool,
                 density_draw_style: DensityDrawStyle, n_samples_density_map: int, n_bins_density_map: int,
                 draw_layers: Optional[List[DrawLayer]]):
        FilterSimulator.__init__(self, data_provider, output_coord_system_conversion, fn_out, fn_out_video, limits, limits_mode, observer, logging)
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

        self.__ext_states: List[List[np.ndarray]] = []

    def _set_sim_loop_step_part_conf(self):
        # Configure the processing steps
        sim_step_part_conf = SimStepPartConf()

        sim_step_part_conf.add_user_step(self.__sim_loop_predict_and_update)
        sim_step_part_conf.add_user_step(self.__sim_loop_prune)
        sim_step_part_conf.add_user_step(self.__sim_loop_extract_states)
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

    def __sim_loop_extract_states(self):
        if self._step >= 0:
            # Extract states
            ext_states = self._extract_states(bias=self.__ext_states_bias, use_integral=self.__ext_states_use_integral)
            self.__remove_duplikate_states(ext_states)
            self.__ext_states.append(ext_states)
        # end if
    # end def

    @staticmethod
    def __remove_duplikate_states(states: List[np.ndarray]):
        for i in range(len(states)):
            if i >= len(states):
                break
            else:
                loc_a = states[i]
            # end if

            for k in reversed(range(len(states))):
                if k <= i:
                    break
                else:
                    loc_b = states[k]

                    if np.array_equal(loc_a, loc_b):
                        del states[k]
                    # end if
                # end if
            # end for
        # end for
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
                    self._ax.scatter([det.x for det in frame], [det.y for det in frame], s=10, edgecolor="green", marker="o", zorder=zorder)
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
                if self._cur_frame is not None:
                    # Estimated states
                    est_items = self.__ext_states[-1]
                    self._ax.scatter([est_item[0] for est_item in est_items], [est_item[1] for est_item in est_items], s=200, c="gray", edgecolor="black", marker="o", zorder=zorder)
                # end if

            elif ly == DrawLayer.ALL_EST_STATE:
                if self._cur_frame is not None:
                    # Estimated states
                    est_items = [est_item for est_items in self.__ext_states for est_item in est_items]
                    self._ax.scatter([est_item[0] for est_item in est_items], [est_item[1] for est_item in est_items], s=50, c="red", edgecolor="black", marker="o", zorder=zorder)
                # end if

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

    def _cb_keyboard(self, cmd: str) -> None:
        if cmd == "":
            self._next_part_step = True

        elif cmd == "h":
            print(GmPhdFilterSimulatorParam.usage())

        else:
            pass
        # end if
    # end def
# end class GmPhdFilterSimulator


class GmPhdFilterSimulatorParam:
    @staticmethod
    def usage() -> str:
        return "{} <PARAMETERS>\n".format(os.path.basename(__file__)) + \
               "\n" + \
               "-h, --help: Prints this help.\n" + \
               "\n" + \
               "    --data_provider=DATA_PROVIDER:\n" + \
               "       Sets the data provider type that defines the data source. Possible values are: DataProviderType.FILE_READER (reads lines from file defined in --input_file), " \
               "DataProviderType.SIMULATOR (simulates the PHD behaviour defined by the paraemters given in section SIMULATOR).\n" + \
               "    Example: DensityDrawStyle.DRAW_HEATMAP" + \
               "\n" + \
               "-l, --limits=LIMITS:\n" + \
               "    Sets the limits for the canvas to LIMITS. Its format is 'Limits(X_MIN, Y_MIN, X_MAX Y_MAX)'.\n" + \
               "    Example: Limits(-10, -10, 10, 10)\n" + \
               "\n" + \
               "    --limits_mode=LIMITS_MODE:\n" + \
               "    Sets the limits mode, which defines how the limits for the plotting window are set initially and while updating the plot. " \
               "LimitsMode.ALL_DETECTIONS_INIT_ONLY, LimitsMode.ALL_DETECTIONS_FIXED_UPDATE, LimitsMode.ALL_CANVAS_ELEMENTS_DYN_UPDATE, " \
               "LimitsMode.MANUAL_AREA_INIT_ONLY [Default], LimitsMode.MANUAL_AREA_FIXED_UPDATE\n" + \
               "\n" + \
               "-v, --verbosity_level=VERBOSITY:\n" \
               "    Sets the programs verbosity level to VERBOSITY. 0 = Silent [Default], >0 = decreasing verbosity: 1 = CRITICAL, 2 = ERROR, 3 = WARNING, 4 = INFO, 5 = DEBUG.\n" \
               "\n" + \
               "-p, --observer_position=OBSERVER_POSITION:\n" \
               "    Sets the geodetic position of the observer in WGS84 to OBSERVER_POSITION. " \
               "Can be used instead of the automatically used center of all detections or in case of only manually creating detections, " \
               "which needed to be transformed back to WGS84. Its format is 'LAT;LON'.\n" + \
               "\n" + \
               "    --birth_gmm=BIRTH_GMM:\n" + \
               "    List ([]) of GmComponent which defines the birth-GMM. Format for a single GmComponent: GmComponent(weight, mean, covariance_matrix).\n" + \
               "    Example: [GmComponent(0.1, [0, 0], np.array([[5, 2], [2, 5]]))]\n" + \
               "\n" + \
               "    --p_survival=P_SURVIVAL:\n" + \
               "    Sets the survival probability for the PHD from time step k to k+1.\n" + \
               "\n" + \
               "    --n_birth=N_BIRTH:\n" + \
               "    Sets the average number of newly born objects in each step to N_BIRTH.\n" + \
               "\n" + \
               "    --var_birth=VAR_BIRTH:\n" + \
               "    Sets the variance of newly born objects to VAR_BIRTH.\n" + \
               "\n" + \
               "    --p_detection=P_DETECTION:\n" + \
               "    Sets the (sensor's) detection probability for the measurements.\n" + \
               "\n" + \
               "-v, --transition_model=TRANSITION_MODEL:\n" \
               "    Sets the transition model. If set to TransitionModel.INDIVIDUAL, the matrices f and q need to be specified. " \
               "TransitionModel.INDIVIDUAL [Default], TransitionModel.PCW_CONST_WHITE_ACC_MODEL_2xND (Piecewise Constant Acceleration Model)\n" \
               "\n" + \
               "    --delta_t=DELTA_T:\n" + \
               "    Sets the time betwen two measurements to DELTA_T. Works only with the --transition_model option set to something difeerent than TransitionModel.INDIVIDUAL. Default: 1.0\n" + \
               "\n" + \
               "    --mat_f=F:\n" + \
               "    Sets the transition model matrix for the PHD.\n" + \
               "    Example: np.eye(2)\n" + \
               "\n" + \
               "    --mat_q=Q:\n" + \
               "    Sets the process noise covariance matrix.\n" + \
               "    Example: np.eye(2) * 0.\n" + \
               "\n" + \
               "    --mat_h=H:\n" + \
               "    Sets the measurement model matrix.\n" + \
               "    Example: np.eye(2)\n" + \
               "\n" + \
               "    --mat_r=R:\n" + \
               "    Sets the measurement noise covariance matrix.\n" + \
               "    Example: np.eye(2) * .1\n" + \
               "\n" + \
               "    --sigma_accel_x=SIGMA_ACCEL_X:\n" + \
               "    Sets the variance of the acceleration's x-component to calculate the process noise covariance_matrix Q. Only evaluated when using the " \
               "TransitionModel.PCW_CONST_WHITE_ACC_MODEL_2xND (see parameter --transition_model) and in this case ignores the value given for Q (see parameter --mat_q).\n" + \
               "\n" + \
               "    --sigma_accel_x=SIGMA_ACCEL_Y:\n" + \
               "    Sets the variance of the acceleration's y-component to calculate the process noise covariance_matrix Q. Only evaluated when using the " \
               "TransitionModel.PCW_CONST_WHITE_ACC_MODEL_2xND (see parameter --transition_model) and in this case ignores the value given for Q (see parameter --mat_q).\n" + \
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
               "    Defines if the integral approach for extracting the current states should be used. 0 = False, 1 = True.\n" + \
               "\n" + \
               "    --density_draw_style=DENSITY_DRAW_STYLE:\n" + \
               "    Sets the drawing style to visualizing the density/intensity map. Possible values are: DensityDrawStyle.KDE (kernel density estimator), " \
               "DensityDrawStyle.EVAL (evaluate the correct value for each cell in a grid) and DensityDrawStyle.HEATMAP (heatmap made of sampled points from the PHD).\n" + \
               "    Example: DensityDrawStyle.DRAW_HEATMAP\n" + \
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
               "    --input_coord_system_conversion=INPUT_COORD_SYSTEM_CONVERSION:\n" + \
               "    Defines the coordinates-conversion of the provided values from the INPUT_COORD_SYSTEM_CONVERSION into the internal system (ENU). " \
               "Possible values are class CoordSysConv.NONE [Default], CoordSysConv.WGS84\n" + \
               "\n" + \
               "\n" + \
               "SIMULATOR\n" + \
               "    Calculates detections from simulation.\n" \
               "\n" + \
               "    --sim_t_max=SIM_T_MAX:\n" + \
               "    Sets the number of simulation steps to SIM_T_MAX when using the DataProviderType.SIMULATOR (see parameter --data_provider).\n" + \
               "\n" + \
               "    --sigma_vel_x=SIGMA_VEL_X:\n" + \
               "    Sets the variance of the velocitiy's initial x-component of a newly born object to SIGMA_VEL_X.\n" + \
               "\n" + \
               "    --sigma_vel_y=SIGMA_VEL_Y:\n" + \
               "    Sets the variance of the velocitiy's initial y-component of a newly born object to SIGMA_VEL_Y.\n" + \
               "\n" + \
               "\n" + \
               "FILE STORAGE\n" \
               "    Stores detection to file.\n" \
               "\n" + \
               "-o, --output=OUTPUT_FILE:\n" \
               "    Sets the output file to store the (manually set or simulated) detections' coordinates to OUTPUT_FILE. Default: out.lst.\n" + \
               "\n" + \
               "    --output_seq_max=OUTPUT_SEQ_MAX:\n" + \
               "    Sets the max. number to append at the end of the output file name (see parameter --output). This allows for automatically continuously named files and prevents overwriting " \
               "previously stored results. The format will be x_0000, depending on the filename and the number of digits of OUTPUT_SEQ_MAX. Default: 9999\n" + \
               "\n" + \
               "    --output_fill_gaps=OUTPUT_FILL_GAPS:\n" + \
               "    Indicates if the first empty file name will be used when determining a output file name (see parameters --output and --output_seq_max) or if the next number " \
               "(to create the file name) will be N+1 with N is the highest number in the range and format given by --output_seq_max. 0 = False [Default], 1 = True\n" \
               "\n" + \
               "    --output_coord_system_conversion=OUTPUT_COORD_SYSTEM_CONVERSION:\n" + \
               "    Defines the coordinates-conversion of the internal system (ENU) to the OUTPUT_COORD_SYSTEM_CONVERSION for storing the values. " \
               "Possible values are class CoordSysConv.NONE [Default], CoordSysConv.WGS84\n" + \
               "\n" + \
               "    --output_video=OUTPUT_VIDEO:\n" \
               "    Sets the output file name to store the video captures from the single frames of the plotting window. Default: Not storing any video.\n" + \
               "\n" + \
               "\n" + \
               "GUI\n" + \
               "    Mouse and keyboard events on the plotting window (GUI).\n" \
               "\n" + \
               "    There are two operating modes:\n" \
               "    * SIMULATION [Default]\n" \
               "    * MANUAL_EDITING\n" \
               "\n" + \
               "    To switch between these two modes, one needs to click (at least) three times with the LEFT mouse button while holding the CTRL + SHIFT buttons pressed without interruption. " \
               "Release the keyboard buttons to complete the mode switch.\n" \
               "\n" + \
               "    SIMULATION mode\n" \
               "        In the SIMULATION mode there are following commands:\n" + \
               "        * CTRL + RIGHT CLICK: Navigate forwards (load measurement data of the next time step).\n" + \
               "        * CTRL + SHIFT + RIGHT CLICK: Stores the current detections (either created manually or by simulation) to the specified output file." \
               "        * CTRL + ALT + SHIFT + RIGHT CLICK: Stores the plot window frames as video, if its filename got specified." \
               "\n" + \
               "    MANUAL_EDITING mode\n" \
               "        In the MANUAL_EDITING mode there are following commands:\n" + \
               "        * CTRL + LEFT CLICK: Add point to current (time) frame.\n" + \
               "        * SHIFT + LEFT CLICK: Add frame.\n" + \
               "        * CTRL + RIGHT CLICK: Remove last set point.\n" + \
               "        * SHIFT + RIGHT CLICK: Remove last frame.\n" + \
               "        * CTRL + SHIFT + RIGHT CLICK: Stores the current detections (either created manually or by simulation) to the specified output file." \
               "        * CTRL + ALT + SHIFT + RIGHT CLICK: Stores the plot window frames as video, if its filename got specified." \
               "\n" + \
               ""
    # end def

    def run(self, argv: List[str]):
        # Library settings
        sns.set(color_codes=True)

        # Initialize random generator
        random.seed(datetime.now())

        # Read command line arguments
        data_provider_type: DataProviderType = DataProviderType.FILE_READER

        sim_t_max = 50
        limits: Limits = Limits(-10, -10, 10, 10)
        limits_mode: LimitsMode = LimitsMode.MANUAL_AREA_INIT_ONLY
        verbosity: Logging = Logging.INFO
        observer: Optional[Position] = None

        birth_gmm: Gmm = Gmm([GmComponent(0.1, [0, 0], np.eye(2) * 10. ** 2)])
        n_birth: int = 1
        var_birth: int = 1
        p_survival: float = 0.9
        p_detection: float = 0.9
        transition_model = TransitionModel.INDIVIDUAL
        dt = 1.
        f: np.ndarray = np.eye(2)
        q: np.ndarray = np.eye(2) * 0.
        h: np.ndarray = np.eye(2)
        r: np.ndarray = np.eye(2) * .1
        sigma_vel_x = .2
        sigma_vel_y = .2
        sigma_accel_x = .1
        sigma_accel_y = .1
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

        input_file: str = ""
        input_coord_system_conversion: CoordSysConv = CoordSysConv.NONE

        output: str = "out.lst"
        output_seq_max = 9999
        output_fill_gaps = False
        output_coord_system_conversion: CoordSysConv = CoordSysConv.NONE
        output_video: Optional[str] = None

        try:
            opts, args = getopt.getopt(argv[1:], "hi:l:o:p:v:", ["help", "data_provider=", "sim_t_max=", "limits=", "limits_mode=", "observer_position=", "verbosity_level=",
                                                                 "birth_gmm=", "n_birth=", "var_birth=", "p_survival=", "p_detection=", "transition_model=", "delta_t=",
                                                                 "mat_f=", "mat_q=", "mat_h=", "mat_r=", "sigma_vel_x=", "sigma_vel_y=", "sigma_accel_x=", "sigma_accel_y=", "clutter=",
                                                                 "trunc_thresh=", "merge_thresh=", "max_components=",
                                                                 "ext_states_bias=", "ext_states_use_integral=", "density_draw_style=", "n_samples_density_map=", "n_bins_density_map=", "draw_layers=",
                                                                 "input=", "input_coord_system_conversion=", "output=", "output_seq_max=", "output_fill_gaps=", "output_coord_system_conversion=",
                                                                 "output_video="])

        except getopt.GetoptError as e:
            print("Reading parameters caused error {}".format(e))
            print(self.usage())
            sys.exit(1)
        # end try

        for opt, arg in opts:
            err: bool = False
            err_msg: Optional[str] = None

            if opt in ("-h", "--help"):
                print(self.usage())
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

            elif opt == "--sim_t_max":
                sim_t_max = int(arg)

            elif opt in ("-l", "--limits"):
                try:
                    limits = eval(arg)
                except Exception as e:
                    err_msg = str(e)
                # end try

                if not isinstance(limits, Limits):
                    err = True
                # end if

            elif opt == "--limits_mode":
                try:
                    limits_mode = eval(arg)
                except Exception as e:
                    err_msg = str(e)
                # end try

                if not isinstance(limits_mode, LimitsMode):
                    err = True
                # end if

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

            elif opt == "--n_birth":
                n_birth = int(arg)

            elif opt == "--var_birth":
                var_birth = int(arg)

            elif opt == "--p_survival":
                p_survival = float(arg)

            elif opt == "--p_detection":
                p_detection = float(arg)

            elif opt == "--transition_model":
                try:
                    transition_model = eval(arg)
                except Exception as e:
                    err_msg = str(e)
                # end try

                if not isinstance(transition_model, TransitionModel):
                    err = True
                # end if

            elif opt == "--delta_t":
                dt = float(arg)

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

            elif opt == "--sigma_vel_x":
                sigma_vel_x = float(arg)

            elif opt == "--sigma_vel_y":
                sigma_vel_y = float(arg)

            elif opt == "--sigma_accel_x":
                sigma_accel_x = float(arg)

            elif opt == "--sigma_accel_y":
                sigma_accel_y = float(arg)

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
                ext_states_use_integral = bool(int(arg))

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
                input_file = arg

            elif opt == "--input_coord_system_conversion":
                try:
                    input_coord_system_conversion = eval(arg)
                except Exception as e:
                    err_msg = str(e)
                # end try

                if not isinstance(input_coord_system_conversion, CoordSysConv):
                    err = True
                # end if
            # end if

            elif opt in ("-o", "--output"):
                output = arg

            elif opt == "--output_seq_max":
                output_seq_max = int(arg)

            elif opt == "--output_fill_gaps":
                output_fill_gaps = bool(arg)

            elif opt == "--output_coord_system_conversion":
                try:
                    output_coord_system_conversion = eval(arg)
                except Exception as e:
                    err_msg = str(e)
                # end try

                if not isinstance(output_coord_system_conversion, CoordSysConv):
                    err = True
                # end if
            # end if

            elif opt == "--output_video":
                output_video = arg

            if err or err_msg:
                print(f"Reading parameter \'{opt}\' caused an error. Argument not provided in correct format.")

                if err_msg is not None:
                    print(f"Evaluation error: {err_msg}.")
                # end if
                sys.exit(2)
            # end if
        # end for

        # Evaluate dynamic matrices
        if transition_model == TransitionModel.PCW_CONST_WHITE_ACC_MODEL_2xND:
            m = PcwConstWhiteAccelModelNd(dim=2, sigma=(sigma_accel_x, sigma_accel_y))

            f = m.eval_f(dt)
            q = m.eval_q(dt)
        # end if

        # Get data from a data provider
        if data_provider_type == DataProviderType.FILE_READER:
            # Read all measurements from file
            file_reader: FileReader = FileReader(input_file)
            line_handler: InputLineHandlerLatLonIdx = InputLineHandlerLatLonIdx()
            file_reader.read(line_handler)
            data_provider = line_handler

        else:  # data_provider_type == DataProviderType.SIMULATOR
            data_provider = PhdFilterDataProvider(f=f, q=q, dt=dt, t_max=sim_t_max, n_birth=n_birth, var_birth=var_birth, n_fa=int(clutter), var_fa=int(clutter), limits=limits,
                                                  p_survival=p_survival, p_detection=p_detection, sigma_vel_x=sigma_vel_x, sigma_vel_y=sigma_vel_y)
        # end if

        # Convert data from certain coordinate systems to ENU, which is used internally
        if input_coord_system_conversion == CoordSysConv.WGS84:
            data_provider = Wgs84ToEnuConverter(data_provider.frame_list, observer)
        # end if
        sim: GmPhdFilterSimulator = GmPhdFilterSimulator(data_provider=data_provider, output_coord_system_conversion=output_coord_system_conversion, fn_out=output, fn_out_video=output_video,
                                                         limits=limits, limits_mode=limits_mode, observer=observer, logging=verbosity,
                                                         birth_gmm=birth_gmm, p_survival=p_survival, p_detection=p_detection,
                                                         f=f, q=q, h=h, r=r, clutter=clutter,
                                                         trunc_thresh=trunc_thresh, merge_thresh=merge_thresh, max_components=max_components,
                                                         ext_states_bias=ext_states_bias, ext_states_use_integral=ext_states_use_integral,
                                                         density_draw_style=density_draw_style, n_samples_density_map=n_samples_density_map, n_bins_density_map=n_bins_density_map,
                                                         draw_layers=draw_layers)

        sim.fn_out_seq_max = output_seq_max
        sim.fn_out_fill_gaps = output_fill_gaps

        sim.run()
    # end def
# end class


def main(argv: List[str]):
    sim_param = GmPhdFilterSimulatorParam()

    sim_param.run(argv)
# end def main


if __name__ == "__main__":
    main(sys.argv)
