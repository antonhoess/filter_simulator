#!/usr/bin/env python3

from __future__ import annotations
import numpy as np
from enum import auto, Enum
from abc import ABC
from typing import Sequence, List, Optional, Union
from matplotlib.patches import Ellipse, Rectangle
import seaborn as sns

from filter_simulator.common import Logging, Limits, Position
from filter_simulator.dyn_matrix import TransitionModel
from gm_phd_filter import GmComponent, Gmm, DistMeasure
from scenario_data.scenario_data_simulator import BirthDistribution
from base_filter_simulator import BaseFilterSimulator, BaseFilterSimulatorConfig, DrawLayerBase, SimStepPartBase, DensityDrawStyleBase
from filter_simulator.window_helper import LimitsMode
from scenario_data.scenario_data import ScenarioData
from scenario_data.scenario_data_converter import CoordSysConv


class DrawLayer(DrawLayerBase):
    DENSITY_MAP = auto()
    FOV = auto()
    BIRTH_AREA = auto()
    ALL_TRAJ_LINE = auto()
    ALL_TRAJ_POS = auto()
    UNTIL_TRAJ_LINE = auto()
    UNTIL_TRAJ_POS = auto()
    ALL_DET = auto()
    ALL_DET_CONN = auto()
    UNTIL_MISSED_DET = auto()
    UNTIL_FALSE_ALARM = auto()
    CUR_GMM_COV_ELL = auto()
    CUR_GMM_COV_MEAN = auto()
    CUR_EST_STATE = auto()
    UNTIL_EST_STATE = auto()
    CUR_DET = auto()
# end class


class DensityDrawStyle(DensityDrawStyleBase):
    NONE = auto()
    KDE = auto()
    EVAL = auto()
    HEATMAP = auto()
# end class


class DataProviderType(Enum):
    FILE_READER = auto()
    SIMULATOR = auto()
# end class


class GmPhdBaseFilterSimulator(BaseFilterSimulator):
    def __init__(self, scenario_data: ScenarioData, output_coord_system_conversion: CoordSysConv,
                 fn_out: str, fn_out_video: Optional[str], auto_step_interval: int, auto_step_autostart: bool, fov: Limits, birth_area: Limits, limits_mode: LimitsMode, observer: Position, logging: Logging,
                 trunc_thresh: float, merge_dist_measure: DistMeasure, merge_thresh: float, max_components: int,
                 ext_states_bias: float, ext_states_use_integral: bool,
                 gospa_c: float, gospa_p: int,
                 gui: bool, density_draw_style: DensityDrawStyle, n_samples_density_map: int, n_bins_density_map: int,
                 draw_layers: Optional[List[DrawLayer]], sim_loop_step_parts: List[SimStepPartBase], show_legend: Optional[Union[int, str]], show_colorbar: bool, start_window_max: bool,
                 init_kbd_cmds: List[str]):

        BaseFilterSimulator.__init__(self, scenario_data, output_coord_system_conversion,
                                     fn_out, fn_out_video, auto_step_interval, auto_step_autostart, fov, limits_mode, observer, logging,
                                     gospa_c, gospa_p,
                                     gui, density_draw_style, n_bins_density_map,
                                     draw_layers, sim_loop_step_parts, show_legend, show_colorbar, start_window_max,
                                     init_kbd_cmds)

        self._trunc_thresh: float = trunc_thresh
        self._merge_dist_measure: DistMeasure = merge_dist_measure
        self._merge_thresh: float = merge_thresh
        self._max_components: int = max_components
        self._ext_states_bias: float = ext_states_bias
        self._ext_states_use_integral: bool = ext_states_use_integral
        self._n_samples_density_map: int = n_samples_density_map
        self._birth_area: Limits = birth_area
    # end def

    def _sim_loop_extract_states(self):
        self._last_step_part = "Extract States"

        if self._step >= 0:
            # Extract states
            ext_states = self.f.extract_states(bias=self._ext_states_bias, use_integral=self._ext_states_use_integral)
            self._remove_duplikate_states(ext_states)
            self._ext_states.append(ext_states)
        # end if
    # end def

    def _calc_density(self, x: np.ndarray, y: np.ndarray) -> float:
        # Code taken from eval_grid_2d()
        points = np.stack((x, y), axis=-1).reshape(-1, 2)

        vals = self.f.gmm.eval_list(points, which_dims=(0, 1))

        return np.array(vals).reshape(x.shape)
    # end def

    @staticmethod
    def _get_cov_ellipse_from_comp(comp: GmComponent, n_std: float, which_dims: Sequence[int] = (0, 1), **kwargs):
        which_dims = list(which_dims)
        comp = comp.get_with_reduced_dims(which_dims)
        return GmPhdBaseFilterSimulator._get_cov_ellipse(comp.cov, comp.loc, n_std, **kwargs)
    # end def

    @staticmethod
    def _get_cov_ellipse(cov, centre, n_std, **kwargs):
        """ Return a matplotlib Ellipse patch representing the covariance matrix
        cov centred at centre and scaled by the factor n_std. """
        # Find and sort eigenvalues and eigenvectors into descending order
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        order = eig_vals.argsort()[::-1]
        eig_vals, eig_vecs = eig_vals[order], eig_vecs[:, order]

        # The counter-clockwise angle to rotate our ellipse by
        vx, vy = eig_vecs[:, 0][0], eig_vecs[:, 0][1]
        theta = np.arctan2(vy, vx)

        # Width and height of ellipse to draw
        width, height = 2 * n_std * np.sqrt(eig_vals)
        return Ellipse(xy=centre, width=width, height=height, angle=float(np.degrees(theta)), **kwargs)
    # end def

    def _draw_density_map(self, zorder):
        # Draw density map
        if self._density_draw_style == DensityDrawStyle.KDE:
            if len(self.f.gmm) > 0:
                samples = self.f.gmm.samples(self._n_samples_density_map)
                x = [s[0] for s in samples]
                y = [s[1] for s in samples]
                self._draw_plot = sns.kdeplot(x, y, shade=True, ax=self._ax, shade_lowest=False, cmap=self._draw_cmap, cbar=(self._show_colorbar and not self._colorbar_is_added), cbar_ax=self._cax, zorder=zorder)  # Colorbar instead of label
                self._colorbar_is_added = True
            # end if

        elif self._density_draw_style == DensityDrawStyle.EVAL:
            x, y, z = self._calc_density_map(self._draw_limits, grid_res=self._n_bins_density_map)
            self._draw_plot = self._ax.contourf(x, y, z, 100, cmap=self._draw_cmap, zorder=zorder)  # Colorbar instead of label

        elif self._density_draw_style == DensityDrawStyle.HEATMAP:
            samples = self.f.gmm.samples(self._n_samples_density_map)
            det_limits = self._det_limits
            plot = self._ax.hist2d([s[0] for s in samples], [s[1] for s in samples], bins=self._n_bins_density_map,
                                   range=[[det_limits.x_min, det_limits.x_max], [det_limits.y_min, det_limits.y_max]], density=False, cmap=self._draw_cmap, zorder=zorder)  # Colorbar instead of label
            # It may make no sense to change the limits, since the sampling happens anyway until infinity in each direction of the
            # state space and therefore the bins will be quite empty when zooming in, which results in a poor visualization
            # plot = self._ax.hist2d([s[0] for s in samples], [s[1] for s in samples], bins=self.__n_bins_density_map,
            #                        range=[[limits.x_min, limits.x_max], [limits.y_min, limits.y_max]], density=False, cmap=cmap, zorder=zorder)  # Colorbar instead of label
            self._draw_plot = plot[3]  # Get the image itself

        else:  # DensityDrawStyle.NONE:
            pass
        # end if
    # end def

    def _draw_birth_area(self, zorder):
        # The rectangle defining the birth area of the simulator
        width = self._birth_area.x_max - self._birth_area.x_min
        height = self._birth_area.y_max - self._birth_area.y_min
        ell = Rectangle(xy=(self._birth_area.x_min, self._birth_area.y_min), width=width, height=height, fill=False, edgecolor="black",
                        linestyle=":", linewidth=0.5, zorder=zorder, label="birth area")
        self._ax.add_patch(ell)
    # end def

    def _draw_cur_gmm_cov_ell(self, zorder):
        if self.f.cur_frame is not None:
            # GM-PHD components covariance ellipses
            for comp in self.f.gmm:
                ell = self._get_cov_ellipse_from_comp(comp, 1., facecolor='none', edgecolor="black", linewidth=.5, zorder=zorder, label="gmm cov. ell.")
                self._ax.add_patch(ell)
            # end for
        # end if
    # end def

    def _draw_cur_gmm_cov_mean(self, zorder):
        if self.f.cur_frame is not None:
            # GM-PHD components means
            self._ax.scatter([comp.loc[0] for comp in self.f.gmm], [comp.loc[1] for comp in self.f.gmm], s=5, edgecolor="blue", marker="o", zorder=zorder, label="gmm comp. mean")
        # end if
    # end def

    def get_draw_routine_by_layer(self, layer: DrawLayerBase):
        if layer == DrawLayer.DENSITY_MAP:
            return self._draw_density_map

        elif layer == DrawLayer.FOV:
            return self._draw_fov

        elif layer == DrawLayer.BIRTH_AREA:
            return self._draw_birth_area

        elif layer == DrawLayer.ALL_TRAJ_LINE:
            return self._draw_all_traj_line

        elif layer == DrawLayer.ALL_TRAJ_POS:
            return self._draw_all_traj_pos

        elif layer == DrawLayer.UNTIL_TRAJ_LINE:
            return self._draw_until_traj_line

        elif layer == DrawLayer.UNTIL_TRAJ_POS:
            return self._draw_until_traj_pos

        elif layer == DrawLayer.ALL_DET:
            return self._draw_all_det

        elif layer == DrawLayer.ALL_DET_CONN:
            return self._draw_all_det_conn

        elif layer == DrawLayer.UNTIL_MISSED_DET:
            return self._draw_until_missed_det

        elif layer == DrawLayer.UNTIL_FALSE_ALARM:
            return self._draw_until_false_alarm

        elif layer == DrawLayer.CUR_GMM_COV_ELL:
            return self._draw_cur_gmm_cov_ell

        elif layer == DrawLayer.CUR_GMM_COV_MEAN:
            return self._draw_cur_gmm_cov_mean

        elif layer == DrawLayer.CUR_EST_STATE:
            return self._draw_cur_est_state

        elif layer == DrawLayer.UNTIL_EST_STATE:
            return self._draw_until_est_state

        elif layer == DrawLayer.CUR_DET:
            return self._draw_cur_det
        # end if
    # end def

    @staticmethod
    def get_draw_layer_enum() -> DrawLayerBase:
        return DrawLayer
    # end def
# end class GmPhdBaseFilterSimulator


class GmPhdBaseFilterSimulatorConfig(ABC, BaseFilterSimulatorConfig):
    def __init__(self):
        BaseFilterSimulatorConfig.__init__(self)

        self._parser.epilog = GmPhdBaseFilterSimulatorConfig.__epilog()

        # General group
        group = self._parser_groups["general"]

        group.add_argument("--data_provider", action=self._EvalAction, comptype=DataProviderType, user_eval=self._user_eval, choices=[str(t) for t in DataProviderType], default=DataProviderType.FILE_READER,
                           help="Sets the data provider type that defines the data source. DataProviderType.FILE_READER reads lines from file defined in --input, "
                                "DataProviderType.SIMULATOR simulates the target behaviour defined by the parameters given in section SIMULATOR).")

        # PHD group - derived from filter group
        group = self._parser_groups["filter"]

        group.add_argument("--birth_gmm", action=self._EvalListToTypeAction, comptype=GmComponent, user_eval=self._user_eval, restype=Gmm, default=Gmm([GmComponent(0.1, [0, 0], np.eye(2) * 10. ** 2)]),
                           help="List ([]) of GmComponent which defines the birth-GMM. Format for a single GmComponent: GmComponent(weight, mean, covariance_matrix).")

        group.add_argument("--p_survival", metavar="[>0.0 - 1.0]", type=BaseFilterSimulatorConfig.InRange(float, min_ex_val=.0, max_val=1.), default=.9,
                           help="Sets the survival probability for the GM components from time step k to k+1.")

        group.add_argument("--n_birth", metavar="[>0.0 - N]", type=BaseFilterSimulatorConfig.InRange(float, min_ex_val=.0), default=1.,
                           help="Sets the mean number of newly born objects in each step to N_BIRTH.")

        group.add_argument("--var_birth", metavar="[>0.0 - N]", type=BaseFilterSimulatorConfig.InRange(float, min_ex_val=.0), default=1.,
                           help="Sets the variance of newly born objects to VAR_BIRTH.")

        group.add_argument("--p_detection", metavar="[>0.0 - 1.0]", type=BaseFilterSimulatorConfig.InRange(float, min_ex_val=.0, max_val=1.), default=.9,
                           help="Sets the (sensor's) detection probability for the measurements.")

        group.add_argument("--transition_model", action=self._EvalAction, comptype=TransitionModel, user_eval=self._user_eval, choices=[str(t) for t in TransitionModel], default=TransitionModel.INDIVIDUAL,
                           help=f"Sets the transition model. If set to {str(TransitionModel.INDIVIDUAL)}, the matrices F (see parameter --mat_f) and Q (see parameter --mat_q) need to be specified. "
                                f"{str(TransitionModel.PCW_CONST_WHITE_ACC_MODEL_2xND)} stands for Piecewise Constant Acceleration Model.")

        group.add_argument("--delta_t", metavar="[>0.0 - N]", dest="dt", type=BaseFilterSimulatorConfig.InRange(float, min_ex_val=.0), default=1.,
                           help="Sets the time betwen two measurements to DELTA_T. Does not work with the --transition_model parameter set to TransitionModel.INDIVIDUAL.")

        group.add_argument("--mat_f", dest="f", action=self._EvalAction, comptype=np.ndarray, user_eval=self._user_eval, default=np.eye(2),
                           help="Sets the transition model matrix.")

        group.add_argument("--mat_q", dest="q", action=self._EvalAction, comptype=np.ndarray, user_eval=self._user_eval, default=np.eye(2) * 0.,
                           help="Sets the process noise covariance matrix.")

        group.add_argument("--mat_h", dest="h", action=self._EvalAction, comptype=np.ndarray, user_eval=self._user_eval, default=np.eye(2),
                           help="Sets the measurement model matrix.")

        group.add_argument("--mat_r", dest="r", action=self._EvalAction, comptype=np.ndarray, user_eval=self._user_eval, default=np.eye(2) * .1,
                           help="Sets the measurement noise covariance matrix.")

        group.add_argument("--sigma_accel_x", type=float, default=.1,
                           help="Sets the variance of the acceleration's x-component to calculate the process noise covariance_matrix Q. Only evaluated when using the "
                                f"{str(TransitionModel.PCW_CONST_WHITE_ACC_MODEL_2xND)} (see parameter --transition_model) and in this case ignores the value specified for Q (see parameter --mat_q).")

        group.add_argument("--sigma_accel_y", type=float, default=.1,
                           help="Sets the variance of the acceleration's y-component to calculate the process noise covariance_matrix Q. Only evaluated when using the "
                                f"{str(TransitionModel.PCW_CONST_WHITE_ACC_MODEL_2xND)} (see parameter --transition_model) and in this case ignores the value specified for Q (see parameter --mat_q).")

        group.add_argument("--gate_thresh", metavar="[0.0 - 1.0]", type=BaseFilterSimulatorConfig.InRange(float, min_val=0, max_val=1.), default=None,
                           help="Sets the confidence threshold for chi^2 gating on new measurements to GATE_THRESH.")

        group.add_argument("--rho_fa", metavar="[>0.0 - N]", type=BaseFilterSimulatorConfig.InRange(float, min_ex_val=.0), default=None,
                           help="Sets the probability of false alarms per volume unit to RHO_FA. If specified, the mean number of false alarms (see parameter --n_fa) will be recalculated "
                                "based on RHO_FA and the FoV. ")

        group.add_argument("--trunc_thresh", metavar="[>0.0 - N]", type=BaseFilterSimulatorConfig.InRange(float, min_ex_val=.0), default=1e-6,
                           help="Sets the truncation threshold for the prunging step. GM components with weights lower than this value get directly removed.")

        group.add_argument("--merge_dist_measure", action=self._EvalAction, comptype=DistMeasure, user_eval=self._user_eval, choices=[str(t) for t in DistMeasure],
                           default=DistMeasure.MAHALANOBIS_MOD,
                           help="Defines the measurement for calculating the distance between two GMM components.")

        group.add_argument("--merge_thresh", metavar="[>0.0 - N]", type=BaseFilterSimulatorConfig.InRange(float, min_ex_val=.0), default=.01,
                           help="Sets the merge threshold for the prunging step. GM components with a distance distance lower than this value get merged. The distacne measure is given by the "
                                "parameter --merge_dist_measure and depending in this parameter the threashold needs to be set differently.")

        group.add_argument("--max_components", metavar="[1 - N]", type=BaseFilterSimulatorConfig.InRange(int, min_val=1), default=100,
                           help="Sets the max. number of GM components used for the GMM representing the current intensity.")

        group.add_argument("--ext_states_bias", metavar="[>0.0 - N]", type=BaseFilterSimulatorConfig.InRange(float, min_ex_val=.0), default=1.,
                           help="Sets the bias for extracting the current states to EXT_STATES_BIAS. It works as a factor for the GM component's weights and is used, "
                                "in case the weights are too small to reach a value higher than 0.5, which in needed to get extracted as a state.")

        group.add_argument("--ext_states_use_integral", type=BaseFilterSimulatorConfig.IsBool, nargs="?", default=False, const=True, choices=[True, False, 1, 0],
                           help="Specifies if the integral approach for extracting the current states should be used.")

        # Data Simulator group
        group = self._parser.add_argument_group("Data Simulator - calculates detections from simulation")

        group.add_argument("--birth_area", metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"), action=self._LimitsAction, type=float, nargs=4, default=None,
                           help="Sets the are for newly born targets. It not set, the same limits as defined by --fov will get used.")

        group.add_argument("--sim_t_max",  metavar="[0 - N]", type=BaseFilterSimulatorConfig.InRange(int, min_val=0), default=50,
                           help="Sets the number of simulation steps to SIM_T_MAX when using the DataProviderType.SIMULATOR (see parameter --data_provider).")

        group.add_argument("--n_fa", metavar="[0.0 - N]", type=BaseFilterSimulatorConfig.InRange(float, min_val=.0), default=1.,
                           help="Sets the mean number of false alarms in the FoV to N_FA.")

        group.add_argument("--var_fa", metavar="[>0.0 - N]", type=BaseFilterSimulatorConfig.InRange(float, min_ex_val=.0),  default=1.,
                           help="Sets the variance of false alarms in the FoV to VAR_FA.")

        group.add_argument("--birth_dist", action=self._EvalAction, comptype=BirthDistribution, user_eval=self._user_eval, choices=[str(t) for t in BirthDistribution], default=BirthDistribution.UNIFORM_AREA,
                           help="Sets the type of probability distribution for new born objects. In case BirthDistribution.UNIFORM_AREA is set, the newly born objects are distributed uniformly "
                                "over the area defined by the parameter --birth_area (or the FoV if not set) and the initial velocity will be set to the values defineed by the perameters "
                                f"--sigma_vel_x and --sigma_vel_y. If {str(BirthDistribution.GMM_FILTER)} is set, the same GMM will get used for the creating of new objects, as the filter uses for "
                                f"their detection. This parameter only takes effect when the parameter --data_provider is set to {str(DataProviderType.SIMULATOR)}.")

        group.add_argument("--sigma_vel_x", metavar="[0.0 - N]", type=BaseFilterSimulatorConfig.InRange(float, min_val=.0), default=.2,
                           help="Sets the variance of the velocitiy's initial x-component of a newly born object to SIGMA_VEL_X. Only takes effect if the parameter --birth_dist is set to "
                                f"{str(BirthDistribution.UNIFORM_AREA)}.")

        group.add_argument("--sigma_vel_y", metavar="[0.0 - N]", type=BaseFilterSimulatorConfig.InRange(float, min_val=.0), default=.2,
                           help="Sets the variance of the velocitiy's initial y-component of a newly born object to SIGMA_VEL_y. Only takes effect if the parameter --birth_dist is set to "
                                f"{str(BirthDistribution.UNIFORM_AREA)}.")

        # Simulator group
        # -> Nothing to add, but helper functions implemented

        # File Reader group
        # -> Nothing to add

        # File Storage group
        # -> Nothing to add

        # Visualization group
        group = self._parser_groups["visualization"]

        group.add_argument("--density_draw_style", action=self._EvalAction, comptype=DensityDrawStyle, user_eval=self._user_eval, choices=[str(t) for t in DensityDrawStyle],
                           default=DensityDrawStyle.NONE,
                           help=f"Sets the drawing style to visualizing the density/intensity map. Possible values are: {str(DensityDrawStyle.KDE)} (kernel density estimator), "
                                f"{str(DensityDrawStyle.EVAL)} (evaluate the correct value for each cell in a grid) and {str(DensityDrawStyle.HEATMAP)} (heatmap made of sampled points from "
                           f"the intensity distribution).")

        group.add_argument("--n_samples_density_map", metavar="[100 - N]", type=BaseFilterSimulatorConfig.InRange(int, min_val=1000),  default=1000,
                           help="Sets the number samples to draw from the intensity distribution for drawing the density map. A good number might be 10000.")

        group.add_argument("--draw_layers", metavar=f"[{{{  ','.join([str(t) for t in DrawLayer]) }}}*]", action=self._EvalListAction, comptype=DrawLayerBase, user_eval=self._user_eval,
                           default=[ly for ly in DrawLayer if ly not in[DrawLayer.ALL_TRAJ_LINE, DrawLayer.ALL_TRAJ_POS, DrawLayer.ALL_DET, DrawLayer.ALL_DET_CONN,
                                                                        DrawLayer.CUR_GMM_COV_ELL, DrawLayer.CUR_GMM_COV_MEAN]],
                           help=f"Sets the list of drawing layers. Allows to draw only the required layers and in the desired order. If not set, a fixes set of layers are drawn in a fixed order. "
                           f"Example 1: [{str(DrawLayer.DENSITY_MAP)}, {str(DrawLayer.UNTIL_EST_STATE)}]\n"
                           f"Example 2: [layer for layer in DrawLayer if not layer == {str(DrawLayer.CUR_GMM_COV_ELL)} and not layer == {str(DrawLayer.CUR_GMM_COV_MEAN)}]")
    # end def __init__

    @staticmethod
    def __epilog():
        return "GUI\n" + \
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
               "\n" + \
               "    MANUAL_EDITING mode\n" \
               "        In the MANUAL_EDITING mode there are following commands:\n" + \
               "        * CTRL + LEFT CLICK: Add point to current (time) frame.\n" + \
               "        * SHIFT + LEFT CLICK: Add frame.\n" + \
               "        * CTRL + RIGHT CLICK: Remove last set point.\n" + \
               "        * SHIFT + RIGHT CLICK: Remove last frame.\n" + \
               "\n" + \
               "    Any mode\n" \
               "        In any mode there are following commands:\n" + \
               "        * CTRL + SHIFT + RIGHT CLICK: Stores the current detections (either created manually or by simulation) to the specified output file. Attention: the plotting window will " \
               "change its size back to the size where movie writer (which is the one, that creates the video) got instantiated. This is neccessary, since the frame/figure size needs to be " \
               "constant. However, immediately before saving the video, the window can be resized to the desired size for capturing the next video, since the next movie writer will use this new " \
               "window size. This way, at the very beginning after starting the program, the window might get resized to the desired size and then a (more or less) empty video might be saved, " \
               "which starts a new one on the desired size, directly at the beginning of the simulation.\n" \
               "        * CTRL + ALT + SHIFT + RIGHT CLICK: Stores the plot window frames as video, if its filename got specified.\n" \
               "        * CTRL + WHEEL UP: Zooms in.\n" \
               "        * CTRL + WHEEL DOWN: Zooms out.\n" \
               "        * CTRL + LEFT MOUSE DOWN + MOUSE MOVE: Moves the whole scene with the mouse cursor.\n" \
               "        * SHIFT + LEFT CLICK: De-/activate automatic stepping."
    # end def

    @staticmethod
    def _user_eval(s: str):
        return eval(s)
    # end def
# end class
