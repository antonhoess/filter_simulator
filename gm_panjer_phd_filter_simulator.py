#!/usr/bin/env python3

from __future__ import annotations
from typing import Sequence, List, Optional, Union
import sys
import random
import numpy as np
from datetime import datetime
from matplotlib.patches import Ellipse, Rectangle
import seaborn as sns
from enum import auto, Enum

from filter_simulator.common import Logging, Limits, Position
from filter_simulator.filter_simulator import SimStepPartConf
from scenario_data.scenario_data_converter import CoordSysConv, Wgs84ToEnuConverter
from filter_simulator.dyn_matrix import TransitionModel, PcwConstWhiteAccelModelNd
from gm_panjer_phd_filter import GmPanjerPhdFilter
from gm import GmComponent, Gmm, DistMeasure
from scenario_data.scenario_data_simulator import ScenarioDataSimulator, BirthDistribution
from filter_simulator.window_helper import LimitsMode
from scenario_data.scenario_data import ScenarioData
from base_filter_simulator import BaseFilterSimulatorConfig, BaseFilterSimulator, DrawLayerBase, SimStepPartBase, DensityDrawStyleBase
from gm_phd_filter_simulator import GmPhdBaseFilterSimulatorConfig


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


class SimStepPart(SimStepPartBase):
    DRAW = 0  # Draw the current scene
    WAIT_FOR_TRIGGER = 1  # Wait for user input to continue with the next step
    LOAD_NEXT_FRAME = 2  # Load the next data (frame)
    USER_PREDICT = 3
    USER_UPDATE = 4
    USER_PRUNE = 5
    USER_MERGE = 6
    USER_EXTRACT_STATES = 7
    USER_CALC_GOSPA = 8
    USER_INITIAL_KEYBOARD_COMMANDS = 9
# end class


class DensityDrawStyle(DensityDrawStyleBase):
    NONE = 0
    KDE = 1
    EVAL = 2
    HEATMAP = 3
# end class


class DataProviderType(Enum):
    FILE_READER = 0
    SIMULATOR = 1
# end class


class GmPanjerPhdFilterSimulator(BaseFilterSimulator):
    def __init__(self, scenario_data: ScenarioData, output_coord_system_conversion: CoordSysConv,
                 fn_out: str, fn_out_video: Optional[str], auto_step_interval: int, auto_step_autostart: bool, fov: Limits, birth_area: Limits, limits_mode: LimitsMode, observer: Position, logging: Logging,
                 birth_gmm: List[GmComponent], var_birth: float, p_survival: float, p_detection: float,
                 f: np.ndarray, q: np.ndarray, h: np.ndarray, r: np.ndarray, rho_fa: float, gate_thresh: Optional[float],
                 trunc_thresh: float, merge_dist_measure: DistMeasure, merge_thresh: float, max_components: int,
                 ext_states_bias: float, ext_states_use_integral: bool,
                 gospa_c: float, gospa_p: int,
                 gui: bool, density_draw_style: DensityDrawStyle, n_samples_density_map: int, n_bins_density_map: int,
                 draw_layers: Optional[List[DrawLayer]], sim_loop_step_parts: List[SimStepPart], show_legend: Optional[Union[int, str]], show_colorbar: bool, start_window_max: bool,
                 init_kbd_cmds: List[str]):

        self.__sim_loop_step_parts: List[SimStepPart] = sim_loop_step_parts  # Needs to be set before calling the contructor of the FilterSimulator, since it already needs this values there

        BaseFilterSimulator.__init__(self, scenario_data, output_coord_system_conversion,
                                     fn_out, fn_out_video, auto_step_interval, auto_step_autostart, fov, limits_mode, observer, logging,
                                     gospa_c, gospa_p,
                                     gui, density_draw_style, n_bins_density_map,
                                     draw_layers, sim_loop_step_parts, show_legend, show_colorbar, start_window_max,
                                     init_kbd_cmds)

        self.f = GmPanjerPhdFilter(birth_gmm=birth_gmm, var_birth=var_birth, survival=p_survival, detection=p_detection, f=f, q=q, h=h, r=r, rho_fa=rho_fa, gate_thresh=gate_thresh, logging=logging)

        self._trunc_thresh: float = trunc_thresh
        self._merge_dist_measure: DistMeasure = merge_dist_measure
        self._merge_thresh: float = merge_thresh
        self._max_components: int = max_components
        self._ext_states_bias: float = ext_states_bias
        self._ext_states_use_integral: bool = ext_states_use_integral
        self._n_samples_density_map: int = n_samples_density_map
        self._birth_area: Limits = birth_area
    # end def

    def _set_sim_loop_step_part_conf(self):
        # Configure the processing steps
        sim_step_part_conf = SimStepPartConf()

        for step_part in self.__sim_loop_step_parts:
            if step_part is SimStepPart.DRAW:
                sim_step_part_conf.add_draw_step()

            elif step_part is SimStepPart.WAIT_FOR_TRIGGER:
                sim_step_part_conf.add_wait_for_trigger_step()

            elif step_part is SimStepPart.LOAD_NEXT_FRAME:
                sim_step_part_conf.add_load_next_frame_step()

            elif step_part is SimStepPart.USER_PREDICT:
                sim_step_part_conf.add_user_step(self._sim_loop_predict)

            elif step_part is SimStepPart.USER_UPDATE:
                sim_step_part_conf.add_user_step(self._sim_loop_update)

            elif step_part is SimStepPart.USER_PRUNE:
                sim_step_part_conf.add_user_step(self._sim_loop_prune)

            elif step_part is SimStepPart.USER_MERGE:
                sim_step_part_conf.add_user_step(self._sim_loop_merge)

            elif step_part is SimStepPart.USER_EXTRACT_STATES:
                sim_step_part_conf.add_user_step(self._sim_loop_extract_states)

            elif step_part is SimStepPart.USER_CALC_GOSPA:
                sim_step_part_conf.add_user_step(self._sim_loop_calc_gospa)

            elif step_part is SimStepPart.USER_INITIAL_KEYBOARD_COMMANDS:
                sim_step_part_conf.add_user_step(self._sim_loop_initial_keyboard_commands)

            else:
                raise ValueError
            # end if
        # end for

        return sim_step_part_conf
    # end def

    def _sim_loop_predict(self):
        self._last_step_part = "Predict"
        self.f.predict()
    # end def

    def _sim_loop_update(self):
        self._last_step_part = "Update"

        if self._step < 0:
            self.f.update([])

        else:
            # Set current frame
            self.f.cur_frame = self._frames[self._step]

            # Predict and update
            self.f.update([np.array([det.x, det.y]) for det in self.f.cur_frame])
        # end if
    # end def

    def _sim_loop_prune(self):
        self._last_step_part = "Prune"

        if self._step >= 0:
            # Prune
            self.f.prune(trunc_thresh=self._trunc_thresh)
        # end if
    # end def

    def _sim_loop_merge(self):
        self._last_step_part = "Merge"

        if self._step >= 0:
            # Merge
            self.f.merge(merge_dist_measure=self._merge_dist_measure, merge_thresh=self._merge_thresh, max_components=self._max_components)
        # end if
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
        return GmPanjerPhdFilterSimulator._get_cov_ellipse(comp.cov, comp.loc, n_std, **kwargs)
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
                self._draw_plot = sns.kdeplot(x, y, shade=True, ax=self._ax, shade_lowest=False, cmap=self._draw_cmap, cbar=(not self._colorbar_is_added), zorder=zorder)  # Colorbar instead of label
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

    def get_fig_suptitle(self) -> str:
        return "GM-Panjer-PHD Filter Simulator"

    # end def

    def get_ax_title(self) -> str:
        return f"Sim-Step: {self._step if self._step >= 0 else '-'}, Sim-SubStep: {self._last_step_part}, # Est. States: " \
            f"{len(self._ext_states[-1]) if len(self._ext_states) > 0 else '-'}, # GMM-Components: {len(self.f.gmm)}, # GOSPA: " \
            f"{self._gospa_values[-1] if len(self._gospa_values) > 0 else '-':.04}"
    # end def

    def get_density_label(self) -> str:
        return "Panjer PHD intensity"
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

    def get_draw_layer_enum(self) -> DrawLayerBase:
        return DrawLayer
    # end def

    @staticmethod
    def get_help() -> str:
        return GmPanjerPhdFilterSimulatorConfig().help()
    # end def
# end class GmPanjerPhdFilterSimulator


class GmPanjerPhdFilterSimulatorConfig(GmPhdBaseFilterSimulatorConfig):
    def __init__(self):
        GmPhdBaseFilterSimulatorConfig.__init__(self)

        self._parser.description = "This is a simulator for the GM-Panjer-PHD filter."

        # General group
        # -> Nothing to add

        # PHD group - derived from filter group
        group = self._parser_groups["filter"]
        group.title = "Panjer PHD Filter - parameters for the Panjer PHD filter setup"

        # Data Simulator group
        # -> Nothing to add

        # Simulator group
        # -> Nothing to add, but helper functions implemented

        # File Reader group
        # -> Nothing to add

        # File Storage group
        # -> Nothing to add

        # Visualization group
        # -> Nothing to add
    # end def __init__

    @staticmethod
    def get_sim_step_part():
        return SimStepPart
    # end def

    @staticmethod
    def get_sim_loop_step_parts_default() -> List[SimStepPartBase]:
        return [SimStepPart.USER_INITIAL_KEYBOARD_COMMANDS, SimStepPart.USER_PREDICT, SimStepPart.USER_UPDATE, SimStepPart.USER_PRUNE, SimStepPart.USER_MERGE,
                SimStepPart.USER_EXTRACT_STATES, SimStepPart.USER_CALC_GOSPA,
                SimStepPart.DRAW, SimStepPart.WAIT_FOR_TRIGGER, SimStepPart.LOAD_NEXT_FRAME]
    # end def

    @staticmethod
    def _user_eval(s: str):
        return eval(s)
    # end def
# end class


def main(argv: List[str]):
    # Library settings
    sns.set(color_codes=True)

    # Initialize random generator
    random.seed(datetime.now())

    # Read command line arguments
    config = GmPanjerPhdFilterSimulatorConfig()
    args = config.read(argv[1:])

    # Update read parameters
    if args.birth_area is None:
        args.birth_area = args.fov
    # end if

    # Evaluate dynamic matrices
    if args.transition_model == TransitionModel.PCW_CONST_WHITE_ACC_MODEL_2xND:
        m = PcwConstWhiteAccelModelNd(dim=2, sigma=(args.sigma_accel_x, args.sigma_accel_y))

        args.f = m.eval_f(args.dt)
        args.q = m.eval_q(args.dt)
    # end if

    # Set the false alarm rate
    def get_state_space_volume_from_fov(fov) -> float:
        return (fov.x_max - fov.x_min) * (fov.y_max - fov.y_min)
    # end def

    if not args.rho_fa:
        args.rho_fa = args.n_fa / get_state_space_volume_from_fov(args.fov)
    else:
        factor = float(args.var_fa) / args.n_fa
        args.n_fa = int(args.rho_fa * get_state_space_volume_from_fov(args.fov))
        args.var_fa = int(args.n_fa * factor)
    # end if

    # Get data from a data provider
    if args.data_provider == DataProviderType.FILE_READER:
        scenario_data = ScenarioData().read_file(args.input)

        if not scenario_data.cross_check():
            print(f"Error while reading scenario data file {args.input}. For details see above. Program terminates.")
            return
        # end if

        # Convert data from certain coordinate systems to ENU, which is used internally
        if scenario_data.meta.coordinate_system == CoordSysConv.WGS84.value:
            scenario_data = Wgs84ToEnuConverter.convert(scenario_data, args.observer)
        # end if

    else:  # data_provider == DataProviderType.SIMULATOR
        scenario_data = ScenarioDataSimulator(f=args.f, q=args.q, dt=args.dt, t_max=args.sim_t_max, n_birth=args.n_birth, var_birth=args.var_birth, n_fa=args.n_fa, var_fa=args.var_fa,
                                              fov=args.fov, birth_area=args.birth_area,
                                              p_survival=args.p_survival, p_detection=args.p_detection, birth_dist=args.birth_dist, sigma_vel_x=args.sigma_vel_x, sigma_vel_y=args.sigma_vel_y,
                                              birth_gmm=args.birth_gmm).run()
    # end if

    sim = GmPanjerPhdFilterSimulator(scenario_data=scenario_data, output_coord_system_conversion=args.output_coord_system_conversion, fn_out=args.output,
                                     fn_out_video=args.output_video,
                                     auto_step_interval=args.auto_step_interval, auto_step_autostart=args.auto_step_autostart, fov=args.fov, birth_area=args.birth_area,
                                     limits_mode=args.limits_mode, observer=args.observer, logging=args.verbosity,
                                     birth_gmm=args.birth_gmm, var_birth=args.var_birth, p_survival=args.p_survival, p_detection=args.p_detection,
                                     f=args.f, q=args.q, h=args.h, r=args.r, rho_fa=args.rho_fa, gate_thresh=args.gate_thresh,
                                     trunc_thresh=args.trunc_thresh, merge_dist_measure=args.merge_dist_measure, merge_thresh=args.merge_thresh, max_components=args.max_components,
                                     ext_states_bias=args.ext_states_bias, ext_states_use_integral=args.ext_states_use_integral, gospa_c=args.gospa_c, gospa_p=args.gospa_p,
                                     gui=args.gui, density_draw_style=args.density_draw_style, n_samples_density_map=args.n_samples_density_map, n_bins_density_map=args.n_bins_density_map,
                                     draw_layers=args.draw_layers, sim_loop_step_parts=args.sim_loop_step_parts, show_legend=args.show_legend, show_colorbar=args.show_colorbar,
                                     start_window_max=args.start_window_max, init_kbd_cmds=args.init_kbd_cmds)

    sim.fn_out_seq_max = args.output_seq_max
    sim.fn_out_fill_gaps = args.output_fill_gaps

    sim.run()
# end def main


if __name__ == "__main__":
    main(sys.argv)
# end if
