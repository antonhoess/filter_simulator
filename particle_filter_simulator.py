#!/usr/bin/env python3

from __future__ import annotations
from typing import List, Optional, Union
import sys
import random
import numpy as np
from datetime import datetime
from matplotlib.patches import Ellipse
import seaborn as sns
from sklearn.cluster import MeanShift
from enum import auto

from filter_simulator.common import Logging, Limits, Position
from filter_simulator.filter_simulator import SimStepPartConf
from filter_simulator.window_helper import LimitsMode
from scenario_data.scenario_data_converter import CoordSysConv, Wgs84ToEnuConverter
from particle_filter import ParticleFilter
from scenario_data.scenario_data import ScenarioData
from base_filter_simulator import BaseFilterSimulatorConfig, BaseFilterSimulatorConfigSettings, BaseFilterSimulator, DrawLayerBase, SimStepPartBase, DensityDrawStyleBase, AdditionalAxisBase


class DrawLayer(DrawLayerBase):
    DENSITY_MAP = auto()
    FOV = auto()
    ALL_TRAJ_LINE = auto()
    ALL_TRAJ_POS = auto()
    UNTIL_TRAJ_LINE = auto()
    UNTIL_TRAJ_POS = auto()
    ALL_DET = auto()
    ALL_DET_CONN = auto()
    UNTIL_MISSED_DET = auto()
    UNTIL_FALSE_ALARM = auto()
    PARTICLES = auto()
    IMPORTANCE_WEIGHT_COV = auto()
    CUR_EST_STATE = auto()
    UNTIL_EST_STATE = auto()
    CUR_DET = auto()
# end class


class SimStepPart(SimStepPartBase):
    DRAW = auto()  # Draw the current scene
    WAIT_FOR_TRIGGER = auto()  # Wait for user input to continue with the next step
    LOAD_NEXT_FRAME = auto()  # Load the next data (frame)
    USER_PREDICT = auto()
    USER_UPDATE = auto()
    USER_RESAMPLE = auto()
    USER_EXTRACT_STATES = auto()
    USER_CALC_GOSPA = auto()
    USER_INITIAL_KEYBOARD_COMMANDS = auto()
# end class


class DensityDrawStyle(DensityDrawStyleBase):
    NONE = auto()
    KDE = auto()
    EVAL = auto()
# end class


class AdditionalAxis(AdditionalAxisBase):
    NONE = auto()
    GOSPA = auto()
# end class


class ParticleFilterSimulator(BaseFilterSimulator):
    def __init__(self, s: ParticleFilterSimulatorConfigSettings) -> None:
        self.__sim_loop_step_parts: List[SimStepPart] = s.sim_loop_step_parts  # Needs to be set before calling the contructor of the FilterSimulator, since it already needs this values there

        s_sup = BaseFilterSimulatorConfigSettings.from_obj(s)
        BaseFilterSimulator.__init__(self, s_sup)

        self.f = ParticleFilter(n_part=s.n_particles, s_gauss=s.sigma_gauss_kernel, noise=s.particle_movement_noise, speed=s.speed, fov=s.fov, logging=s.verbosity)

        self._ms_bandwidth: float = s.ext_states_ms_bandwidth
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

            elif step_part is SimStepPart.USER_RESAMPLE:
                sim_step_part_conf.add_user_step(self._sim_loop_resample)

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
        # Set current frame
        self.f.cur_frame = self._frames[self._step]

        self._last_step_part = "Predict"
        self.f.predict()
    # end def

    def _sim_loop_update(self):
        self._last_step_part = "Update"
        self.f.update()
    # end def

    def _sim_loop_resample(self):
        self._last_step_part = "Resample"
        self.f.resample()
    # end def

    def _sim_loop_extract_states(self):
        self._last_step_part = "Extract States"

        # Calculate mean shift
        clustering: bool = True
        if clustering:
            cluster_samples: np.array = np.array([[p.x, p.y] for p in self.f.particles])
            clust: MeanShift = MeanShift(bandwidth=self._ms_bandwidth).fit(cluster_samples)

            # Extract states
            ext_states = clust.cluster_centers_
            self._remove_duplikate_states(ext_states)
            self._ext_states.append(ext_states)
            self._logging.print_verbose(Logging.DEBUG, clust.labels_)
            self._logging.print_verbose(Logging.DEBUG, clust.cluster_centers_)
        # end if
    # end def

    def _calc_density(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        accum = np.zeros(x.shape)

        for p in self.f.particles:
            d_x: float = p.x - x
            d_y: float = p.y - y
            p_d: float = 1. / np.sqrt(d_x * d_x + d_y * d_y)
            accum += p_d
        # end for

        return accum
    # end def

    def _draw_density_map(self, zorder):
        # Draw density map
        if self._density_draw_style == DensityDrawStyle.KDE:
            if len(self.f.particles) > 0:
                x = [p.x for p in self.f.particles]
                y = [p.y for p in self.f.particles]
                self._draw_plot = sns.kdeplot(x=x, y=y, shade=True, ax=self._ax, thresh=0.05, cmap=self._draw_cmap, cbar=(self._show_colorbar and not self._colorbar_is_added), cbar_ax=self._cax, zorder=zorder)  # Colorbar instead of label
                self._colorbar_is_added = True
            # end if

        elif self._density_draw_style == DensityDrawStyle.EVAL:
            x, y, z = self._calc_density_map(self._draw_limits, grid_res=self._n_bins_density_map)
            self._draw_plot = self._ax.contourf(x, y, z, 100, cmap=self._draw_cmap, zorder=zorder)  # Colorbar instead of label

        else:  # DensityDrawStyle.NONE:
            pass
        # end if
    # end def

    def _eval_at(self, x: float, y: float) -> float:
        return self._calc_density(np.asarray([x]), np.asarray([x]))[0]
    # end def

    def _draw_particles(self, zorder):
        # Particles
        self._ax.scatter([p.x for p in self.f.particles], [p.y for p in self.f.particles], s=5, edgecolor="blue", marker="o", zorder=zorder, label="part. ($t_{k}$)")
    # end def

    def _draw_importance_weight_cov(self, zorder):
        if self.f.cur_frame is not None:
            ell_radius_x: float = self.f.s_gauss
            ell_radius_y: float = self.f.s_gauss

            for det in self.f.cur_frame:
                ellipse: Ellipse = Ellipse((det.x, det.y), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor='none', edgecolor="black", linewidth=.5, zorder=zorder,
                                           label="I.W. cov. ($t_{k}$)")
                self._ax.add_patch(ellipse)
            # end for
        # end if
    # end def

    def get_fig_suptitle(self) -> str:
        return "Particle Filter Simulator"
    # end def

    def get_ax_title(self) -> str:
        n_states_real = sum(map(lambda gtt: gtt.begin_step <= self._step < gtt.begin_step + len(gtt.points), self._scenario_data.gtts))  # Real number of targets

        return f"Sim-Step: {self._step if self._step >= 0 else '-'}, Sim-SubStep: {self._last_step_part}, # Est. States: " \
            f"{len(self._ext_states[-1]) if len(self._ext_states) > 0 else '-'} ({n_states_real}), # GOSPA: " \
            f"{self._gospa_values[-1] if len(self._gospa_values) > 0 else '-':.04}"
    # end def

    def get_density_label(self) -> str:
        return "Intensity"
    # end def

    def get_draw_routine_by_layer(self, layer: DrawLayerBase):
        if layer == DrawLayer.DENSITY_MAP:
            return self._draw_density_map

        elif layer == DrawLayer.FOV:
            return self._draw_fov

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

        elif layer == DrawLayer.CUR_MISSED_DET:
            return self._draw_cur_missed_det

        elif layer == DrawLayer.UNTIL_MISSED_DET:
            return self._draw_until_missed_det

        elif layer == DrawLayer.CUR_FALSE_ALARM:
            return self._draw_cur_false_alarm

        elif layer == DrawLayer.UNTIL_FALSE_ALARM:
            return self._draw_until_false_alarm

        elif layer == DrawLayer.PARTICLES:
            return self._draw_particles

        elif layer == DrawLayer.IMPORTANCE_WEIGHT_COV:
            return self._draw_importance_weight_cov

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

    @staticmethod
    def get_additional_axis_enum() -> AdditionalAxisBase:
        return AdditionalAxis
    # end def

    @staticmethod
    def get_additional_axis_by_short_name(short_name: str) -> Optional[AdditionalAxisBase]:
        if short_name == "g":
            return AdditionalAxis.GOSPA
        # end if

        return None
    # end def

    def do_additional_axis_plot(self, axis: AdditionalAxisBase) -> bool:
        # No additional axes
        return False
    # end def

    @staticmethod
    def get_help() -> str:
        return ParticleFilterSimulatorConfig().help()
    # end def
# end class ParticleFilterSimulator


class ParticleFilterSimulatorConfig(BaseFilterSimulatorConfig):
    def __init__(self):
        BaseFilterSimulatorConfig.__init__(self)

        self._parser.description = "This is a simulator for the particle-filter."

        # General group
        # -> Nothing to add

        # Particle filter group - derived from filter group
        group = self._parser_groups["filter"]
        group.title = "Particle Filter - parameters for the particle filter setup"

        group.add_argument("--sigma_gauss_kernel", metavar="[>0.0 - N]", type=BaseFilterSimulatorConfig.InRange(float, min_ex_val=.0), default=20.,
                           help="Sets sigma of the Gaussian importance weight kernel to SIGMA_GAUSS_KERNEL.")

        group.add_argument("--n_particles", metavar="[100 - N]", type=BaseFilterSimulatorConfig.InRange(int, min_val=100), default=100,
                           help="Sets the particle filter's number of particles to N_PARTICLES.")

        group.add_argument("--particle_movement_noise", metavar="[>0.0 - N]", type=BaseFilterSimulatorConfig.InRange(float, min_ex_val=.0), default=.1,
                           help="Sets the particle's movement noise to PARTICLE_MOVEMENT_NOISE")

        group.add_argument("--speed", metavar="[0.0 - 1.0]", type=BaseFilterSimulatorConfig.InRange(float, min_val=.0, max_val=1.), default=1.,
                           help="Sets the speed the particles move towards their nearest detection to SPEED.")

        group.add_argument("--ext_states_ms_bandwidth", metavar="[>0.0 - N]", type=BaseFilterSimulatorConfig.InRange(float, min_ex_val=.0), default=None,
                           help="Sets the mean shift bandwidth used in the RBF kernel for extracting the current states to EXT_STATES_MS_BANDWIDTH. If not given, the bandwidth is estimated.")

        # Simulator group
        # -> Nothing to add, but helper functions implemented

        # File Reader group
        # -> Nothing to add

        # File Storage group
        # -> Nothing to add

        # Visualization group
        group = self._parser_groups["visualization"]

        group.add_argument("--density_draw_style", action=self._EvalAction, comptype=DensityDrawStyleBase, user_eval=self._user_eval, choices=[str(t) for t in DensityDrawStyle],
                           default=DensityDrawStyle.NONE,
                           help=f"Sets the drawing style to visualizing the density/intensity map. Possible values are: {str(DensityDrawStyle.KDE)} (kernel density estimator) and "
                                f"{str(DensityDrawStyle.EVAL)} (evaluate the correct value for each cell in a grid).")

        group.add_argument("--draw_layers", metavar=f"[{{{  ','.join([str(t) for t in DrawLayer]) }}}*]", action=self._EvalListAction, comptype=DrawLayerBase, user_eval=self._user_eval,
                           default=[ly for ly in DrawLayer if ly not in [DrawLayer.ALL_TRAJ_LINE, DrawLayer.ALL_TRAJ_POS, DrawLayer.ALL_DET, DrawLayer.ALL_DET_CONN]],
                           help=f"Sets the list of drawing layers. Allows to draw only the required layers and in the desired order. If not set, a fixes set of layers are drawn in a fixed order. "
                           f"Example 1: [{str(DrawLayer.DENSITY_MAP)}, {str(DrawLayer.PARTICLES)}]\n"
                           f"Example 2: [layer for layer in DrawLayer if not layer == {str(DrawLayer.ALL_DET)} and not layer == {str(DrawLayer.ALL_DET_CONN)}]")
    # end def __init__

    @staticmethod
    def get_sim_step_part():
        return SimStepPart
    # end def

    @staticmethod
    def get_sim_loop_step_parts_default() -> List[SimStepPartBase]:
        return [SimStepPart.USER_INITIAL_KEYBOARD_COMMANDS,
                SimStepPart.DRAW, SimStepPart.WAIT_FOR_TRIGGER, SimStepPart.LOAD_NEXT_FRAME,
                SimStepPart.USER_PREDICT, SimStepPart.USER_UPDATE, SimStepPart.USER_RESAMPLE, SimStepPart.USER_EXTRACT_STATES, SimStepPart.USER_CALC_GOSPA]
    # end def

    @staticmethod
    def _user_eval(s: str):
        return eval(s)
    # end def
# end class


class ParticleFilterSimulatorConfigSettings(BaseFilterSimulatorConfigSettings):
    def __init__(self):
        super().__init__()

        # Filter group - subclasses will rename it
        self._add_attribute("sigma_gauss_kernel", float)
        self._add_attribute("n_particles", int)
        self._add_attribute("particle_movement_noise", float)
        self._add_attribute("speed", float)
        self._add_attribute("ext_states_ms_bandwidth", float, nullable=True)
    # end def
# end def


def main(argv: List[str]):
    # Library settings
    sns.set(color_codes=True)

    # Initialize random generator
    random.seed(datetime.now())

    # Read command line arguments
    config = ParticleFilterSimulatorConfig()
    args = config.read(argv[1:])

    # Get data from a data provider
    scenario_data = ScenarioData().read_file(args.input)

    if not scenario_data.cross_check():
        print(f"Error while reading scenario data file {args.input}. For details see above. Program terminates.")
        return
    # end if

    # Convert data from certain coordinate systems to ENU, which is used internally
    if scenario_data.meta.coordinate_system == CoordSysConv.WGS84.value:
        scenario_data = Wgs84ToEnuConverter.convert(scenario_data, args.observer_position)
    # end if

    args.scenario_data = scenario_data

    del args.input

    # Run the simulator
    s = ParticleFilterSimulatorConfigSettings.from_obj(args)
    sim = ParticleFilterSimulator(s)
    sim.run()
# end def main


if __name__ == "__main__":
    main(sys.argv)
# end if
