#!/usr/bin/env python3

from __future__ import annotations
from typing import List, Optional, Union
import sys
import random
import numpy as np
from datetime import datetime
import seaborn as sns
from enum import auto

from filter_simulator.common import Logging, Limits, Position
from filter_simulator.filter_simulator import SimStepPartConf
from scenario_data.scenario_data_converter import CoordSysConv, Wgs84ToEnuConverter
from filter_simulator.dyn_matrix import TransitionModel, PcwConstWhiteAccelModelNd
from gm_phd_filter import GmPhdFilter, GmComponent, DistMeasure
from scenario_data.scenario_data_simulator import ScenarioDataSimulator
from filter_simulator.window_helper import LimitsMode
from scenario_data.scenario_data import ScenarioData
from base_filter_simulator import SimStepPartBase, AdditionalAxisBase
from gm_phd_base_filter_simulator import GmPhdBaseFilterSimulator, GmPhdBaseFilterSimulatorConfig
from gm_phd_base_filter_simulator import DrawLayer, DensityDrawStyle, DataProviderType


class SimStepPart(SimStepPartBase):
    DRAW = auto()  # Draw the current scene
    WAIT_FOR_TRIGGER = auto()  # Wait for user input to continue with the next step
    LOAD_NEXT_FRAME = auto()  # Load the next data (frame)
    USER_PREDICT_AND_UPDATE = auto()
    USER_PRUNE = auto()
    USER_EXTRACT_STATES = auto()
    USER_CALC_GOSPA = auto()
    USER_INITIAL_KEYBOARD_COMMANDS = auto()
# end class


class AdditionalAxis(AdditionalAxisBase):
    NONE = auto()
    GOSPA = auto()
# end class


class GmPhdFilterSimulator(GmPhdBaseFilterSimulator):
    def __init__(self, scenario_data: ScenarioData, output_coord_system_conversion: CoordSysConv,
                 fn_out: str, fn_out_video: Optional[str], auto_step_interval: int, auto_step_autostart: bool, fov: Limits, birth_area: Limits, limits_mode: LimitsMode, observer: Position, logging: Logging,
                 birth_gmm: List[GmComponent], p_survival: float, p_detection: float,
                 f: np.ndarray, q: np.ndarray, h: np.ndarray, r: np.ndarray, rho_fa: float, gate_thresh: Optional[float],
                 trunc_thresh: float, merge_dist_measure: DistMeasure, merge_thresh: float, max_components: int,
                 ext_states_bias: float, ext_states_use_integral: bool,
                 gospa_c: float, gospa_p: int,
                 gui: bool, density_draw_style: DensityDrawStyle, n_samples_density_map: int, n_bins_density_map: int,
                 draw_layers: Optional[List[DrawLayer]], sim_loop_step_parts: List[SimStepPart], show_legend: Optional[Union[int, str]], show_colorbar: bool, start_window_max: bool,
                 init_kbd_cmds: List[str]):

        GmPhdBaseFilterSimulator.__init__(self, scenario_data, output_coord_system_conversion,
                                          fn_out, fn_out_video, auto_step_interval, auto_step_autostart, fov, birth_area, limits_mode, observer, logging,
                                          trunc_thresh, merge_dist_measure, merge_thresh, max_components,
                                          ext_states_bias, ext_states_use_integral,
                                          gospa_c, gospa_p,
                                          gui, density_draw_style, n_samples_density_map, n_bins_density_map,
                                          draw_layers, sim_loop_step_parts, show_legend, show_colorbar, start_window_max,
                                          init_kbd_cmds)

        self.f = GmPhdFilter(birth_gmm=birth_gmm, survival=p_survival, detection=p_detection, f=f, q=q, h=h, r=r, rho_fa=rho_fa, gate_thresh=gate_thresh, logging=logging)
    # end def

    def _set_sim_loop_step_part_conf(self):
        # Configure the processing steps
        sim_step_part_conf = SimStepPartConf()

        for step_part in self._sim_loop_step_parts:
            if step_part is SimStepPart.DRAW:
                sim_step_part_conf.add_draw_step()

            elif step_part is SimStepPart.WAIT_FOR_TRIGGER:
                sim_step_part_conf.add_wait_for_trigger_step()

            elif step_part is SimStepPart.LOAD_NEXT_FRAME:
                sim_step_part_conf.add_load_next_frame_step()

            elif step_part is SimStepPart.USER_PREDICT_AND_UPDATE:
                sim_step_part_conf.add_user_step(self._sim_loop_predict_and_update)

            elif step_part is SimStepPart.USER_PRUNE:
                sim_step_part_conf.add_user_step(self._sim_loop_prune)

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

    def _sim_loop_predict_and_update(self):
        self._last_step_part = "Predict + Update"

        if self._step < 0:
            self.f.predict_and_update([])

        else:
            # Set current frame
            self.f.cur_frame = self._frames[self._step]

            # Predict and update
            self.f.predict_and_update([np.array([det.x, det.y]) for det in self.f.cur_frame])
        # end if
    # end def

    def _sim_loop_prune(self):
        self._last_step_part = "Prune"

        if self._step >= 0:
            # Prune
            self.f.prune(trunc_thresh=self._trunc_thresh, merge_dist_measure=self._merge_dist_measure, merge_thresh=self._merge_thresh, max_components=self._max_components)
        # end if
    # end def

    def get_fig_suptitle(self) -> str:
        return "GM-PHD Filter Simulator"
    # end def

    def get_ax_title(self) -> str:
        return f"Sim-Step: {self._step if self._step >= 0 else '-'}, Sim-SubStep: {self._last_step_part}, # Est. States: " \
            f"{len(self._ext_states[-1]) if len(self._ext_states) > 0 else '-'}, # GMM-Components: {len(self.f.gmm)}, # GOSPA: " \
            f"{self._gospa_values[-1] if len(self._gospa_values) > 0 else '-':.04}"
    # end def

    def get_density_label(self) -> str:
        return "PHD intensity"
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
        return GmPhdFilterSimulatorConfig().help()
    # end def
# end class GmPhdFilterSimulator


class GmPhdFilterSimulatorConfig(GmPhdBaseFilterSimulatorConfig):
    def __init__(self):
        GmPhdBaseFilterSimulatorConfig.__init__(self)

        self._parser.description = "This is a simulator for the GM-PHD filter."

        # General group
        # -> Nothing to add

        # PHD group - derived from filter group
        group = self._parser_groups["filter"]
        group.title = "PHD Filter - parameters for the PHD filter setup"

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
        return [SimStepPart.USER_INITIAL_KEYBOARD_COMMANDS, SimStepPart.USER_PREDICT_AND_UPDATE, SimStepPart.USER_PRUNE,
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
    config = GmPhdFilterSimulatorConfig()
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

    sim = GmPhdFilterSimulator(scenario_data=scenario_data, output_coord_system_conversion=args.output_coord_system_conversion, fn_out=args.output,
                               fn_out_video=args.output_video,
                               auto_step_interval=args.auto_step_interval, auto_step_autostart=args.auto_step_autostart, fov=args.fov, birth_area=args.birth_area,
                               limits_mode=args.limits_mode, observer=args.observer, logging=args.verbosity,
                               birth_gmm=args.birth_gmm, p_survival=args.p_survival, p_detection=args.p_detection,
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
