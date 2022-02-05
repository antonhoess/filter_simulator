#!/usr/bin/env python3

from __future__ import annotations
from typing import List, Optional, Union, Dict
import sys
import random
import numpy as np
from datetime import datetime
import seaborn as sns
from enum import auto
import math

from filter_simulator.common import Logging, Limits, Position
from filter_simulator.filter_simulator import SimStepPartConf
from scenario_data.scenario_data_converter import CoordSysConv, Wgs84ToEnuConverter
from filter_simulator.dyn_matrix import TransitionModel, PcwConstWhiteAccelModelNd
from gm_panjer_phd_filter import GmPanjerPhdFilter
from gm import Gmm
from gm import GmComponent, DistMeasure
from scenario_data.scenario_data_simulator import ScenarioDataSimulator
from filter_simulator.window_helper import LimitsMode
from scenario_data.scenario_data import ScenarioData
from base_filter_simulator import SimStepPartBase, AdditionalAxisBase
from gm_phd_base_filter_simulator import GmPhdBaseFilterSimulator, GmPhdBaseFilterSimulatorConfig, GmPhdBaseFilterSimulatorConfigSettings
from gm_phd_base_filter_simulator import DrawLayer, DensityDrawStyle, DataProviderType


class SimStepPart(SimStepPartBase):
    DRAW = auto()  # Draw the current scene
    WAIT_FOR_TRIGGER = auto()  # Wait for user input to continue with the next step
    LOAD_NEXT_FRAME = auto()  # Load the next data (frame)
    USER_PREDICT = auto()
    USER_UPDATE = auto()
    USER_PRUNE = auto()
    USER_MERGE = auto()
    USER_EXTRACT_STATES = auto()
    USER_CALC_GOSPA = auto()
    USER_INITIAL_KEYBOARD_COMMANDS = auto()
# end class


class AdditionalAxis(AdditionalAxisBase):
    NONE = auto()
    GOSPA = auto()
    VARIANCE = auto()
# end class


class GmPanjerPhdFilterSimulator(GmPhdBaseFilterSimulator):
    def __init__(self, s: GmPanjerPhdFilterSimulatorConfigSettings) -> None:
        s_sup = GmPhdBaseFilterSimulatorConfigSettings.from_obj(s)
        GmPhdBaseFilterSimulator.__init__(self, s_sup)

        self.f = GmPanjerPhdFilter(birth_gmm=s.birth_gmm, var_birth=s.var_birth, survival=s.p_survival, detection=s.p_detection, f=s.f, q=s.q,
                                   h=s.h, r=s.r, rho_fa=s.rho_fa, gate_thresh=s.gate_thresh, logging=s.verbosity)

        # --
        self._variance_values: List[float] = list()
    # end def

    def restart(self):
        self._variance_values.clear()

        GmPhdBaseFilterSimulator.restart(self)
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
            self._variance_values.append(self.f.variance)
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

    def get_fig_suptitle(self) -> str:
        return "GM-Panjer-PHD Filter Simulator"

    # end def

    def get_ax_title(self) -> str:
        n_states_real = sum(map(lambda gtt: gtt.begin_step <= self._step < gtt.begin_step + len(gtt.points), self._scenario_data.gtts))  # Real number of targets
        n_targets = self.f.gmm.get_total_weight()  # Estimated targets by the filter intensity

        return f"Sim-Step: {self._step if self._step >= 0 else '-'}, Sim-SubStep: {self._last_step_part}, # Est. States: " \
            f"{len(self._ext_states[-1]) if len(self._ext_states) > 0 else '-'} ({n_targets:.02f} / {n_states_real}), # GMM-Components: {len(self.f.gmm)}, # Variance: {self.f.variance:.2f}, # GOSPA: " \
            f"{self._gospa_values[-1] if len(self._gospa_values) > 0 else '-':.04}"
    # end def

    def get_density_label(self) -> str:
        return "Panjer PHD intensity"
    # end def

    @staticmethod
    def get_additional_axis_enum() -> AdditionalAxisBase:
        return AdditionalAxis
    # end def

    @staticmethod
    def get_additional_axis_by_short_name(short_name: str) -> Optional[AdditionalAxisBase]:
        if short_name == "g":
            return AdditionalAxis.GOSPA

        elif short_name == "v":
            return AdditionalAxis.VARIANCE
        # end if

        return None
    # end def

    def do_additional_axis_plot(self, axis: AdditionalAxisBase) -> bool:
        if axis is AdditionalAxis.VARIANCE:
            self._ax_add.clear()
            self._ax_add.plot(self._variance_values)

            self._ax_add.set_title(f"Variance in # of targets", fontsize=8)
            self._ax_add.set_xlabel("Simulation step", fontsize=8)
            self._ax_add.set_ylabel("Variance", fontsize=8)

            return True

        else:
            return False
        # end if
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

    # XXX Temp. for testing purposes
    @staticmethod
    def get_sim_loop_step_parts_default2() -> List[SimStepPartBase]:
        return [SimStepPart.USER_INITIAL_KEYBOARD_COMMANDS,
                SimStepPart.LOAD_NEXT_FRAME, SimStepPart.DRAW, SimStepPart.WAIT_FOR_TRIGGER,
                SimStepPart.USER_PREDICT, SimStepPart.USER_UPDATE, SimStepPart.USER_PRUNE, SimStepPart.USER_MERGE,
                SimStepPart.USER_EXTRACT_STATES, SimStepPart.USER_CALC_GOSPA]
    # end def
    @staticmethod
    def get_sim_loop_step_parts_default3() -> List[SimStepPartBase]:
        return [SimStepPart.USER_INITIAL_KEYBOARD_COMMANDS,
                SimStepPart.DRAW, SimStepPart.WAIT_FOR_TRIGGER,
                SimStepPart.USER_PREDICT,
                SimStepPart.DRAW, SimStepPart.WAIT_FOR_TRIGGER,
                SimStepPart.USER_UPDATE,
                SimStepPart.DRAW, SimStepPart.WAIT_FOR_TRIGGER,
                SimStepPart.USER_PRUNE,
                SimStepPart.DRAW, SimStepPart.WAIT_FOR_TRIGGER,
                SimStepPart.USER_MERGE,
                SimStepPart.DRAW, SimStepPart.WAIT_FOR_TRIGGER,
                SimStepPart.USER_EXTRACT_STATES,
                SimStepPart.DRAW, SimStepPart.WAIT_FOR_TRIGGER,
                SimStepPart.USER_CALC_GOSPA,
                SimStepPart.DRAW, SimStepPart.WAIT_FOR_TRIGGER, SimStepPart.LOAD_NEXT_FRAME]
    # end def

    @staticmethod
    def _user_eval(s: str):
        return eval(s)
    # end def
# end class


class GmPanjerPhdFilterSimulatorConfigSettings(GmPhdBaseFilterSimulatorConfigSettings):
    def __init__(self):
        super().__init__()

        # Filter group - subclasses will rename it
        self._add_attribute("var_birth", float)
    # end def
# end def


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

        args.f = m.eval_f(args.delta_t)
        args.q = m.eval_q(args.delta_t)
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

    if args.n_birth > args.var_birth:
        var_birth_ori = args.var_birth
        alphaval = args.n_birth ** 2 / (args.var_birth - args.n_birth)
        alphaval = math.floor(alphaval)  # Must not be a non-integer!
        args.var_birth = args.n_birth + args.n_birth ** 2 / alphaval

        if var_birth_ori != args.var_birth:
            print(f"Warning: binomial birth. Adjusted variance to {args.var_birth:g}\n")
        # end if
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
        scenario_data = ScenarioDataSimulator(f=args.f, q=args.q, dt=args.delta_t, t_max=args.sim_t_max, n_birth=args.n_birth, var_birth=args.var_birth, n_fa=args.n_fa, var_fa=args.var_fa,
                                              fov=args.fov, birth_area=args.birth_area,
                                              p_survival=args.p_survival, p_detection=args.p_detection, birth_dist=args.birth_dist, sigma_vel_x=args.sigma_vel_x, sigma_vel_y=args.sigma_vel_y,
                                              birth_gmm=args.birth_gmm).run()
    # end if

    args.scenario_data = scenario_data
    del args.input
    del args.data_provider

    # Run the simulator
    s = GmPanjerPhdFilterSimulatorConfigSettings.from_obj(args)
    sim = GmPanjerPhdFilterSimulator(s)
    sim.run()
# end def main


if __name__ == "__main__":
    main(sys.argv)
# end if
