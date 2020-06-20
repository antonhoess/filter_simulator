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
import argparse

from filter_simulator.common import Logging, Limits, Position
from filter_simulator.filter_simulator import SimStepPartConf, FilterSimulatorConfig
from filter_simulator.window_helper import LimitsMode
from scenario_data.scenario_data_converter import CoordSysConv, Wgs84ToEnuConverter
from particle_filter import ParticleFilter
from scenario_data.scenario_data import ScenarioData
from base_filter_simulator import BaseFilterSimulator, AdditionalAxis, DrawLayerBase, SimStepPartBase, DensityDrawStyleBase


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
    DRAW = 0  # Draw the current scene
    WAIT_FOR_TRIGGER = 1  # Wait for user input to continue with the next step
    LOAD_NEXT_FRAME = 2  # Load the next data (frame)
    USER_PREDICT = 3
    USER_UPDATE = 4
    USER_RESAMPLE = 5
    USER_EXTRACT_STATES = 6
    USER_CALC_GOSPA = 7
    USER_INITIAL_KEYBOARD_COMMANDS = 8
# end class


class DensityDrawStyle(DensityDrawStyleBase):
    NONE = 0
    KDE = 1
    EVAL = 2
# end class


class ParticleFilterSimulator(BaseFilterSimulator):
    def __init__(self, scenario_data: ScenarioData, output_coord_system_conversion: CoordSysConv,
                 fn_out: str, fn_out_video: Optional[str], auto_step_interval: int, auto_step_autostart: bool, fov: Limits, limits_mode: LimitsMode, observer: Position, logging: Logging,
                 ext_states_ms_bandwidth: float, gospa_c: float, gospa_p: int, n_particles: int, sigma_gauss_kernel: float, particle_movement_noise: float, speed: float,
                 gui: bool, density_draw_style: DensityDrawStyle, n_bins_density_map: int,
                 draw_layers: Optional[List[DrawLayer]], sim_loop_step_parts: List[SimStepPart], show_legend: Optional[Union[int, str]], show_colorbar: bool, start_window_max: bool,
                 init_kbd_cmds: List[str]):

        self.__sim_loop_step_parts: List[SimStepPart] = sim_loop_step_parts  # Needs to be set before calling the contructor of the FilterSimulator, since it already needs this values there

        BaseFilterSimulator.__init__(self, scenario_data, output_coord_system_conversion,
                                     fn_out, fn_out_video, auto_step_interval, auto_step_autostart, fov, limits_mode, observer, logging,
                                     gospa_c, gospa_p,
                                     gui, density_draw_style, n_bins_density_map,
                                     draw_layers, sim_loop_step_parts, show_legend, show_colorbar, start_window_max,
                                     init_kbd_cmds)

        self.f = ParticleFilter(n_particles, sigma_gauss_kernel, particle_movement_noise, speed, fov=fov, logging=logging)

        self._ms_bandwidth: float = ext_states_ms_bandwidth
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

    def _calc_density(self, x: np.ndarray, y: np.ndarray) -> float:
        accum: float = 0.

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
                self._draw_plot = sns.kdeplot(x, y, shade=True, ax=self._ax, shade_lowest=False, cmap=self._draw_cmap, cbar=(not self._colorbar_is_added), zorder=zorder)  # Colorbar instead of label
                self._colorbar_is_added = True
            # end if

        elif self._density_draw_style == DensityDrawStyle.EVAL:
            x, y, z = self._calc_density_map(self._draw_limits, grid_res=self._n_bins_density_map)
            self._draw_plot = self._ax.contourf(x, y, z, 100, cmap=self._draw_cmap, zorder=zorder)  # Colorbar instead of label

        else:  # DensityDrawStyle.NONE:
            pass
        # end if
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
        return f"Sim-Step: {self._step if self._step >= 0 else '-'}, Sim-SubStep: {self._last_step_part}, # Est. States: " \
            f"{len(self._ext_states[-1]) if len(self._ext_states) > 0 else '-'}, # GOSPA: " \
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

        elif layer == DrawLayer.UNTIL_MISSED_DET:
            return self._draw_until_missed_det

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

    def get_draw_layer_enum(self) -> DrawLayerBase:
        return DrawLayer
    # end def
# end class ParticleFilterSimulator


class ParticleFilterSimulatorConfig(FilterSimulatorConfig):
    def __init__(self):
        FilterSimulatorConfig.__init__(self)

        self._parser = argparse.ArgumentParser(add_help=False, formatter_class=self._ArgumentDefaultsRawDescriptionHelpFormatter, description="This is a simulator for the particle-filter.")

        # General group
        group = self._parser.add_argument_group('General - common program settings')

        group.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS,
                           help="Shows this help and exits the program.")

        group.add_argument("--fov", metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"), action=self._LimitsAction, type=float, nargs=4, default=Limits(-10, -10, 10, 10),
                           help="Sets the Field of View (FoV) of the scene.")

        group.add_argument("--limits_mode", action=self._EvalAction, comptype=LimitsMode, choices=[str(t) for t in LimitsMode], default=LimitsMode.FOV_INIT_ONLY,
                           help="Sets the limits mode, which defines how the limits for the plotting window are set initially and while updating the plot.")

        group.add_argument("--verbosity", action=self._EvalAction, comptype=Logging, choices=[str(t) for t in Logging], default=Logging.INFO,
                           help="Sets the programs verbosity level to VERBOSITY. If set to Logging.NONE, the program will be silent.")

        group.add_argument("--observer_position", dest="observer", metavar=("LAT", "LON"), action=self._PositionAction, type=float, nargs=2, default=None,
                           help="Sets the geodetic position of the observer in WGS84 to OBSERVER_POSITION. Can be used instead of the automatically used center of all detections or in case of only "
                                "manually creating detections, which needed to be transformed back to WGS84.")

        # Particle filter group
        group = self._parser.add_argument_group('Particle Filter - parameters for the particle filter setup')

        group.add_argument("--sigma_gauss_kernel", metavar="[>0.0 - N]", type=ParticleFilterSimulatorConfig.InRange(float, min_ex_val=.0), default=20.,
                           help="Sets sigma of the Gaussian importance weight kernel to SIGMA_GAUSS_KERNEL.")

        group.add_argument("--n_particles", metavar="[100 - N]", type=ParticleFilterSimulatorConfig.InRange(int, min_val=100), default=100,
                           help="Sets the particle filter's number of particles to N_PARTICLES.")

        group.add_argument("--particle_movement_noise", metavar="[>0.0 - N]", type=ParticleFilterSimulatorConfig.InRange(float, min_ex_val=.0), default=.1,
                           help="Sets the particle's movement noise to PARTICLE_MOVEMENT_NOISE")

        group.add_argument("--speed", metavar="[0.0 - 1.0]", type=ParticleFilterSimulatorConfig.InRange(float, min_val=.0, max_val=1.), default=1.,
                           help="Sets the speed the particles move towards their nearest detection to SPEED.")

        group.add_argument("--ext_states_ms_bandwidth", metavar="[>0.0 - N]", type=ParticleFilterSimulatorConfig.InRange(float, min_ex_val=.0), default=None,
                           help="Sets the mean shift bandwidth used in the RBF kernel for extracting the current states to EXT_STATES_MS_BANDWIDTH. If not given, the bandwidth is estimated.")

        group.add_argument("--gospa_c", metavar="[>0.0 - N]", type=ParticleFilterSimulatorConfig.InRange(float, min_ex_val=.0), default=1.,
                           help="Sets the value c for GOSPA, which calculates an assignment metric between tracks and measurements. "
                                "It serves two purposes: first, it is a distance measure where in case the distance between the two compared points is greater than c, it is classified as outlier "
                                "and second it is incorporated inthe punishing value.")

        group.add_argument("--gospa_p", metavar="[>0 - N]", type=ParticleFilterSimulatorConfig.InRange(int, min_ex_val=0), default=1,
                           help="Sets the value p for GOSPA, which is used to calculate the p-norm of the sum of the GOSPA error terms (distance, false alarms and missed detections).")

        # Simulator group
        group = self._parser.add_argument_group("Simulator - Automates the simulation")

        group.add_argument("--init_kbd_cmds", action=self._EvalListAction, comptype=str,
                           default=[],
                           help=f"Specifies a list of keyboard commands that will be executed only once. These commands will be executed only if and when "
                           f"{str(SimStepPart.USER_INITIAL_KEYBOARD_COMMANDS)} is set with the parameter --sim_loop_step_parts.")

        group.add_argument("--sim_loop_step_parts", metavar=f"[{{{  ','.join([str(t) for t in SimStepPart]) }}}*]", action=self._EvalListAction, comptype=SimStepPart,
                           default=[SimStepPart.USER_INITIAL_KEYBOARD_COMMANDS,
                                    SimStepPart.DRAW, SimStepPart.WAIT_FOR_TRIGGER, SimStepPart.LOAD_NEXT_FRAME,
                                    SimStepPart.USER_PREDICT, SimStepPart.USER_UPDATE, SimStepPart.USER_RESAMPLE, SimStepPart.USER_EXTRACT_STATES, SimStepPart.USER_CALC_GOSPA],
                           help=f"Sets the loops step parts and their order. This determindes how the main loop in the simulation behaves, when the current state is drawn, the user can interact, "
                           f"etc. Be cautious that some elements need to be present to make the program work (see default value)!")

        group.add_argument("--auto_step_interval", metavar="[0 - N]", type=ParticleFilterSimulatorConfig.InRange(int, min_val=0, ), default=0,
                           help="Sets the time interval [ms] for automatic stepping of the filter.")

        group.add_argument("--auto_step_autostart", type=ParticleFilterSimulatorConfig.IsBool, nargs="?", default=False, const=True, choices=[True, False, 1, 0],
                           help="Indicates if the automatic stepping mode will start (if properly set) at the beginning of the simulation. "
                                "If this value is not set the automatic mode is not active, but the manual stepping mode instead.")

        # File Reader group
        group = self._parser.add_argument_group("File Reader - reads detections from file")

        group.add_argument("--input", default="No file.", help="Parse detections with coordinates from INPUT_FILE.")

        # File Storage group
        group = self._parser.add_argument_group("File Storage - stores data as detections and videos to file")

        group.add_argument("--output", default=None,
                           help="Sets the output file to store the (manually set or simulated) data as detections' coordinates to OUTPUT_FILE. The parts of the filename equals ?? gets replaced "
                                "by the continuous number defined by the parameter output_seq_max. Default: Not storing any data.")

        group.add_argument("--output_seq_max", type=int, default=9999,
                           help="Sets the max. number to append at the end of the output filename (see parameter --output). This allows for automatically continuously named files and prevents "
                                "overwriting previously stored results. The format will be x_0000, depending on the filename and the number of digits of OUTPUT_SEQ_MAX.")

        group.add_argument("--output_fill_gaps", type=ParticleFilterSimulatorConfig.IsBool, nargs="?", default=False, const=True, choices=[True, False, 1, 0],
                           help="Indicates if the first empty file name will be used when determining a output filename (see parameters --output and --output_seq_max) or if the next number "
                                "(to create the filename) will be N+1 with N is the highest number in the range and format given by the parameter --output_seq_max.")

        group.add_argument("--output_coord_system_conversion", metavar="OUTPUT_COORD_SYSTEM", action=self._EvalAction, comptype=CoordSysConv,
                           choices=[str(t) for t in CoordSysConv], default=CoordSysConv.ENU,
                           help="Defines the coordinates-conversion of the internal system (ENU) to the OUTPUT_COORD_SYSTEM for storing the values.")

        group.add_argument("--output_video", metavar="OUTPUT_FILE", default=None,
                           help="Sets the output filename to store the video captures from the single frames of the plotting window. The parts of the filename equals ?? gets replaced by the "
                                "continuous number defined by the parameter output_seq_max. Default: Not storing any video.")

        # Visualization group
        group = self._parser.add_argument_group('Visualization - options for visualizing the simulated and filtered results')

        group.add_argument("--gui", type=ParticleFilterSimulatorConfig.IsBool, nargs="?", default=True, const=False, choices=[True, False, 1, 0],
                           help="Specifies, if the GUI should be shown und be user or just run the program. Note: if the GUI is not active, there's no interaction with possible "
                                "and therefore anythin need to be done by command line parameters (esp. see --auto_step_autostart) or keyboard commands.")

        group.add_argument("--density_draw_style", action=self._EvalAction, comptype=DensityDrawStyle, choices=[str(t) for t in DensityDrawStyle],
                           default=DensityDrawStyle.NONE,
                           help=f"Sets the drawing style to visualizing the density/intensity map. Possible values are: {str(DensityDrawStyle.KDE)} (kernel density estimator) and "
                                f"{str(DensityDrawStyle.EVAL)} (evaluate the correct value for each cell in a grid).")

        group.add_argument("--n_bins_density_map", metavar="[100-N]", type=ParticleFilterSimulatorConfig.InRange(int, min_val=100), default=100,
                           help="Sets the number bins for drawing the PHD density map. A good number might be 100.")

        group.add_argument("--draw_layers", metavar=f"[{{{  ','.join([str(t) for t in DrawLayer]) }}}*]", action=self._EvalListAction, comptype=DrawLayer,
                           default=[ly for ly in DrawLayer if ly not in [DrawLayer.ALL_TRAJ_LINE, DrawLayer.ALL_TRAJ_POS, DrawLayer.ALL_DET, DrawLayer.ALL_DET_CONN]],
                           help=f"Sets the list of drawing layers. Allows to draw only the required layers and in the desired order. If not set, a fixes set of layers are drawn in a fixed order. "
                           f"Example 1: [{str(DrawLayer.DENSITY_MAP)}, {str(DrawLayer.PARTICLES)}]\n"
                           f"Example 2: [layer for layer in DrawLayer if not layer == {str(DrawLayer.ALL_DET)} and not layer == {str(DrawLayer.ALL_DET_CONN)}]")

        group.add_argument("--show_legend", action=self._IntOrWhiteSpaceStringAction, nargs="+", default="lower right",
                           help="If set, the legend will be shown. SHOW_LEGEND itself specifies the legend's location. The location can be specified with a number of the corresponding string from "
                                "the following possibilities: 0 = 'best', 1 = 'upper right', 2 = 'upper left', 3 = 'lower left', 4 = 'lower right', 5 = 'right', 6 = 'center left', "
                                "7 = 'center right', 8 = 'lower center', 9 = 'upper center', 10 = 'center'. Default: 4.")

        group.add_argument("--show_colorbar", type=ParticleFilterSimulatorConfig.IsBool, nargs="?", default=True, const=False, choices=[True, False, 1, 0],
                           help="Specifies, if the colorbar should be shown.")

        group.add_argument("--start_window_max", type=ParticleFilterSimulatorConfig.IsBool, nargs="?", default=False, const=True, choices=[True, False, 1, 0],
                           help="Specifies, if the plotting window will be maximized at program start. Works only if the parameter --output_video is not set.")
    # end def __init__

    @staticmethod
    def _doeval(s: str):
        return eval(s)
    # end def
# end class


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
        scenario_data = Wgs84ToEnuConverter.convert(scenario_data, args.observer)
    # end if

    sim: ParticleFilterSimulator = ParticleFilterSimulator(scenario_data=scenario_data, output_coord_system_conversion=args.output_coord_system_conversion, fn_out=args.output,
                                                           fn_out_video=args.output_video,
                                                           auto_step_interval=args.auto_step_interval, auto_step_autostart=args.auto_step_autostart, fov=args.fov,
                                                           limits_mode=args.limits_mode, observer=args.observer, logging=args.verbosity,
                                                           ext_states_ms_bandwidth=args.ext_states_ms_bandwidth, gospa_c=args.gospa_c, gospa_p=args.gospa_p,
                                                           n_particles=args.n_particles, sigma_gauss_kernel=args.sigma_gauss_kernel, particle_movement_noise=args.particle_movement_noise,
                                                           speed=args.speed,
                                                           gui=args.gui, density_draw_style=args.density_draw_style,
                                                           n_bins_density_map=args.n_bins_density_map,
                                                           draw_layers=args.draw_layers, sim_loop_step_parts=args.sim_loop_step_parts, show_legend=args.show_legend, show_colorbar=args.show_colorbar,
                                                           start_window_max=args.start_window_max, init_kbd_cmds=args.init_kbd_cmds)

    sim.fn_out_seq_max = args.output_seq_max
    sim.fn_out_fill_gaps = args.output_fill_gaps

    sim.run()
# end def main


if __name__ == "__main__":
    main(sys.argv)
