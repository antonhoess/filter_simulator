from __future__ import annotations
from typing import List, Tuple, Optional, Union
from abc import abstractmethod
import numpy as np
import matplotlib.gridspec
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.legend_handler import HandlerPatch
from enum import Enum, IntEnum
# import os
# os.environ['COLUMNS'] = "120"

from filter_simulator.common import Logging, Limits, Position
from filter_simulator.filter_simulator import FilterSimulator
from filter_simulator.gospa import Gospa, GospaResult
from scenario_data.scenario_data_converter import CoordSysConv
from filter_simulator.window_helper import LimitsMode
from scenario_data.scenario_data import ScenarioData
import argparse
from distutils.util import strtobool


class DrawLayerBase(IntEnum):
    pass
# end class


class SimStepPartBase(Enum):
    pass
# end class


class DensityDrawStyleBase(Enum):
    pass
# end class


class DataProviderTypeBase(Enum):
    pass
# end class


class AdditionalAxisBase(Enum):
    pass
# end class


class BaseFilterSimulatorConfig:
    def __init__(self):
        self._parser_groups = dict()
        self._parser = argparse.ArgumentParser(add_help=False, formatter_class=self._ArgumentDefaultsRawDescriptionHelpFormatter)

        # General group
        group = self._parser.add_argument_group("General - common program settings")
        self._parser_groups["general"] = group

        group.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS,
                           help="Shows this help and exits the program.")

        group.add_argument("--fov", metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"), action=self._LimitsAction, type=float, nargs=4, default=Limits(-10, -10, 10, 10),
                           help="Sets the Field of View (FoV) of the scene.")

        group.add_argument("--limits_mode", action=self._EvalAction, comptype=LimitsMode, user_eval=self._user_eval, choices=[str(t) for t in LimitsMode], default=LimitsMode.FOV_INIT_ONLY,
                           help="Sets the limits mode, which defines how the limits for the plotting window are set initially and while updating the plot.")

        group.add_argument("--verbosity", action=self._EvalAction, comptype=Logging, user_eval=self._user_eval, choices=[str(t) for t in Logging], default=Logging.INFO,
                           help="Sets the programs verbosity level to VERBOSITY. If set to Logging.NONE, the program will be silent.")

        group.add_argument("--observer_position", dest="observer", metavar=("LAT", "LON"), action=self._PositionAction, type=float, nargs=2, default=None,
                           help="Sets the geodetic position of the observer in WGS84 to OBSERVER_POSITION. Can be used instead of the automatically used center of all detections or in case of only "
                                "manually creating detections, which needed to be transformed back to WGS84.")

        # Filter group - subclasses will rename it
        group = self._parser.add_argument_group()
        self._parser_groups["filter"] = group

        group.add_argument("--gospa_c", metavar="[>0.0 - N]", type=BaseFilterSimulatorConfig.InRange(float, min_ex_val=.0), default=1.,
                           help="Sets the value c for GOSPA, which calculates an assignment metric between tracks and measurements. "
                                "It serves two purposes: first, it is a distance measure where in case the distance between the two compared points is greater than c, it is classified as outlier "
                                "and second it is incorporated inthe punishing value.")

        group.add_argument("--gospa_p", metavar="[>0 - N]", type=BaseFilterSimulatorConfig.InRange(int, min_ex_val=0), default=1,
                           help="Sets the value p for GOSPA, which is used to calculate the p-norm of the sum of the GOSPA error terms (distance, false alarms and missed detections).")

        # Simulator group
        group = self._parser.add_argument_group("Simulator - Automates the simulation")
        self._parser_groups["simulator"] = group

        group.add_argument("--init_kbd_cmds", action=self._EvalListAction, comptype=str, user_eval=self._user_eval, default=[],
                           help=f"Specifies a list of keyboard commands that will be executed only once. These commands will be executed only if and when "
                           f"{str(self.get_sim_step_part().USER_INITIAL_KEYBOARD_COMMANDS)} is set with the parameter --sim_loop_step_parts.")

        group.add_argument("--sim_loop_step_parts", metavar=f"[{{{  ','.join([str(t) for t in self.get_sim_step_part()]) }}}*]", action=self._EvalListAction, comptype=self.get_sim_step_part(),
                           user_eval=self._user_eval, default=self.get_sim_loop_step_parts_default(),
                           help=f"Sets the loops step parts and their order. This determindes how the main loop in the simulation behaves, when the current state is drawn, the user can interact, "
                           f"etc. Be cautious that some elements need to be present to make the program work (see default value)!")

        group.add_argument("--auto_step_interval", metavar="[0 - N]", type=BaseFilterSimulatorConfig.InRange(int, min_val=0), default=0,
                           help="Sets the time interval [ms] for automatic stepping of the filter.")

        group.add_argument("--auto_step_autostart", type=BaseFilterSimulatorConfig.IsBool, nargs="?", default=False, const=True, choices=[True, False, 1, 0],
                           help="Indicates if the automatic stepping mode will start (if properly set) at the beginning of the simulation. "
                                "If this value is not set the automatic mode is not active, but the manual stepping mode instead.")

        # File Reader group
        group = self._parser.add_argument_group("File Reader - reads detections from file")
        self._parser_groups["file_reader"] = group

        group.add_argument("--input", default="No file.", help="Parse detections with coordinates from INPUT_FILE.")

        # File Storage group
        group = self._parser.add_argument_group("File Storage - stores data liks detections and videos to file")
        self._parser_groups["file_storage"] = group

        group.add_argument("--output", default=None,
                           help="Sets the output file to store the (manually set or simulated) data as detections' coordinates to OUTPUT_FILE. The parts of the filename equals ?? gets replaced "
                                "by the continuous number defined by the parameter output_seq_max. Default: Not storing any data.")

        group.add_argument("--output_seq_max", type=int, default=9999,
                           help="Sets the max. number to append at the end of the output filename (see parameter --output). This allows for automatically continuously named files and prevents "
                                "overwriting previously stored results. The format will be x_0000, depending on the filename and the number of digits of OUTPUT_SEQ_MAX.")

        group.add_argument("--output_fill_gaps", type=BaseFilterSimulatorConfig.IsBool, nargs="?", default=False, const=True, choices=[True, False, 1, 0],
                           help="Indicates if the first empty file name will be used when determining a output filename (see parameters --output and --output_seq_max) or if the next number "
                                "(to create the filename) will be N+1 with N is the highest number in the range and format given by the parameter --output_seq_max.")

        group.add_argument("--output_coord_system_conversion", metavar="OUTPUT_COORD_SYSTEM", action=self._EvalAction, comptype=CoordSysConv, user_eval=self._user_eval,
                           choices=[str(t) for t in CoordSysConv], default=CoordSysConv.ENU,
                           help="Defines the coordinates-conversion of the internal system (ENU) to the OUTPUT_COORD_SYSTEM for storing the values.")

        group.add_argument("--output_video", metavar="OUTPUT_FILE", default=None,
                           help="Sets the output filename to store the video captures from the single frames of the plotting window. The parts of the filename equals ?? gets replaced by the "
                                "continuous number defined by the parameter output_seq_max. Default: Not storing any video.")

        # Visualization group
        group = self._parser.add_argument_group("Visualization - options for visualizing the simulated and filtered results")
        self._parser_groups["visualization"] = group

        group.add_argument("--gui", type=BaseFilterSimulatorConfig.IsBool, nargs="?", default=True, const=False, choices=[True, False, 1, 0],
                           help="Specifies, if the GUI should be shown und be user or just run the program. Note: if the GUI is not active, there's no interaction with possible "
                                "and therefore anythin need to be done by command line parameters (esp. see --auto_step_autostart) or keyboard commands.")

        group.add_argument("--n_bins_density_map", metavar="[100 - N]", type=BaseFilterSimulatorConfig.InRange(int, min_val=100), default=100,
                           help="Sets the number bins for drawing the density map. A good number might be 100.")

        group.add_argument("--show_legend", action=self._IntOrWhiteSpaceStringAction, nargs="+", default="lower right",
                           help="If set, the legend will be shown. SHOW_LEGEND itself specifies the legend's location. The location can be specified with a number of the corresponding string from "
                                "the following possibilities: 0 = 'best', 1 = 'upper right', 2 = 'upper left', 3 = 'lower left', 4 = 'lower right', 5 = 'right', 6 = 'center left', "
                                "7 = 'center right', 8 = 'lower center', 9 = 'upper center', 10 = 'center'. Default: 4.")

        group.add_argument("--show_colorbar", type=BaseFilterSimulatorConfig.IsBool, nargs="?", default=True, const=False, choices=[True, False, 1, 0],
                           help="Specifies, if the colorbar should be shown.")

        group.add_argument("--start_window_max", type=BaseFilterSimulatorConfig.IsBool, nargs="?", default=False, const=True, choices=[True, False, 1, 0],
                           help="Specifies, if the plotting window will be maximized at program start. Works only if the parameter --output_video is not set.")
    # end def

    @staticmethod
    @abstractmethod
    def get_sim_step_part():
        pass
    # end def

    @staticmethod
    @abstractmethod
    def get_sim_loop_step_parts_default() -> List[SimStepPartBase]:
        pass
    # end def

    @staticmethod
    @abstractmethod
    def _user_eval(s: str):  # This needs to be overridden, since eval() needs the indidivual globals and locals, which are not available here.
        pass
    # end def

    def help(self) -> str:
        return self._parser.format_help()
    # end def

    def read(self, argv: List[str]):
        args, unknown_args = self._parser.parse_known_args(argv)

        if len(unknown_args) > 0:
            print("Unknown argument(s) found:")
            for arg in unknown_args:
                print(arg)
            # end for
        # end if

        return args
    # end def

    class _ArgumentDefaultsRawDescriptionHelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        def _split_lines(self, text, width):
            return super()._split_lines(text, width) + ['']  # Add empty line between the entries
    # end class

    class _EvalAction(argparse.Action):
        def __init__(self, option_strings, dest, nargs=None, const=None, default=None, type=None, choices=None, required=False, help=None, metavar=None, comptype=None, user_eval=None):
            argparse.Action.__init__(self, option_strings, dest, nargs, const, default, type, choices, required, help, metavar)
            self.comptype = comptype
            self.user_eval = user_eval
        # end def

        def __call__(self, parser, namespace, values, option_string=None, comptype=None):
            x = self.silent_eval(values)

            if isinstance(x, self.comptype):
                setattr(namespace, self.dest, x)
            else:
                raise TypeError(f"'{str(x)}' is not of type '{self.comptype}', but of '{type(x)}'.")
            # end if
        # end def

        def silent_eval(self, expr: str):
            res = None

            try:
                if self.user_eval:
                    res = self.user_eval(expr)
                else:
                    res = eval(expr)
                # end if
            except SyntaxError:
                pass
            except AttributeError:
                pass
            # end try

            return res
        # end def
    # end class

    class _EvalListAction(_EvalAction):
        def __call__(self, parser, namespace, values, option_string=None, comptype=None, user_eval=None):
            x = self.silent_eval(values)

            if isinstance(x, list) and all(isinstance(item, self.comptype) for item in x):
                setattr(namespace, self.dest, x)
            # end if
        # end def
    # end class

    class _EvalListToTypeAction(_EvalAction):
        def __init__(self, option_strings, dest, nargs=None, const=None, default=None, type=None, choices=None, required=False, help=None, metavar=None, comptype=None, restype=None, user_eval=None):
            BaseFilterSimulatorConfig._EvalAction.__init__(self, option_strings, dest, nargs, const, default, type, choices, required, help, metavar, comptype, user_eval)
            self.restype = restype
        # end def

        def __call__(self, parser, namespace, values, option_string=None, comptype=None, restype=None):
            x = self.silent_eval(values)

            if isinstance(x, list) and all(isinstance(item, self.comptype) for item in x):
                setattr(namespace, self.dest, self.restype(x))
            # end if
        # end def
    # end class

    class _LimitsAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            x = Limits(*[float(x) for x in values])
            setattr(namespace, self.dest, x)
        # end def
    # end class

    class _PositionAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            x = Position(*[float(x) for x in values])
            setattr(namespace, self.dest, x)
        # end def
    # end class

    class _IntOrWhiteSpaceStringAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            x = ' '.join(values)
            setattr(namespace, self.dest, self.int_or_str(x))
        # end def

        @staticmethod
        def int_or_str(arg: str):
            try:
                res = int(arg)
            except ValueError:
                res = arg
            # end if

            return res
        # end def
    # end class

    class InRange:
        def __init__(self, dtype, min_val=None, max_val=None, min_ex_val=None, max_ex_val=None, excl=False):
            self.__dtype = dtype
            self.__min_val = min_val
            self.__max_val = max_val
            self.__min_ex_val = min_ex_val
            self.__max_ex_val = max_ex_val
            self.__excl = excl

            # The min-ex and max-ax values outplay the non-ex-values which just will be ignored
            if min_ex_val is not None:
                self.__min_val = None
            # end if

            if max_ex_val is not None:
                self.__max_val = None
            # end if
        # end def

        def __str__(self):
            elements = list()
            elements.append(f"dtype={type(self.__dtype()).__name__}")
            elements.append(f"min_val={self.__min_val}" if self.__min_val is not None else None)
            elements.append(f"min_ex_val={self.__min_ex_val}" if self.__min_ex_val is not None else None)
            elements.append(f"max_val={self.__max_val}" if self.__max_val is not None else None)
            elements.append(f"max_ex_val={self.__max_ex_val}" if self.__max_ex_val is not None else None)
            elements.append(f"excl={self.__excl}" if self.__excl else None)

            elements = [e for e in elements if e is not None]

            return f"InRange({', '.join(elements)})"
        # end def

        def __repr__(self):
            return str(self)
        # end def

        def __call__(self, x):
            x = self.__dtype(x)

            err = False

            if self.__min_val is not None and x < self.__min_val or \
                    self.__min_ex_val is not None and x <= self.__min_ex_val or \
                    self.__max_val is not None and x > self.__max_val or \
                    self.__max_ex_val is not None and x >= self.__max_ex_val:
                err = True
            # end if

            # If the defined range is ment to be exclusive, the previously determinded result gets inverted.
            if self.__excl:
                err = not err
            # end if

            if err:
                raise ValueError
            # end if

            return x
        # end def
    # end class

    class IsBool:
        def __new__(cls, x) -> bool:
            try:
                x = bool(strtobool(x))
                return x
            except ValueError:
                raise ValueError
            # end try
        # end def
    # end class
# end class


class BaseFilterSimulator(FilterSimulator):
    class _HandlerEllipse(HandlerPatch):
        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
            center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
            p = Ellipse(xy=center, width=width + xdescent, height=height + ydescent)
            self.update_prop(p, orig_handle, legend)
            p.set_transform(trans)

            return [p]
        # end
    # end

    def __init__(self, scenario_data: ScenarioData, output_coord_system_conversion: CoordSysConv,
                 fn_out: str, fn_out_video: Optional[str], auto_step_interval: int, auto_step_autostart: bool, fov: Limits, limits_mode: LimitsMode, observer: Position, logging: Logging,
                 gospa_c: float, gospa_p: int,
                 gui: bool, density_draw_style: DensityDrawStyleBase, n_bins_density_map: int,
                 draw_layers: Optional[List[DrawLayerBase]], sim_loop_step_parts: List[SimStepPartBase], show_legend: Optional[Union[int, str]], show_colorbar: bool, start_window_max: bool,
                 init_kbd_cmds: List[str]):

        self.__sim_loop_step_parts: List[SimStepPartBase] = sim_loop_step_parts  # Needs to be set before calling the contructor of the FilterSimulator, since it already needs this values there

        FilterSimulator.__init__(self, scenario_data, output_coord_system_conversion, fn_out, fn_out_video, auto_step_interval, auto_step_autostart,
                                 fov, limits_mode, observer, show_colorbar, start_window_max, gui, logging)

        self.f = None

        self._scenario_data: ScenarioData = scenario_data
        self._gospa_c: float = gospa_c
        self._gospa_p: int = gospa_p
        self._density_draw_style: DensityDrawStyleBase = density_draw_style
        self._n_bins_density_map: int = n_bins_density_map
        self._logging: Logging = logging
        self._draw_layers: Optional[List[DrawLayerBase]] = draw_layers
        self._show_legend: Optional[Union[int, str]] = show_legend
        self._fov: Limits = fov
        self._init_kbd_cmds = init_kbd_cmds
        # --
        self._active_draw_layers: List[bool] = [True for _ in self._draw_layers]
        self._ext_states: List[List[np.ndarray]] = list()
        self._gospa_values: List[float] = list()
        self._colorbar_is_added: bool = False
        self._last_step_part: str = "Initialized"
        self._initial_keyboard_commands_executed = False
        self._ax_add = None
        self._ax_add_enum = self.get_additional_axis_enum()
        self._ax_add_type = self._ax_add_enum.NONE
        self._draw_limits = None
        self._draw_cmap = "Blues"
        self._draw_plot = None
    # end def

    @property
    def _sim_loop_step_parts(self):
        return self.__sim_loop_step_parts
    # end def

    def _sim_loop_calc_gospa(self):
        self._last_step_part = "Calculate GOSPA"

        if self._step >= 0 and self._scenario_data.gtts is not None:
            # GOSPA
            gtt_points = list()

            for gtt in self._scenario_data.gtts:
                index = self._step - gtt.begin_step

                if 0 <= index < len(gtt.points):
                    gtt_points.append(gtt.points[index])
                # end if
            # end for

            gospa: GospaResult = Gospa.calc([Position(ext_state[0], ext_state[1]) for ext_state in self._ext_states[-1]], gtt_points, c=self._gospa_c, p=self._gospa_p,
                                            assignment_cost_function=lambda x, y: np.linalg.norm(np.asarray([x.x, x.y]) - np.asarray([y.x, y.y])))

            self._gospa_values.append(gospa.gospa)
        # end if
    # end def

    def _sim_loop_initial_keyboard_commands(self):
        if not self._initial_keyboard_commands_executed:
            self._initial_keyboard_commands_executed = True

            for cmd in self._init_kbd_cmds:
                self._cb_keyboard(cmd)
        # end if
    # end def

    @staticmethod
    def _remove_duplikate_states(states: List[np.ndarray]):
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

    @abstractmethod
    def _calc_density(self, x: np.ndarray, y: np.ndarray) -> float:
        pass
    # end def

    def _calc_density_map(self, limits: Limits, grid_res: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_: np.ndarray = np.linspace(limits.x_min, limits.x_max, grid_res)
        y_: np.ndarray = np.linspace(limits.y_min, limits.y_max, grid_res)

        x, y = np.meshgrid(x_, y_)
        z: np.ndarray = np.array(self._calc_density(x, y))

        return x, y, z
    # end def

    def _draw_fov(self, zorder):
        # The rectangle defining the Field of View (FoV)
        width = self._fov.x_max - self._fov.x_min
        height = self._fov.y_max - self._fov.y_min
        ell = Rectangle(xy=(self._fov.x_min, self._fov.y_min), width=width, height=height, fill=False, edgecolor="black",
                        linestyle="--", linewidth=0.5, zorder=zorder, label="fov")
        self._ax.add_patch(ell)
    # end def

    def _draw_all_traj_line(self, zorder):
        if self._scenario_data.gtts is not None:
            # Target trajectories
            for gtt in self._scenario_data.gtts:
                self._ax.plot([point.x for point in gtt.points], [point.y for point in gtt.points], color="blue", linewidth=.5, zorder=zorder, label="traj. ($t_{0..T}$)")
            # end for
        # end if
    # end def

    def _draw_all_traj_pos(self, zorder):
        if self._scenario_data.gtts is not None:
            # Target trajectories
            for gtt in self._scenario_data.gtts:
                self._ax.scatter([point.x for point in gtt.points], [point.y for point in gtt.points],
                                 s=25, color="none", marker="o", edgecolor="blue", linewidth=.5, zorder=zorder, label="traj. pos. ($t_{0..T}$)")
                # Mark dead trajectories as such by changing the color of the last marker
                if gtt.begin_step + len(gtt.points) < self._scenario_data.meta.number_steps:
                    self._ax.scatter([gtt.points[-1].x], [gtt.points[-1].y],
                                     s=25, color="none", marker="o", edgecolor="black", linewidth=.5, zorder=zorder)

            # end for
        # end if
    # end def

    def _draw_until_traj_line(self, zorder):
        if self._scenario_data.gtts is not None:
            # Target trajectories
            for gtt in self._scenario_data.gtts:
                if gtt.begin_step <= self._step:
                    self._ax.plot([point.x for point in gtt.points[:self._step - gtt.begin_step + 1]], [point.y for point in gtt.points[:self._step - gtt.begin_step + 1]],
                                  color="blue", linewidth=.5, zorder=zorder, label="traj. ($t_{0..k}$)")
            # end for
        # end if
    # end def

    def _draw_until_traj_pos(self, zorder):
        if self._scenario_data.gtts is not None:
            # Target trajectories
            for gtt in self._scenario_data.gtts:
                if gtt.begin_step <= self._step:
                    self._ax.scatter([point.x for point in gtt.points[:self._step - gtt.begin_step + 1]], [point.y for point in gtt.points[:self._step - gtt.begin_step + 1]],
                                     s=25, color="none", marker="o", edgecolor="blue", linewidth=.5, zorder=zorder, label="traj. pos. ($t_{0..k}$)")
                    # Mark dead trajectories as such by changing the color of the last marker
                    if gtt.begin_step + len(gtt.points) - 1 <= self._step:
                        self._ax.scatter([gtt.points[-1].x], [gtt.points[-1].y],
                                         s=25, color="none", marker="o", edgecolor="black", linewidth=.5, zorder=zorder)
                    # end if
                # end if
            # end for
        # end if
    # end def

    def _draw_all_det(self, zorder):
        # All detections - each frame's detections in a different color
        for frame in self._frames:
            self._ax.scatter([det.x for det in frame], [det.y for det in frame], s=5, linewidth=.5, edgecolor="green", marker="o", zorder=zorder, label="det. ($t_{0..T}$)")
        # end for
    # end def

    def _draw_all_det_conn(self, zorder):
        # Connections between all detections - only makes sense, if they are manually created or created in a very ordered way, otherwise it's just chaos
        for frame in self._frames:
            self._ax.plot([det.x for det in frame], [det.y for det in frame], color="black", linewidth=.5, linestyle="--", zorder=zorder, label="conn. det. ($t_{0..T}$)")
        # end for
    # end def

    def _draw_until_missed_det(self, zorder):
        if self._scenario_data.mds is not None:
            for frame in self._scenario_data.mds[:self._step + 1]:
                self._ax.scatter([det.x for det in frame], [det.y for det in frame], s=12, linewidth=.5, color="black", marker="x", zorder=zorder, label="missed det. ($t_{0..k}$)")
            # end for
        # end if
    # end def

    def _draw_until_false_alarm(self, zorder):
        if self._scenario_data.fas is not None:
            for frame in self._scenario_data.fas[:self._step + 1]:
                self._ax.scatter([det.x for det in frame], [det.y for det in frame], s=12, linewidth=.5, color="red", edgecolors="darkred", marker="o", zorder=zorder,
                                 label="false alarm ($t_{0..k}$)")
            # end for
        # end if
    # end def

    def _draw_cur_est_state(self, zorder):
        if self.f.cur_frame is not None:
            # Estimated states
            if len(self._ext_states) > 0:
                est_items = self._ext_states[-1]
                self._ax.scatter([est_item[0] for est_item in est_items], [est_item[1] for est_item in est_items], s=100, c="gray", edgecolor="black", alpha=.5, marker="o", zorder=zorder,
                                 label="est. states ($t_k$)")
            # end if
        # end if
    # end def

    def _draw_until_est_state(self, zorder):
        if self.f.cur_frame is not None:
            # Estimated states
            est_items = [est_item for est_items in self._ext_states for est_item in est_items]
            self._ax.scatter([est_item[0] for est_item in est_items], [est_item[1] for est_item in est_items], s=10, c="gold", edgecolor="black", linewidths=.2, marker="o", zorder=zorder,
                             label="est. states ($t_{0..k}$)")
        # end if
    # end def

    def _draw_cur_det(self, zorder):
        if self.f.cur_frame is not None:
            # Current detections
            self._ax.scatter([det.x for det in self.f.cur_frame], [det.y for det in self.f.cur_frame], s=70, c="red", linewidth=.5, marker="x", zorder=zorder, label="det. ($t_k$)")
        # end if
    # end def

    @abstractmethod
    def get_fig_suptitle(self) -> str:
        pass
    # end def

    @abstractmethod
    def get_ax_title(self) -> str:
        pass
    # end def

    @abstractmethod
    def get_density_label(self) -> str:
        pass
    # end def

    @abstractmethod
    def get_draw_routine_by_layer(self, layer: DrawLayerBase):
        pass
    # end def

    def _update_window(self, limits: Limits) -> None:
        if self._step == -1:
            self._draw_limits = self._fov
        else:
            self._draw_limits = limits
        # end if

        self._fig.suptitle(self.get_fig_suptitle())
        self._ax.set_title(self.get_ax_title(), fontsize=8)

        self._draw_plot = None

        for l, ly in enumerate(self._draw_layers):
            if not self._active_draw_layers[l]:
                continue

            self.get_draw_routine_by_layer(ly)(zorder=l)
        # end for

        # Colorbar
        if self._show_colorbar and not self._colorbar_is_added and self._draw_plot:
            cb = self._fig.colorbar(self._draw_plot, ax=self._ax, cax=self._cax)
            cb.set_label(self.get_density_label())
            self._colorbar_is_added = True
        # end if

        # Organize the legend handlers
        if self._show_legend:
            handler_map = dict()
            handler_map[Ellipse] = BaseFilterSimulator._HandlerEllipse()

            handles, labels = self._ax.get_legend_handles_labels()  # Easy way to prevent labels appear multiple times (in case where alements are placed in a for loop)

            if len(labels) > 0:
                by_label = dict(zip(labels, handles))
                legend = self._ax.legend(by_label.values(), by_label.keys(), loc=self._show_legend, fontsize="xx-small", handler_map=handler_map)
                legend.set_zorder(len(self._draw_layers))  # Put the legend on top
            # end if
        # end if

        # self._fig.tight_layout(rect=[.0, .05, 1., .9])

        # Additional subplot
        if self._ax_add is not None:
            if self._ax_add_type is self._ax_add_enum.GOSPA:
                self._ax_add.clear()
                self._ax_add.plot(self._gospa_values)

                self._ax_add.set_title(f"GOSPA", fontsize=8)
                self._ax_add.set_xlabel("Simulation step", fontsize=8)
                self._ax_add.set_ylabel("GOSPA", fontsize=8)

            else:  # Handle all cases not known in this class
                self.do_additional_axis_plot(self._ax_add_type)
            # end if
        # end if
    # end def

    @staticmethod
    @abstractmethod
    def get_draw_layer_enum() -> DrawLayerBase:
        pass
    # end def

    @staticmethod
    @abstractmethod
    def get_additional_axis_enum() -> AdditionalAxisBase:
        pass
    # end def

    @staticmethod
    @abstractmethod
    def get_additional_axis_by_short_name(short_name: str) -> Optional[AdditionalAxisBase]:
        pass
    # end def

    @abstractmethod
    def do_additional_axis_plot(self, axis: AdditionalAxisBase) -> bool:
        pass
    # end def

    @staticmethod
    @abstractmethod
    def get_help() -> str:  # Need to be abstract to enforce the evaluation in the place, where the subclass is known, since in this place here it's unknown
        pass
    # end def

    def _cb_keyboard(self, cmd: str) -> None:
        draw_layer_enum: DrawLayerBase = self.get_draw_layer_enum()

        fields = cmd.split()

        if cmd == "":
            self._next_part_step = True

        elif cmd == "h":
            print(self.get_help())

        elif len(fields) > 0 and fields[0] == "l":
            if len(fields) > 1:
                if fields[1] == "t":  # Toggle
                    for layer in range(len(self._active_draw_layers)):
                        self._active_draw_layers[layer] = not self._active_draw_layers[layer]
                    # end for
                    print("Activity status for all drawing layers toggled.")
                    self._refresh.set()

                elif fields[1] == "s":  # Switch all layers on/off
                    new_state = True

                    # Only if all layers are active, deactivate all, otherwise active all
                    if all(self._active_draw_layers):
                        new_state = False
                    # end if

                    for layer in range(len(self._active_draw_layers)):
                        self._active_draw_layers[layer] = new_state
                    # end for
                    print(f"All drawing layers activity set to {'active' if new_state else 'inactive'}.")
                    self._refresh.set()

                else:
                    try:
                        layer = int(fields[1])
                        if min(draw_layer_enum) <= layer <= max(draw_layer_enum):
                            for l, ly in enumerate(self._draw_layers):
                                if DrawLayerBase(layer) is ly:
                                    self._active_draw_layers[l] = not self._active_draw_layers[l]
                                    print(f"Layer {DrawLayerBase(layer).name} {'de' if not self._active_draw_layers[l] else ''}activated.")
                                    self._refresh.set()
                                # end if
                            # end for
                        else:
                            print(f"Entered layer number ({layer}) not valid. Allowed values range from {min(draw_layer_enum)} to {max(draw_layer_enum)}.")
                        # end if
                    except ValueError:
                        pass
                    # end try
                # end if

            else:
                for l, layer in enumerate(self._draw_layers):
                    print(f"{'+' if self._active_draw_layers[l] else ' '} ({'{:2d}'.format(int(layer))}) {layer.name}")
                # end for
            # end if

        elif len(fields) > 0 and fields[0] == "p":  # plot
            if len(fields) > 1:
                axis = self.get_additional_axis_by_short_name(fields[1])

                if axis is not None:
                    self._toggle_additional_axis(axis)
                else:
                    print(f"No additional axis matching '{fields[1]}' found.")
                # end if
            else:
                if self._ax_add is None:
                    # We can only know what plot to show, if any plot was active before
                    if self._ax_add_type is not self._ax_add_enum.NONE:
                        self._toggle_additional_axis(self._ax_add_type)
                    else:
                        print("Additional plot window not set before - so cannot toggle it (back) on.")
                else:
                    self._toggle_additional_axis(self._ax_add_enum.NONE)
                # end if
            # end if
        else:
            pass
        # end if
    # end def

    def _toggle_additional_axis(self, axis_type: AdditionalAxisBase):
        # Cases
        # -----
        # off -> off
        # off -> on
        # on -> on (same)
        # on -> on (different)
        # on -> off

        toggle = False

        # If we set the same layer as is currently active -> Switch off. Only apply when the window is shown.
        # To check this, we cannot use self._ax_add_type to check this, since it stays on its old value for easier toggling.
        if self._ax_add is not None and axis_type is self._ax_add_type:
            axis_type_effective = self._ax_add_enum.NONE
        else:
            axis_type_effective = axis_type
        # end if

        new_is_none = axis_type_effective is self._ax_add_enum.NONE
        old_is_none = self._ax_add is None  # To check this, we cannot use self._ax_add_type to check this, since it stays on its old value for easier toggling.

        # If we switched: off -> on or on -> off
        if old_is_none is not new_is_none:
            toggle = True
        # end if

        # Here we handle only, if the additional plot should be added or removed - but not the content of the window itself.
        if toggle:
            n = 3
            gs = matplotlib.gridspec.GridSpec(n, 1, hspace=1)

            if axis_type_effective is not self._ax_add_enum.NONE:  # Show additional plot window
                # Reduce size of main plot
                self._ax.set_position(gs[0:n-1].get_position(self._fig))
                self._ax.set_subplotspec(gs[0:n-1])  # Only necessary if using tight_layout()

                # Add additional plot
                self._ax_add = self._fig.add_subplot(gs[n-1])
                self._ax_add.set_subplotspec(gs[n-1])

            else:  # Hide additional plot window
                # Remove additional axis
                self._fig.delaxes(self._ax_add)
                self._ax_add = None

                # Reset size of main plot to its original
                self._ax.change_geometry(1, 1, 1)
            # end if
        # end if

        # Refresh of either toggled on <=> off or changed information to display (whilst staying visible).
        if toggle or axis_type is not self._ax_add_type:
            self._refresh.set()
        # end if

        # Only store the current plotting information, if it is not None, since we need it for later.
        if axis_type is not self._ax_add_enum.NONE:
            self._ax_add_type = axis_type
        # end if
    # end def
# end class
