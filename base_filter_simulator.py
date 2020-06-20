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
from filter_simulator.filter_simulator import FilterSimulator, FilterSimulatorConfig
from filter_simulator.gospa import Gospa, GospaResult
from scenario_data.scenario_data_converter import CoordSysConv
from filter_simulator.window_helper import LimitsMode
from scenario_data.scenario_data import ScenarioData


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


class AdditionalAxis(Enum):
    AX_NONE = 0
    AX_GOSPA = 1
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
                                 fov, limits_mode, observer, start_window_max, gui, logging)

        self.f = None

        self._scenario_data: ScenarioData = scenario_data
        self._gospa_c: float = gospa_c
        self._gospa_p: int = gospa_p
        self._density_draw_style: DensityDrawStyleBase = density_draw_style
        self._n_bins_density_map: int = n_bins_density_map
        self._logging: Logging = logging
        self._draw_layers: Optional[List[DrawLayerBase]] = draw_layers
        self._show_legend: Optional[Union[int, str]] = show_legend
        self._show_colorbar: bool = show_colorbar
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
        self._ax_add_type = AdditionalAxis.AX_NONE
        self._draw_limits = None
        self._draw_cmap = "Blues"
        self._draw_plot = None
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
            cb = self._fig.colorbar(self._draw_plot, ax=self._ax)
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
            if self._ax_add_type is AdditionalAxis.AX_GOSPA:
                self._ax_add.clear()
                self._ax_add.plot(self._gospa_values)

                self._ax_add.set_title(f"GOSPA", fontsize=8)
                self._ax_add.set_xlabel("Simulation step", fontsize=8)
                self._ax_add.set_ylabel("GOSPA", fontsize=8)
                self._ax_add.grid()
            # end if
        # end if
    # end def

    @abstractmethod
    def get_draw_layer_enum(self) -> DrawLayerBase:
        pass
    # end def

    def _cb_keyboard(self, cmd: str) -> None:
        draw_layer_enum: DrawLayerBase = self.get_draw_layer_enum()

        fields = cmd.split()

        if cmd == "":
            self._next_part_step = True

        elif cmd == "h":
            print(FilterSimulatorConfig().help())

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
                if fields[1] == "g":  # Plot GOSPA history
                    self._toggle_additional_axis(AdditionalAxis.AX_GOSPA)
                # end if
            else:
                if self._ax_add is None:
                    # We can only can know what plot to show, if any plot was active before
                    if self._ax_add_type is not AdditionalAxis.AX_NONE:
                        self._toggle_additional_axis(self._ax_add_type)
                    else:
                        print("Additional plot window not set before - so cannot toggle it (back) on.")
                else:
                    self._toggle_additional_axis(AdditionalAxis.AX_NONE)
                # end if
            # end if
        else:
            pass
        # end if
    # end def

    def _toggle_additional_axis(self, axis_type: AdditionalAxis):
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
            axis_type_effective = AdditionalAxis.AX_NONE
        else:
            axis_type_effective = axis_type
        # end if

        new_is_none = axis_type_effective is AdditionalAxis.AX_NONE
        old_is_none = self._ax_add is None  # To check this, we cannot use self._ax_add_type to check this, since it stays on its old value for easier toggling.

        # If we switched: off -> on or on -> off
        if old_is_none is not new_is_none:
            toggle = True
        # end if

        # Here we handle only, if the additional plot should be added or removed - but not the content of the window itself.
        if toggle:
            n = 3
            gs = matplotlib.gridspec.GridSpec(n, 1, hspace=1)

            if axis_type_effective is not AdditionalAxis.AX_NONE:  # Show additional plot window
                # Reduce size of main plot
                self._ax.set_position(gs[0:n-1].get_position(self._fig))
                self._ax.set_subplotspec(gs[0:n-1])  # only necessary if using tight_layout()

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
        if axis_type is not AdditionalAxis.AX_NONE:
            self._ax_add_type = axis_type
        # end if
    # end def
# end class
