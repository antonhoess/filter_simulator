#!/usr/bin/env python3

from __future__ import annotations
from typing import List, Tuple, Optional, Union
import sys
import random
import numpy as np
from datetime import datetime
import matplotlib.gridspec
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.legend_handler import HandlerPatch
import seaborn as sns
from sklearn.cluster import MeanShift
from enum import auto, Enum, IntEnum
import argparse
# import os
# os.environ['COLUMNS'] = "120"

from filter_simulator.common import Logging, Limits, Position
from filter_simulator.filter_simulator import FilterSimulator, SimStepPartConf, FilterSimulatorConfig
from filter_simulator.gospa import Gospa, GospaResult
from filter_simulator.window_helper import LimitsMode
from scenario_data.scenario_data_converter import CoordSysConv, Wgs84ToEnuConverter
from particle_filter import ParticleFilter
from scenario_data.scenario_data import ScenarioData


class DrawLayer(IntEnum):
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


class SimStepPart(Enum):
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


class DensityDrawStyle(Enum):
    NONE = 0
    KDE = 1
    EVAL = 2
# end class


class AdditionalAxis(Enum):
    AX_NONE = 0
    AX_GOSPA = 1
# end class


class ParticleFilterSimulator(FilterSimulator):
    def __init__(self, scenario_data: ScenarioData, output_coord_system_conversion: CoordSysConv,
                 fn_out: str, fn_out_video: Optional[str], auto_step_interval: int, auto_step_autostart: bool, fov: Limits, limits_mode: LimitsMode, observer: Position, logging: Logging,
                 ext_states_ms_bandwidth: float, gospa_c: float, gospa_p: int, n_particles: int, sigma_gauss_kernel: float, particle_movement_noise: float, speed: float,
                 gui: bool, density_draw_style: DensityDrawStyle, n_bins_density_map: int,
                 draw_layers: Optional[List[DrawLayer]], sim_loop_step_parts: List[SimStepPart], show_legend: Optional[Union[int, str]], show_colorbar: bool, start_window_max: bool,
                 init_kbd_cmds: List[str]):

        self.__sim_loop_step_parts: List[SimStepPart] = sim_loop_step_parts  # Needs to be set before calling the contructor of the FilterSimulator, since it already needs this values there

        FilterSimulator.__init__(self, scenario_data, output_coord_system_conversion, fn_out, fn_out_video, auto_step_interval, auto_step_autostart,
                                 fov, limits_mode, observer, start_window_max, gui, logging)

        self.f = ParticleFilter(n_particles, sigma_gauss_kernel, particle_movement_noise, speed, fov=fov, logging=logging)

        self.__scenario_data: ScenarioData = scenario_data
        self.__ms_bandwidth: float = ext_states_ms_bandwidth
        self.__gospa_c: float = gospa_c
        self.__gospa_p: int = gospa_p
        self.__density_draw_style: DensityDrawStyle = density_draw_style
        self.__n_bins_density_map: int = n_bins_density_map
        self.__logging: Logging = logging
        self.__draw_layers: Optional[List[DrawLayer]] = draw_layers
        self.__show_legend: Optional[Union[int, str]] = show_legend
        self.__show_colorbar: bool = show_colorbar
        self.__fov: Limits = fov
        self.__init_kbd_cmds = init_kbd_cmds
        # --
        self.__active_draw_layers: List[bool] = [True for _ in self.__draw_layers]
        self.__ext_states: List[List[np.ndarray]] = list()
        self.__gospa_values: List[float] = list()
        self.__colorbar_is_added: bool = False
        self.__last_step_part: str = "Initialized"
        self.__initial_keyboard_commands_executed = False
        self._ax_add = None
        self._ax_add_type = AdditionalAxis.AX_NONE

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
                sim_step_part_conf.add_user_step(self.__sim_loop_predict)

            elif step_part is SimStepPart.USER_UPDATE:
                sim_step_part_conf.add_user_step(self.__sim_loop_update)

            elif step_part is SimStepPart.USER_RESAMPLE:
                sim_step_part_conf.add_user_step(self.__sim_loop_resample)

            elif step_part is SimStepPart.USER_EXTRACT_STATES:
                sim_step_part_conf.add_user_step(self.__sim_loop_extract_states)

            elif step_part is SimStepPart.USER_CALC_GOSPA:
                sim_step_part_conf.add_user_step(self.__sim_loop_calc_gospa)

            elif step_part is SimStepPart.USER_INITIAL_KEYBOARD_COMMANDS:
                sim_step_part_conf.add_user_step(self.__sim_loop_initial_keyboard_commands)

            else:
                raise ValueError
            # end if
        # end for

        return sim_step_part_conf
    # end def

    def __sim_loop_predict(self):
        # Set current frame
        self.f.cur_frame = self._frames[self._step]

        self.__last_step_part = "Predict"
        self.f.predict()
    # end def

    def __sim_loop_update(self):
        self.__last_step_part = "Update"
        self.f.update()
    # end def

    def __sim_loop_resample(self):
        self.__last_step_part = "Resample"
        self.f.resample()
    # end def

    def __sim_loop_extract_states(self):
        self.__last_step_part = "Extract States"

        # Calculate mean shift
        clustering: bool = True
        if clustering:
            cluster_samples: np.array = np.array([[p.x, p.y] for p in self.f.particles])
            clust: MeanShift = MeanShift(bandwidth=self.__ms_bandwidth).fit(cluster_samples)

            # Extract states
            ext_states = clust.cluster_centers_
            self.__remove_duplikate_states(ext_states)
            self.__ext_states.append(ext_states)
            self.__logging.print_verbose(Logging.DEBUG, clust.labels_)
            self.__logging.print_verbose(Logging.DEBUG, clust.cluster_centers_)
        # end if
    # end def

    def __sim_loop_calc_gospa(self):
        self.__last_step_part = "Calculate GOSPA"

        if self._step >= 0 and self.__scenario_data.gtts is not None:
            # GOSPA
            gtt_points = list()

            for gtt in self.__scenario_data.gtts:
                index = self._step - gtt.begin_step

                if 0 <= index < len(gtt.points):
                    gtt_points.append(gtt.points[index])
                # end if
            # end for

            gospa: GospaResult = Gospa.calc([Position(ext_state[0], ext_state[1]) for ext_state in self.__ext_states[-1]], gtt_points, c=self.__gospa_c, p=self.__gospa_p,
                                            assignment_cost_function=lambda x, y: np.linalg.norm(np.asarray([x.x, x.y]) - np.asarray([y.x, y.y])))

            self.__gospa_values.append(gospa.gospa)
        # end if
    # end def

    def __sim_loop_initial_keyboard_commands(self):
        if not self.__initial_keyboard_commands_executed:
            self.__initial_keyboard_commands_executed = True

            for cmd in self.__init_kbd_cmds:
                self._cb_keyboard(cmd)
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
        accum: float = 0.

        for p in self.f.particles:
            d_x: float = p.x - x
            d_y: float = p.y - y
            p_d: float = 1. / np.sqrt(d_x * d_x + d_y * d_y)
            accum += p_d
        # end for

        return accum

    def __calc_density_map(self, limits: Limits, grid_res: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_: np.ndarray = np.linspace(limits.x_min, limits.x_max, grid_res)
        y_: np.ndarray = np.linspace(limits.y_min, limits.y_max, grid_res)

        x, y = np.meshgrid(x_, y_)
        z: np.ndarray = np.array(self.__calc_density(x, y))

        return x, y, z
    # end def

    def _update_window(self, limits: Limits) -> None:
        class HandlerEllipse(HandlerPatch):
            def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
                center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
                p = Ellipse(xy=center, width=width + xdescent, height=height + ydescent)
                self.update_prop(p, orig_handle, legend)
                p.set_transform(trans)

                return [p]
            # end
        # end

        if self._step == -1:
            limits = self.__fov
        # end if

        self._fig.suptitle("Particle Filter Simulator")
        self._ax.set_title(f"Sim-Step: {self._step if self._step >= 0 else '-'}, Sim-SubStep: {self.__last_step_part}, # Est. States: "
                           f"{len(self.__ext_states[-1]) if len(self.__ext_states) > 0 else '-'}, # GOSPA: "
                           f"{self.__gospa_values[-1] if len(self.__gospa_values) > 0 else '-':.04}", fontsize=8)

        cmap = "Blues"

        plot = None

        for l, ly in enumerate(self.__draw_layers):
            if not self.__active_draw_layers[l]:
                continue

            zorder = l

            if ly == DrawLayer.DENSITY_MAP:
                # Draw density map
                if self.__density_draw_style == DensityDrawStyle.KDE:
                    if len(self.f.particles) > 0:
                        x = [p.x for p in self.f.particles]
                        y = [p.y for p in self.f.particles]
                        plot = sns.kdeplot(x, y, shade=True, ax=self._ax, shade_lowest=False, cmap=cmap, cbar=(not self.__colorbar_is_added), zorder=zorder)  # Colorbar instead of label
                        self.__colorbar_is_added = True
                    # end if

                elif self.__density_draw_style == DensityDrawStyle.EVAL:
                    x, y, z = self.__calc_density_map(limits, grid_res=self.__n_bins_density_map)
                    plot = self._ax.contourf(x, y, z, 100, cmap=cmap, zorder=zorder)  # Colorbar instead of label

                else:  # DensityDrawStyle.NONE:
                    pass
                # end if

            elif ly == DrawLayer.FOV:
                # The rectangle defining the Field of View (FoV)
                width = self.__fov.x_max - self.__fov.x_min
                height = self.__fov.y_max - self.__fov.y_min
                ell = Rectangle(xy=(self.__fov.x_min, self.__fov.y_min), width=width, height=height, fill=False, edgecolor="black",
                                linestyle="--", linewidth=0.5,  zorder=zorder, label="fov")
                self._ax.add_patch(ell)

            elif ly == DrawLayer.ALL_TRAJ_LINE:
                if self.__scenario_data.gtts is not None:
                    # Target trajectories
                    for gtt in self.__scenario_data.gtts:
                        self._ax.plot([point.x for point in gtt.points], [point.y for point in gtt.points], color="blue", linewidth=.5, zorder=zorder, label="traj. ($t_{0..T}$)")
                    # end for
                # end if

            elif ly == DrawLayer.ALL_TRAJ_POS:
                if self.__scenario_data.gtts is not None:
                    # Target trajectories
                    for gtt in self.__scenario_data.gtts:
                        self._ax.scatter([point.x for point in gtt.points], [point.y for point in gtt.points],
                                         s=25, color="none", marker="o", edgecolor="blue", linewidth=.5, zorder=zorder, label="traj. pos. ($t_{0..T}$)")
                        # Mark dead trajectories as such by changing the color of the last marker
                        if gtt.begin_step + len(gtt.points) < self.__scenario_data.meta.number_steps:
                            self._ax.scatter([gtt.points[-1].x], [gtt.points[-1].y],
                                             s=25, color="none", marker="o", edgecolor="black", linewidth=.5, zorder=zorder)

                    # end for
                # end if

            elif ly == DrawLayer.UNTIL_TRAJ_LINE:
                if self.__scenario_data.gtts is not None and len(self.__scenario_data.gtts) > self._step:
                    # Target trajectories
                    for gtt in self.__scenario_data.gtts:
                        if gtt.begin_step <= self._step:
                            self._ax.plot([point.x for point in gtt.points[:self._step - gtt.begin_step + 1]], [point.y for point in gtt.points[:self._step - gtt.begin_step + 1]],
                                          color="blue", linewidth=.5, zorder=zorder, label="traj. ($t_{0..k}$)")
                    # end for
                # end if

            elif ly == DrawLayer.UNTIL_TRAJ_POS:
                if self.__scenario_data.gtts is not None and len(self.__scenario_data.gtts) > self._step:
                    # Target trajectories
                    for gtt in self.__scenario_data.gtts:
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

            elif ly == DrawLayer.ALL_DET:
                # All detections - each frame's detections in a different color
                for frame in self._frames:
                    self._ax.scatter([det.x for det in frame], [det.y for det in frame], s=5, linewidth=.5, edgecolor="green", marker="o", zorder=zorder, label="det. ($t_{0..T}$)")
                # end for

            elif ly == DrawLayer.ALL_DET_CONN:
                # Connections between all detections - only makes sense, if they are manually created or created in a very ordered way, otherwise it's just chaos
                for frame in self._frames:
                    self._ax.plot([det.x for det in frame], [det.y for det in frame], color="black", linewidth=.5, linestyle="--", zorder=zorder, label="conn. det. ($t_{0..T}$)")
                # end for

            elif ly == DrawLayer.UNTIL_MISSED_DET:
                if self.__scenario_data.mds is not None:
                    for frame in self.__scenario_data.mds[:self._step + 1]:
                        self._ax.scatter([det.x for det in frame], [det.y for det in frame], s=12, linewidth=.5, color="black", marker="x", zorder=zorder, label="missed det. ($t_{0..k}$)")
                    # end for
                # end if

            elif ly == DrawLayer.UNTIL_FALSE_ALARM:
                if self.__scenario_data.fas is not None:
                    for frame in self.__scenario_data.fas[:self._step + 1]:
                        self._ax.scatter([det.x for det in frame], [det.y for det in frame], s=12, linewidth=.5, color="red", edgecolors="darkred", marker="o", zorder=zorder,
                                         label="false alarm ($t_{0..k}$)")
                    # end for
                # end if

            elif ly == DrawLayer.PARTICLES:
                # Particles
                self._ax.scatter([p.x for p in self.f.particles], [p.y for p in self.f.particles], s=5, edgecolor="blue", marker="o", zorder=zorder, label="part. ($t_{k}$)")

            elif ly == DrawLayer.IMPORTANCE_WEIGHT_COV:
                if self.f.cur_frame is not None:
                    ell_radius_x: float = self.f.s_gauss
                    ell_radius_y: float = self.f.s_gauss

                    for det in self.f.cur_frame:
                        ellipse: Ellipse = Ellipse((det.x, det.y), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor='none', edgecolor="black", linewidth=.5, zorder=zorder, label="I.W. cov. ($t_{k}$)")
                        self._ax.add_patch(ellipse)
                    # end for
                # end if

            elif ly == DrawLayer.CUR_EST_STATE:
                if self.f.cur_frame is not None:
                    # Estimated states
                    if len(self.__ext_states) > 0:
                        est_items = self.__ext_states[-1]
                        self._ax.scatter([est_item[0] for est_item in est_items], [est_item[1] for est_item in est_items], s=100, c="gray", edgecolor="black", alpha=.5, marker="o", zorder=zorder,
                                         label="est. states ($t_k$)")
                    # end if
                # end if

            elif ly == DrawLayer.UNTIL_EST_STATE:
                if self.f.cur_frame is not None:
                    # Estimated states
                    est_items = [est_item for est_items in self.__ext_states for est_item in est_items]
                    self._ax.scatter([est_item[0] for est_item in est_items], [est_item[1] for est_item in est_items], s=10, c="gold", edgecolor="black", linewidths=.2, marker="o", zorder=zorder,
                                     label="est. states ($t_{0..k}$)")
                # end if

            elif ly == DrawLayer.CUR_DET:
                if self.f.cur_frame is not None:
                    # Current detections
                    self._ax.scatter([det.x for det in self.f.cur_frame], [det.y for det in self.f.cur_frame], s=70, c="red", linewidth=.5, marker="x", zorder=zorder, label="det. ($t_k$)")
                # end if
            # end if
        # end for

        # Colorbar
        if self.__show_colorbar and not self.__colorbar_is_added and plot:
            cb = self._fig.colorbar(plot, ax=self._ax)
            cb.set_label("Intensity")
            self.__colorbar_is_added = True
        # end if

        # Organize the legend handlers
        if self.__show_legend:
            handler_map = dict()
            handler_map[Ellipse] = HandlerEllipse()

            handles, labels = self._ax.get_legend_handles_labels()  # Easy way to prevent labels appear multiple times (in case where alements are placed in a for loop)

            if len(labels) > 0:
                by_label = dict(zip(labels, handles))
                legend = self._ax.legend(by_label.values(), by_label.keys(), loc=self.__show_legend, fontsize="xx-small", handler_map=handler_map)
                legend.set_zorder(len(self.__draw_layers))  # Put the legend on top
            # end if
        # end if

        # self._fig.tight_layout(rect=[.0, .05, 1., .9])

        # Additional subplot
        if self._ax_add is not None:
            if self._ax_add_type is AdditionalAxis.AX_GOSPA:
                self._ax_add.clear()
                self._ax_add.plot(self.__gospa_values)

                self._ax_add.set_title(f"GOSPA", fontsize=8)
                self._ax_add.set_xlabel("Simulation step", fontsize=8)
                self._ax_add.set_ylabel("GOSPA", fontsize=8)
                self._ax_add.grid()
            # end if
        # end if
    # end def

    def _cb_keyboard(self, cmd: str) -> None:
        fields = cmd.split()

        if cmd == "":
            self._next_part_step = True

        elif cmd == "h":
            print(GmPhdFilterSimulatorConfig().help())

        elif len(fields) > 0 and fields[0] == "l":
            if len(fields) > 1:
                if fields[1] == "t":  # Toggle
                    for layer in range(len(self.__active_draw_layers)):
                        self.__active_draw_layers[layer] = not self.__active_draw_layers[layer]
                    # end for
                    print("Activity status for all drawing layers toggled.")
                    self._refresh.set()

                elif fields[1] == "s":  # Switch all layers on/off
                    new_state = True

                    # Only if all layers are active, deactivate all, otherwise active all
                    if all(self.__active_draw_layers):
                        new_state = False
                    # end if

                    for layer in range(len(self.__active_draw_layers)):
                        self.__active_draw_layers[layer] = new_state
                    # end for
                    print(f"All drawing layers activity set to {'active' if new_state else 'inactive'}.")
                    self._refresh.set()

                else:
                    try:
                        layer = int(fields[1])
                        if min(DrawLayer) <= layer <= max(DrawLayer):
                            for l, ly in enumerate(self.__draw_layers):
                                if DrawLayer(layer) is ly:
                                    self.__active_draw_layers[l] = not self.__active_draw_layers[l]
                                    print(f"Layer {DrawLayer(layer).name} {'de' if not self.__active_draw_layers[l] else ''}activated.")
                                    self._refresh.set()
                                # end if
                            # end for
                        else:
                            print(f"Entered layer number ({layer}) not valid. Allowed values range from {min(DrawLayer)} to {max(DrawLayer)}.")
                        # end if
                    except ValueError:
                        pass
                    # end try
                # end if

            else:
                for l, layer in enumerate(self.__draw_layers):
                    print(f"{'+' if self.__active_draw_layers[l] else ' '} ({'{:2d}'.format(int(layer))}) {layer.name}")
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
# end class ParticleFilterSimulator


class GmPhdFilterSimulatorConfig(FilterSimulatorConfig):
    def __init__(self):
        self.__parser = argparse.ArgumentParser(add_help=False, formatter_class=self._ArgumentDefaultsRawDescriptionHelpFormatter, description="This is a simulator for the particle-filter.")

        # General group
        group = self.__parser.add_argument_group('General - common program settings')

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
        group = self.__parser.add_argument_group('PF - parameters for the particle filter setup')

        group.add_argument("--sigma_gauss_kernel", metavar="[>0.0 - N]", type=GmPhdFilterSimulatorConfig.InRange(float, min_ex_val=.0), default=20.,
                           help="Sets sigma of the Gaussian importance weight kernel to SIGMA_GAUSS_KERNEL.")

        group.add_argument("--n_particles", metavar="[100 - N]", type=GmPhdFilterSimulatorConfig.InRange(int, min_val=100), default=100,
                           help="Sets the particle filter's number of particles to N_PARTICLES.")

        group.add_argument("--particle_movement_noise", metavar="[>0.0 - N]", type=GmPhdFilterSimulatorConfig.InRange(float, min_ex_val=.0), default=.1,
                           help="Sets the particle's movement noise to PARTICLE_MOVEMENT_NOISE")

        group.add_argument("--speed", metavar="[0.0 - 1.0]", type=GmPhdFilterSimulatorConfig.InRange(float, min_val=.0, max_val=1.), default=1.,
                           help="Sets the speed the particles move towards their nearest detection to SPEED.")

        group.add_argument("--ext_states_ms_bandwidth", metavar="[>0.0 - N]", type=GmPhdFilterSimulatorConfig.InRange(float, min_ex_val=.0), default=None,
                           help="Sets the mean shift bandwidth used in the RBF kernel for extracting the current states to EXT_STATES_MS_BANDWIDTH. If not given, the bandwidth is estimated.")

        group.add_argument("--gospa_c", metavar="[>0.0 - N]", type=GmPhdFilterSimulatorConfig.InRange(float, min_ex_val=.0), default=1.,
                           help="Sets the value c for GOSPA, which calculates an assignment metric between tracks and measurements. "
                                "It serves two purposes: first, it is a distance measure where in case the distance between the two compared points is greater than c, it is classified as outlier "
                                "and second it is incorporated inthe punishing value.")

        group.add_argument("--gospa_p", metavar="[>0 - N]", type=GmPhdFilterSimulatorConfig.InRange(int, min_ex_val=0), default=1,
                           help="Sets the value p for GOSPA, which is used to calculate the p-norm of the sum of the GOSPA error terms (distance, false alarms and missed detections).")

        # Simulator group
        group = self.__parser.add_argument_group("Simulator - Automates the simulation")

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

        group.add_argument("--auto_step_interval", metavar="[0 - N]", type=GmPhdFilterSimulatorConfig.InRange(int, min_val=0,), default=0,
                           help="Sets the time interval [ms] for automatic stepping of the filter.")

        group.add_argument("--auto_step_autostart", type=GmPhdFilterSimulatorConfig.IsBool, nargs="?", default=False, const=True, choices=[True, False, 1, 0],
                           help="Indicates if the automatic stepping mode will start (if properly set) at the beginning of the simulation. "
                                "If this value is not set the automatic mode is not active, but the manual stepping mode instead.")

        # File Reader group
        group = self.__parser.add_argument_group("File Reader - reads detections from file")

        group.add_argument("--input", default="No file.", help="Parse detections with coordinates from INPUT_FILE.")

        # File Storage group
        group = self.__parser.add_argument_group("File Storage - stores data as detections and videos to file")

        group.add_argument("--output", default=None,
                           help="Sets the output file to store the (manually set or simulated) data as detections' coordinates to OUTPUT_FILE. The parts of the filename equals ?? gets replaced "
                                "by the continuous number defined by the parameter output_seq_max. Default: Not storing any data.")

        group.add_argument("--output_seq_max", type=int, default=9999,
                           help="Sets the max. number to append at the end of the output filename (see parameter --output). This allows for automatically continuously named files and prevents "
                                "overwriting previously stored results. The format will be x_0000, depending on the filename and the number of digits of OUTPUT_SEQ_MAX.")

        group.add_argument("--output_fill_gaps", type=GmPhdFilterSimulatorConfig.IsBool, nargs="?", default=False, const=True, choices=[True, False, 1, 0],
                           help="Indicates if the first empty file name will be used when determining a output filename (see parameters --output and --output_seq_max) or if the next number "
                                "(to create the filename) will be N+1 with N is the highest number in the range and format given by the parameter --output_seq_max.")

        group.add_argument("--output_coord_system_conversion", metavar="OUTPUT_COORD_SYSTEM", action=self._EvalAction, comptype=CoordSysConv,
                           choices=[str(t) for t in CoordSysConv], default=CoordSysConv.ENU,
                           help="Defines the coordinates-conversion of the internal system (ENU) to the OUTPUT_COORD_SYSTEM for storing the values.")

        group.add_argument("--output_video", metavar="OUTPUT_FILE", default=None,
                           help="Sets the output filename to store the video captures from the single frames of the plotting window. The parts of the filename equals ?? gets replaced by the "
                                "continuous number defined by the parameter output_seq_max. Default: Not storing any video.")

        # Visualization group
        group = self.__parser.add_argument_group('Visualization - options for visualizing the simulated and filtered results')

        group.add_argument("--gui", type=GmPhdFilterSimulatorConfig.IsBool, nargs="?", default=True, const=False, choices=[True, False, 1, 0],
                           help="Specifies, if the GUI should be shown und be user or just run the program. Note: if the GUI is not active, there's no interaction with possible "
                                "and therefore anythin need to be done by command line parameters (esp. see --auto_step_autostart) or keyboard commands.")

        group.add_argument("--density_draw_style", action=self._EvalAction, comptype=DensityDrawStyle, choices=[str(t) for t in DensityDrawStyle],
                           default=DensityDrawStyle.NONE,
                           help=f"Sets the drawing style to visualizing the density/intensity map. Possible values are: {str(DensityDrawStyle.KDE)} (kernel density estimator) and "
                                f"{str(DensityDrawStyle.EVAL)} (evaluate the correct value for each cell in a grid).")

        group.add_argument("--n_bins_density_map", metavar="[100-N]", type=GmPhdFilterSimulatorConfig.InRange(int, min_val=100),  default=100,
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

        group.add_argument("--show_colorbar", type=GmPhdFilterSimulatorConfig.IsBool, nargs="?", default=True, const=False, choices=[True, False, 1, 0],
                           help="Specifies, if the colorbar should be shown.")

        group.add_argument("--start_window_max", type=GmPhdFilterSimulatorConfig.IsBool, nargs="?", default=False, const=True, choices=[True, False, 1, 0],
                           help="Specifies, if the plotting window will be maximized at program start. Works only if the parameter --output_video is not set.")
    # end def __init__

    @staticmethod
    def _doeval(s: str):
        return eval(s)
    # end def

    def help(self) -> str:
        return self.__parser.format_help()
    # end def

    def read(self, argv: List[str]):
        args, unknown_args = self.__parser.parse_known_args(argv)

        if len(unknown_args) > 0:
            print("Unknown argument(s) found:")
            for arg in unknown_args:
                print(arg)
            # end for
        # end if

        return args
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
