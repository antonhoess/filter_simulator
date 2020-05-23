#!/usr/bin/env python3

from __future__ import annotations
from typing import Sequence, List, Tuple, Optional, Union
import sys
import random
import numpy as np
from datetime import datetime
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.legend_handler import HandlerPatch
import seaborn as sns
from enum import auto, Enum, IntEnum
import argparse
from distutils.util import strtobool
# import os
# os.environ['COLUMNS'] = "120"

from filter_simulator.common import Logging, Limits, Position
from filter_simulator.io_helper import FileReader, InputLineHandlerLatLonIdx
from filter_simulator.filter_simulator import FilterSimulator, SimStepPartConf
from simulation_data.data_provider_interface import IDataProvider
from simulation_data.data_provider_converter import CoordSysConv, Wgs84ToEnuConverter
from filter_simulator.dyn_matrix import TransitionModel, PcwConstWhiteAccelModelNd
from gm_phd_filter import GmPhdFilter, GmComponent, Gmm, DistMeasure
from simulation_data.data_provider import DataProvider, BirthDistribution
from filter_simulator.window_helper import LimitsMode


class DrawLayer(IntEnum):
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


class PhdFilterSimStepPart(Enum):
    DRAW = 0  # Draw the current scene
    WAIT_FOR_TRIGGER = 1  # Wait for user input to continue with the next step
    LOAD_NEXT_FRAME = 2  # Load the next data (frame)
    USER_PREDICT_AND_UPDATE = 3
    USER_PRUNE = 4
    USER_EXTRACT_STATES = 5
    USER_INITIAL_KEYBOARD_COMMANDS = 6
# end class


class DensityDrawStyle(Enum):
    NONE = 0
    KDE = 1
    EVAL = 2
    HEATMAP = 3
# end class


class DataProviderType(Enum):
    FILE_READER = 0
    SIMULATOR = 1
# end class


class GmPhdFilterSimulator(FilterSimulator, GmPhdFilter):
    def __init__(self, data_provider: IDataProvider, output_coord_system_conversion: CoordSysConv,
                 fn_out: str, fn_out_video: Optional[str], auto_step_interval: int, auto_step_autostart: bool, fov: Limits, birth_area: Limits, limits_mode: LimitsMode, observer: Position, logging: Logging,
                 birth_gmm: List[GmComponent], p_survival: float, p_detection: float,
                 f: np.ndarray, q: np.ndarray, h: np.ndarray, r: np.ndarray, rho_fa: float, gate_thresh: Optional[float],
                 trunc_thresh: float, merge_dist_measure: DistMeasure, merge_thresh: float, max_components: int,
                 ext_states_bias: float, ext_states_use_integral: bool,
                 density_draw_style: DensityDrawStyle, n_samples_density_map: int, n_bins_density_map: int,
                 draw_layers: Optional[List[DrawLayer]], sim_loop_step_parts: List[PhdFilterSimStepPart], show_legend: Optional[Union[int, str]], show_colorbar: bool, start_window_max: bool,
                 init_kbd_cmds: List[str]):

        self.__sim_loop_step_parts: List[PhdFilterSimStepPart] = sim_loop_step_parts  # Needs to be set before calling the contructor of the FilterSimulator, since it already needs this values there

        FilterSimulator.__init__(self, data_provider, output_coord_system_conversion, fn_out, fn_out_video, auto_step_interval, auto_step_autostart, fov, limits_mode, observer, start_window_max, logging)
        GmPhdFilter.__init__(self, birth_gmm=birth_gmm, survival=p_survival, detection=p_detection, f=f, q=q, h=h, r=r, rho_fa=rho_fa, gate_thresh=gate_thresh, logging=logging)

        self.__data_provider: IDataProvider = data_provider
        self.__trunc_thresh: float = trunc_thresh
        self.__merge_dist_measure: DistMeasure = merge_dist_measure
        self.__merge_thresh: float = merge_thresh
        self.__max_components: int = max_components
        self.__ext_states_bias: float = ext_states_bias
        self.__ext_states_use_integral: bool = ext_states_use_integral
        self.__density_draw_style: DensityDrawStyle = density_draw_style
        self.__n_samples_density_map: int = n_samples_density_map
        self.__n_bins_density_map: int = n_bins_density_map
        self.__logging: Logging = logging
        self.__draw_layers: Optional[List[DrawLayer]] = draw_layers
        self.__show_legend: Optional[Union[int, str]] = show_legend
        self.__show_colorbar: bool = show_colorbar
        self.__fov: Limits = fov
        self.__birth_area: Limits = birth_area
        self.__init_kbd_cmds = init_kbd_cmds
        # --
        self.__active_draw_layers: List[bool] = [True for _ in self.__draw_layers]
        self.__ext_states: List[List[np.ndarray]] = []
        self.__colorbar_is_added: bool = False
        self.__last_step_part: str = "Initialized"
        self.__initial_keyboard_commands_executed = False

    def _set_sim_loop_step_part_conf(self):
        # Configure the processing steps
        sim_step_part_conf = SimStepPartConf()

        for step_part in self.__sim_loop_step_parts:
            if step_part is PhdFilterSimStepPart.DRAW:
                sim_step_part_conf.add_draw_step()

            elif step_part is PhdFilterSimStepPart.WAIT_FOR_TRIGGER:
                sim_step_part_conf.add_wait_for_trigger_step()

            elif step_part is PhdFilterSimStepPart.LOAD_NEXT_FRAME:
                sim_step_part_conf.add_load_next_frame_step()

            elif step_part is PhdFilterSimStepPart.USER_PREDICT_AND_UPDATE:
                sim_step_part_conf.add_user_step(self.__sim_loop_predict_and_update)

            elif step_part is PhdFilterSimStepPart.USER_PRUNE:
                sim_step_part_conf.add_user_step(self.__sim_loop_prune)

            elif step_part is PhdFilterSimStepPart.USER_EXTRACT_STATES:
                sim_step_part_conf.add_user_step(self.__sim_loop_extract_states)

            elif step_part is PhdFilterSimStepPart.USER_INITIAL_KEYBOARD_COMMANDS:
                sim_step_part_conf.add_user_step(self.__sim_loop_initial_keyboard_commands)

            else:
                raise ValueError
            # end if
        # end for

        return sim_step_part_conf
    # end def

    def __sim_loop_predict_and_update(self):
        self.__last_step_part = "Predict + Update"

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
        self.__last_step_part = "Prune"

        if self._step >= 0:
            # Prune
            self._prune(trunc_thresh=self.__trunc_thresh, merge_dist_measure=self.__merge_dist_measure, merge_thresh=self.__merge_thresh, max_components=self.__max_components)
        # end if
    # end def

    def __sim_loop_extract_states(self):
        self.__last_step_part = "Extract States"

        if self._step >= 0:
            # Extract states
            ext_states = self._extract_states(bias=self.__ext_states_bias, use_integral=self.__ext_states_use_integral)
            self.__remove_duplikate_states(ext_states)
            self.__ext_states.append(ext_states)
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
        # Code taken from eval_grid_2d()
        points = np.stack((x, y), axis=-1).reshape(-1, 2)

        vals = self._gmm.eval_list(points, which_dims=(0, 1))

        return np.array(vals).reshape(x.shape)

    def __calc_density_map(self, limits: Limits, grid_res: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_: np.ndarray = np.linspace(limits.x_min, limits.x_max, grid_res)
        y_: np.ndarray = np.linspace(limits.y_min, limits.y_max, grid_res)

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

        self._fig.suptitle("Gm-PHD Filter Simulator")
        self._ax.set_title(f"Sim-Step: {self._step if self._step >= 0 else '-'}, Sim-SubStep: {self.__last_step_part}, # Est. States: {len(self.__ext_states[-1]) if len(self.__ext_states) > 0 else '-'}, # GMM-Components: {len(self._gmm)}")

        # cmap = "Greys"
        # cmap = "plasma"
        cmap = "Blues"

        plot = None

        for l, ly in enumerate(self.__draw_layers):
            if not self.__active_draw_layers[l]:
                continue

            zorder = l

            if ly == DrawLayer.DENSITY_MAP:
                # Draw density map
                if self.__density_draw_style == DensityDrawStyle.KDE:
                    if len(self._gmm) > 0:
                        samples = self._gmm.samples(self.__n_samples_density_map)
                        x = [s[0] for s in samples]
                        y = [s[1] for s in samples]
                        plot = sns.kdeplot(x, y, shade=True, ax=self._ax, shade_lowest=False, cmap=cmap, cbar=(not self.__colorbar_is_added), zorder=zorder)  # Colorbar instead of label
                        self.__colorbar_is_added = True
                    # end if

                elif self.__density_draw_style == DensityDrawStyle.EVAL:
                    x, y, z = self.__calc_density_map(limits, grid_res=self.__n_bins_density_map)
                    plot = self._ax.contourf(x, y, z, 100, cmap=cmap, zorder=zorder)  # Colorbar instead of label

                elif self.__density_draw_style == DensityDrawStyle.HEATMAP:
                    samples = self.__gmm.samples(self.__n_samples_density_map)
                    det_limits = self._det_limits
                    plot = self._ax.hist2d([s[0] for s in samples], [s[1] for s in samples], bins=self.__n_bins_density_map,
                                           range=[[det_limits.x_min, det_limits.x_max], [det_limits.y_min, det_limits.y_max]], density=False, cmap=cmap, zorder=zorder)  # Colorbar instead of label
                    # It may make no sense to change the limits, since the sampling happens anyway until infinity in each direction of the
                    # state space and therefore the bins will be quite empty when zooming in, which results in a poor visualization
                    # plot = self._ax.hist2d([s[0] for s in samples], [s[1] for s in samples], bins=self.__n_bins_density_map,
                    #                        range=[[limits.x_min, limits.x_max], [limits.y_min, limits.y_max]], density=False, cmap=cmap, zorder=zorder)  # Colorbar instead of label
                    plot = plot[3]  # Get the image itself

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

            elif ly == DrawLayer.BIRTH_AREA:
                # The rectangle defining the birth area of the simulator
                width = self.__birth_area.x_max - self.__birth_area.x_min
                height = self.__birth_area.y_max - self.__birth_area.y_min
                ell = Rectangle(xy=(self.__birth_area.x_min, self.__birth_area.y_min), width=width, height=height, fill=False, edgecolor="black",
                                linestyle=":", linewidth=0.5,  zorder=zorder, label="birth area")
                self._ax.add_patch(ell)

            elif ly == DrawLayer.ALL_TRAJ_LINE:
                if self.__data_provider.sim_data.gtts is not None:
                    # Target trajectories
                    for gtt in self.__data_provider.sim_data.gtts:
                        self._ax.plot([point.x for point in gtt.points], [point.y for point in gtt.points], color="blue", linewidth=.5, zorder=zorder, label="traj. ($t_{0..T}$)")
                    # end for
                # end if

            elif ly == DrawLayer.ALL_TRAJ_POS:
                if self.__data_provider.sim_data.gtts is not None:
                    # Target trajectories
                    for gtt in self.__data_provider.sim_data.gtts:
                        self._ax.scatter([point.x for point in gtt.points], [point.y for point in gtt.points],
                                         s=25, color="none", marker="o", edgecolor="blue", linewidth=.5, zorder=zorder, label="traj. pos. ($t_{0..T}$)")
                        # Mark dead trajectories as such by changing the color of the last marker
                        if gtt.begin_step + len(gtt.points) < self.__data_provider.sim_data.meta.number_steps:
                            self._ax.scatter([gtt.points[-1].x], [gtt.points[-1].y],
                                             s=25, color="none", marker="o", edgecolor="black", linewidth=.5, zorder=zorder)

                    # end for
                # end if

            elif ly == DrawLayer.UNTIL_TRAJ_LINE:
                if self.__data_provider.sim_data.gtts is not None and len(self.__data_provider.sim_data.gtts) > self._step:
                    # Target trajectories
                    for gtt in self.__data_provider.sim_data.gtts:
                        if gtt.begin_step <= self._step:
                            self._ax.plot([point.x for point in gtt.points[:self._step - gtt.begin_step + 1]], [point.y for point in gtt.points[:self._step - gtt.begin_step + 1]],
                                          color="blue", linewidth=.5, zorder=zorder, label="traj. ($t_{0..k}$)")
                    # end for
                # end if

            elif ly == DrawLayer.UNTIL_TRAJ_POS:
                if self.__data_provider.sim_data.gtts is not None and len(self.__data_provider.sim_data.gtts) > self._step:
                    # Target trajectories
                    for gtt in self.__data_provider.sim_data.gtts:
                        if gtt.begin_step <= self._step:
                            self._ax.scatter([point.x for point in gtt.points[:self._step - gtt.begin_step + 1]], [point.y for point in gtt.points[:self._step - gtt.begin_step + 1]],
                                             s=25, color="none", marker="o", edgecolor="blue", linewidth=.5, zorder=zorder, label="traj. pos. ($t_{0..k}$)")
                            # Mark dead trajectories as such by changing the color of the last marker
                            if gtt.begin_step + len(gtt.points) - 1 <= self._step:
                                self._ax.scatter([gtt.points[-1].x], [gtt.points[-1].y],
                                                 s=25, color="none", marker="o", edgecolor="black", linewidth=.5, zorder=zorder)
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
                if self.__data_provider.sim_data.mds is not None:
                    for frame in self.__data_provider.sim_data.mds[:self._step + 1]:
                        self._ax.scatter([det.x for det in frame], [det.y for det in frame], s=12, linewidth=.5, color="black", marker="x", zorder=zorder, label="missed det. ($t_{0..k}$)")
                    # end for
                # end if

            elif ly == DrawLayer.UNTIL_FALSE_ALARM:
                if self.__data_provider.sim_data.fas is not None:
                    for frame in self.__data_provider.sim_data.fas[:self._step + 1]:
                        self._ax.scatter([det.x for det in frame], [det.y for det in frame], s=12, linewidth=.5, color="red", edgecolors="darkred", marker="o", zorder=zorder,
                                         label="false alarm ($t_{0..k}$)")
                    # end for
                # end if

            elif ly == DrawLayer.CUR_GMM_COV_ELL:
                if self._cur_frame is not None:
                    # GM-PHD components covariance ellipses
                    for comp in self._gmm:
                        ell = self.__get_cov_ellipse_from_comp(comp, 1., facecolor='none', edgecolor="black", linewidth=.5, zorder=zorder, label="gmm cov. ell.")
                        self._ax.add_patch(ell)
                    # end for
                # end if

            elif ly == DrawLayer.CUR_GMM_COV_MEAN:
                if self._cur_frame is not None:
                    # GM-PHD components means
                    self._ax.scatter([comp.loc[0] for comp in self._gmm], [comp.loc[1] for comp in self._gmm], s=5, edgecolor="blue", marker="o", zorder=zorder, label="gmm comp. mean")
                # end if

            elif ly == DrawLayer.CUR_EST_STATE:
                if self._cur_frame is not None:
                    # Estimated states
                    if len(self.__ext_states) > 0:
                        est_items = self.__ext_states[-1]
                        self._ax.scatter([est_item[0] for est_item in est_items], [est_item[1] for est_item in est_items], s=100, c="gray", edgecolor="black", alpha=.5, marker="o", zorder=zorder,
                                         label="est. states ($t_k$)")
                    # end if
                # end if

            elif ly == DrawLayer.UNTIL_EST_STATE:
                if self._cur_frame is not None:
                    # Estimated states
                    est_items = [est_item for est_items in self.__ext_states for est_item in est_items]
                    self._ax.scatter([est_item[0] for est_item in est_items], [est_item[1] for est_item in est_items], s=10, c="gold", edgecolor="black", linewidths=.2, marker="o", zorder=zorder,
                                     label="est. states ($t_{0..k}$)")
                # end if

            elif ly == DrawLayer.CUR_DET:
                if self._cur_frame is not None:
                    # Current detections
                    self._ax.scatter([det.x for det in self._cur_frame], [det.y for det in self._cur_frame], s=70, c="red", linewidth=.5, marker="x", zorder=zorder, label="det. ($t_k$)")
                # end if
            # end if
        # end for

        # Colorbar
        if self.__show_colorbar and not self.__colorbar_is_added and plot:
            cb = self._fig.colorbar(plot, ax=self._ax)
            cb.set_label("PHD intensity")
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
        else:
            pass
        # end if
    # end def
# end class GmPhdFilterSimulator


class GmPhdFilterSimulatorConfig:
    class _ArgumentDefaultsRawDescriptionHelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        def _split_lines(self, text, width):
            return super()._split_lines(text, width) + ['']  # Add empty line etween the entries
    # end class

    class _EvalAction(argparse.Action):
        def __init__(self, option_strings, dest, nargs=None, const=None, default=None, type=None, choices=None, required=False, help=None, metavar=None, comptype=None):
            argparse.Action.__init__(self, option_strings, dest, nargs, const, default, type, choices, required, help, metavar)
            self.comptype = comptype
        # end def

        def __call__(self, parser, namespace, values, option_string=None, comptype=None):
            x = self.silent_eval(values)

            if isinstance(x, self.comptype):
                setattr(namespace, self.dest, x)
            # end if
        # end def

        @staticmethod
        def silent_eval(expr: str):
            res = None

            try:
                res = eval(expr)
            except SyntaxError:
                pass
            except AttributeError:
                pass
            # end try

            return res
        # end def
    # end class

    class _EvalListAction(_EvalAction):
        def __call__(self, parser, namespace, values, option_string=None, comptype=None):
            x = self.silent_eval(values)

            if isinstance(x, list) and all(isinstance(item, self.comptype) for item in x):
                setattr(namespace, self.dest, x)
            # end if
        # end def
    # end class

    class _EvalListToTypeAction(_EvalAction):
        def __init__(self, option_strings, dest, nargs=None, const=None, default=None, type=None, choices=None, required=False, help=None, metavar=None, comptype=None, restype=None):
            GmPhdFilterSimulatorConfig._EvalAction.__init__(self, option_strings, dest, nargs, const, default, type, choices, required, help, metavar, comptype)
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
        def __init__(self, dtype, min_val=None, max_val=None, excl=False):
            self.__dtype = dtype
            self.__min_val = min_val
            self.__max_val = max_val
            self.__excl = excl
        # end def

        def __str__(self):
            return f"InRange(dtype={type(self.__dtype()).__name__}, min_val={self.__min_val}, max_val={self.__max_val}, excl={self.__excl})"
        # end def

        def __repr__(self):
            return str(self)
        # end def

        def __call__(self, x):
            x = self.__dtype(x)

            if self.__min_val and self.__max_val:
                if not self.__excl:
                    if not(self.__min_val <= x <= self.__max_val):
                        raise ValueError
                    # end if
                else:
                    if self.__min_val <= x <= self.__max_val:
                        raise ValueError
                    # end if

                # end if
            elif self.__min_val and not self.__max_val:
                if x < self.__min_val:
                    raise ValueError
                # end if
            elif not self.__min_val and self.__max_val:
                if x > self.__max_val:
                    raise ValueError
                # end if
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

    def __init__(self):
        self.__parser = argparse.ArgumentParser(add_help=False, formatter_class=self._ArgumentDefaultsRawDescriptionHelpFormatter, epilog=GmPhdFilterSimulatorConfig.__epilog(),
                                                description="This a simulator for the GM-PHD filter.")

        # General group
        group = self.__parser.add_argument_group('General - common program settings')

        group.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS,
                           help="Shows this help and exits the program.")

        group.add_argument("--data_provider", action=self._EvalAction, comptype=DataProviderType, choices=[str(t) for t in DataProviderType], default=DataProviderType.FILE_READER,
                           help="Sets the data provider type that defines the data source. DataProviderType.FILE_READER reads lines from file defined in --input, "
                                "DataProviderType.SIMULATOR simulates the PHD behaviour defined by the parameters given in section SIMULATOR).")

        group.add_argument("--fov", metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"), action=self._LimitsAction, type=float, nargs=4, default=Limits(-10, -10, 10, 10),
                           help="Sets the Field of View (FoV) of the scene.")

        group.add_argument("--limits_mode", action=self._EvalAction, comptype=LimitsMode, choices=[str(t) for t in LimitsMode], default=LimitsMode.FOV_INIT_ONLY,
                           help="Sets the limits mode, which defines how the limits for the plotting window are set initially and while updating the plot.")

        group.add_argument("--verbosity", action=self._EvalAction, comptype=Logging, choices=[str(t) for t in Logging], default=Logging.INFO,
                           help="Sets the programs verbosity level to VERBOSITY. If set to Logging.NONE, the program will be silent.")

        group.add_argument("--observer_position", dest="observer", metavar=("LAT", "LON"), action=self._PositionAction, type=float, nargs=2, default=None,
                           help="Sets the geodetic position of the observer in WGS84 to OBSERVER_POSITION. Can be used instead of the automatically used center of all detections or in case of only "
                                "manually creating detections, which needed to be transformed back to WGS84.")

        group.add_argument("--init_kbd_cmds", action=self._EvalListAction, comptype=str,
                           default=[],
                           help=f"Specifies a list of keyboard commands that will be executed only once. These commands will be executed only if and when "
                           f"{str(PhdFilterSimStepPart.USER_INITIAL_KEYBOARD_COMMANDS)} is set with the parameter --sim_loop_step_parts.")

        # PHD group
        group = self.__parser.add_argument_group('PHD - parameters for the PHD filter setup')

        group.add_argument("--birth_gmm", action=self._EvalListToTypeAction, comptype=GmComponent, restype=Gmm, default=Gmm([GmComponent(0.1, [0, 0], np.eye(2) * 10. ** 2)]),
                           help="List ([]) of GmComponent which defines the birth-GMM. Format for a single GmComponent: GmComponent(weight, mean, covariance_matrix).")

        group.add_argument("--p_survival", metavar="[>0.0 - 1.0]", type=GmPhdFilterSimulatorConfig.InRange(float, .0, 1.), default=.9,
                           help="Sets the survival probability for the PHD from time step k to k+1.")

        group.add_argument("--n_birth", metavar="[>0.0 - N]", type=GmPhdFilterSimulatorConfig.InRange(int, 1, None), default=1,
                           help="Sets the mean number of newly born objects in each step to N_BIRTH.")

        group.add_argument("--var_birth", metavar="[>0.0 - N]", type=GmPhdFilterSimulatorConfig.InRange(int, 1, None), default=1,
                           help="Sets the variance of newly born objects to VAR_BIRTH.")

        group.add_argument("--p_detection", metavar="[>0.0 - 1.0]", type=GmPhdFilterSimulatorConfig.InRange(float, .0, 1.), default=.9,
                           help="Sets the (sensor's) detection probability for the measurements.")

        group.add_argument("--transition_model", action=self._EvalAction, comptype=TransitionModel, choices=[str(t) for t in TransitionModel], default=TransitionModel.INDIVIDUAL,
                           help=f"Sets the transition model. If set to {str(TransitionModel.INDIVIDUAL)}, the matrices F (see paraemter --mat_f) and Q (see paraemter --mat_q) need to be specified. "
                                f"{str(TransitionModel.PCW_CONST_WHITE_ACC_MODEL_2xND)} stands for Piecewise Constant Acceleration Model.")

        group.add_argument("--delta_t", metavar="[>0.0 - N]", dest="dt", type=GmPhdFilterSimulatorConfig.InRange(float, .0, None), default=1.,
                           help="Sets the time betwen two measurements to DELTA_T. Does not work with the --transition_model parameter set to TransitionModel.INDIVIDUAL.")

        group.add_argument("--mat_f", dest="f", action=self._EvalAction, comptype=np.ndarray, default=np.eye(2),
                           help="Sets the transition model matrix for the PHD.")

        group.add_argument("--mat_q", dest="q", action=self._EvalAction, comptype=np.ndarray, default=np.eye(2) * 0.,
                           help="Sets the process noise covariance matrix.")

        group.add_argument("--mat_h", dest="h", action=self._EvalAction, comptype=np.ndarray, default=np.eye(2),
                           help="Sets the measurement model matrix.")

        group.add_argument("--mat_r", dest="r", action=self._EvalAction, comptype=np.ndarray, default=np.eye(2) * .1,
                           help="Sets the measurement noise covariance matrix.")

        group.add_argument("--sigma_accel_x", type=float, default=.1,
                           help="Sets the variance of the acceleration's x-component to calculate the process noise covariance_matrix Q. Only evaluated when using the "
                                f"{str(TransitionModel.PCW_CONST_WHITE_ACC_MODEL_2xND)} (see parameter --transition_model) and in this case ignores the value specified for Q (see parameter --mat_q).")

        group.add_argument("--sigma_accel_y", type=float, default=.1,
                           help="Sets the variance of the acceleration's y-component to calculate the process noise covariance_matrix Q. Only evaluated when using the "
                                "{str(TransitionModel.PCW_CONST_WHITE_ACC_MODEL_2xND)} (see parameter --transition_model) and in this case ignores the value specified for Q (see parameter --mat_q).")

        group.add_argument("--gate_thresh", metavar="[0.0 - 1.0]", type=GmPhdFilterSimulatorConfig.InRange(float, .0, 1.), default=None,
                           help="Sets the confidence threshold for chi^2 gating on new measurements to GATE_THRESH.")

        group.add_argument("--rho_fa", metavar="[>0.0 - N]", type=GmPhdFilterSimulatorConfig.InRange(float, .0, None), default=None,
                           help="Sets the probability of false alarms per volume unit to RHO_FA. If specified, the mean number of false alarms (see parameter --n_fa) will be recalculated "
                                "based on RHO_FA and the FoV. ")

        group.add_argument("--trunc_thresh", metavar="[>0.0 - N]", type=GmPhdFilterSimulatorConfig.InRange(float, .0, None), default=1e-6,
                           help="Sets the truncation threshold for the prunging step. GM components with weights lower than this value get directly removed.")

        group.add_argument("--merge_dist_measure", action=self._EvalAction, comptype=DistMeasure, choices=[str(t) for t in DistMeasure],
                           default=DistMeasure.MAHALANOBIS_MOD,
                           help="Defines the measurement for calculating the distance between two GMM components.")

        group.add_argument("--merge_thresh", metavar="[>0.0 - N]", type=GmPhdFilterSimulatorConfig.InRange(float, .0, None), default=.01,
                           help="Sets the merge threshold for the prunging step. GM components with a distance distance lower than this value get merged. The distacne measure is given by the "
                                "parameter --merge_dist_measure and depending in this parameter the threashold needs to be set differently.")

        group.add_argument("--max_components", metavar="[1 - N]", type=GmPhdFilterSimulatorConfig.InRange(int, 1, None), default=100,
                           help="Sets the max. number of Gm components used for the GMM representing the current PHD.")

        group.add_argument("--ext_states_bias", metavar="[>0.0 - N]", type=GmPhdFilterSimulatorConfig.InRange(float, .0, None), default=1.,
                           help="Sets the bias for extracting the current states. It works as a factor for the GM component's weights and is used, "
                                "in case the weights are too small to reach a value higher than 0.5, which in needed to get extracted as a state.")

        group.add_argument("--ext_states_use_integral", type=GmPhdFilterSimulatorConfig.IsBool, nargs="?", default=False, const=True, choices=[True, False, 1, 0],
                           help="Specifies if the integral approach for extracting the current states should be used.")

        # Data Simulator group
        group = self.__parser.add_argument_group("Simulator - calculates detections from simulation")

        group.add_argument("--birth_area", metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"), action=self._LimitsAction, type=float, nargs=4, default=None,
                           help="Sets the are for newly born targets. It not set, the same limits as defined by --fov will get used.")

        group.add_argument("--sim_t_max",  metavar="[0 - N]", type=GmPhdFilterSimulatorConfig.InRange(int, 0, None), default=50,
                           help="Sets the number of simulation steps to SIM_T_MAX when using the DataProviderType.SIMULATOR (see parameter --data_provider).")

        group.add_argument("--n_fa", metavar="[0.0 - N]", type=GmPhdFilterSimulatorConfig.InRange(float, 0, None), default=1.,
                           help="Sets the mean number of false alarms in the FoV to N_FA.")

        group.add_argument("--var_fa", metavar="[0.0 - N]", type=GmPhdFilterSimulatorConfig.InRange(float, 0, None),  default=1.,
                           help="Sets the variance of false alarms in the FoV to VAR_FA.")

        group.add_argument("--birth_dist", action=self._EvalAction, comptype=BirthDistribution, choices=[str(t) for t in BirthDistribution], default=BirthDistribution.UNIFORM_AREA,
                           help="Sets the type of probability distribution for new born objects. In case BirthDistribution.UNIFORM_AREA is set, the newly born objects are distributed uniformly "
                                "over the area defined by the parameter --birth_area (or the FoV if not set) and the initial velocity will be set to the values defineed by the perameters "
                                f"--sigma_vel_x and --sigma_vel_y. If {str(BirthDistribution.GMM_FILTER)} is set, the same GMM will get used for the creating of new objects, as the filter uses for "
                                f"their detection. This parameter only takes effect when the parameter --data_provider is set to {str(DataProviderType.SIMULATOR)}.")

        group.add_argument("--sigma_vel_x", metavar="[0.0 - N]", type=GmPhdFilterSimulatorConfig.InRange(float, .0, None), default=.2,
                           help="Sets the variance of the velocitiy's initial x-component of a newly born object to SIGMA_VEL_X. Only takes effect if the parameter --birth_dist is set to "
                                f"{str(BirthDistribution.UNIFORM_AREA)}.")

        group.add_argument("--sigma_vel_y", metavar="[0.0 - N]", type=GmPhdFilterSimulatorConfig.InRange(float, .0, None), default=.2,
                           help="Sets the variance of the velocitiy's initial y-component of a newly born object to SIGMA_VEL_y. Only takes effect if the parameter --birth_dist is set to "
                                f"{str(BirthDistribution.UNIFORM_AREA)}.")

        group.add_argument("--sim_loop_step_parts", metavar=f"[{{{  ','.join([str(t) for t in PhdFilterSimStepPart]) }}}*]", action=self._EvalListAction, comptype=PhdFilterSimStepPart,
                           default=[PhdFilterSimStepPart.USER_INITIAL_KEYBOARD_COMMANDS, PhdFilterSimStepPart.USER_PREDICT_AND_UPDATE, PhdFilterSimStepPart.USER_PRUNE, PhdFilterSimStepPart.USER_EXTRACT_STATES,
                                    PhdFilterSimStepPart.DRAW, PhdFilterSimStepPart.WAIT_FOR_TRIGGER, PhdFilterSimStepPart.LOAD_NEXT_FRAME],
                           help=f"Sets the loops step parts and their order. This determindes how the main loop in the simulation behaves, when the current state is drawn, the user can interact, "
                           f"etc. Be cautious that some elements need to be present to make the program work (see default value)!")

        group.add_argument("--auto_step_interval", metavar="[0 - N]", type=GmPhdFilterSimulatorConfig.InRange(int, 0, None), default=0,
                           help="Sets the time interval [ms] for automatic stepping of the filter.")

        group.add_argument("--auto_step_autostart", type=GmPhdFilterSimulatorConfig.IsBool, nargs="?", default=False, const=True, choices=[True, False, 1, 0],
                           help="Indicates if the automatic stepping mode will start (if properly set) at the beginning of the simulation. "
                                "If this value is not set the automatic mode is not active, but the manual stepping mode instead.")
        # File Reader group
        group = self.__parser.add_argument_group("File Reader - reads detections from file")

        group.add_argument("--input", default="No file.", help="Parse detections with coordinates from INPUT_FILE.")

        group.add_argument("--input_coord_system_conversion", action=self._EvalAction, comptype=CoordSysConv, choices=[str(t) for t in CoordSysConv],
                           default=CoordSysConv.NONE,
                           help="Defines the coordinates-conversion of the provided cordinate system into the internal system (ENU).")

        # File Storage group
        group = self.__parser.add_argument_group("File Storage - stores data as detections and videos to file")

        group.add_argument("--output", default=None,
                           help="Sets the output file to store the (manually set or simulated) detections' coordinates to OUTPUT_FILE. Default: out.lst.")

        group.add_argument("--output_seq_max", type=int, default=9999,
                           help="Sets the max. number to append at the end of the output filename (see parameter --output). This allows for automatically continuously named files and prevents "
                                "overwriting previously stored results. The format will be x_0000, depending on the filename and the number of digits of OUTPUT_SEQ_MAX.")

        group.add_argument("--output_fill_gaps", type=GmPhdFilterSimulatorConfig.IsBool, nargs="?", default=False, const=True, choices=[True, False, 1, 0],
                           help="Indicates if the first empty file name will be used when determining a output filename (see parameters --output and --output_seq_max) or if the next number "
                                "(to create the filename) will be N+1 with N is the highest number in the range and format given by the parameter --output_seq_max.")

        group.add_argument("--output_coord_system_conversion", metavar="OUTPUT_COORD_SYSTEM", action=self._EvalAction, comptype=CoordSysConv,
                           choices=[str(t) for t in CoordSysConv], default=CoordSysConv.NONE,
                           help="Defines the coordinates-conversion of the internal system (ENU) to the OUTPUT_COORD_SYSTEM for storing the values.")

        group.add_argument("--output_video", metavar="OUTPUT_FILE", default=None,
                           help="Sets the output filename to store the video captures from the single frames of the plotting window. Default: Not storing any video.")

        # Visualization group
        group = self.__parser.add_argument_group('Visualization - options for visualizing the simulated and filtered results')

        group.add_argument("--density_draw_style", action=self._EvalAction, comptype=DensityDrawStyle, choices=[str(t) for t in DensityDrawStyle],
                           default=DensityDrawStyle.NONE,
                           help=f"Sets the drawing style to visualizing the density/intensity map. Possible values are: {str(DensityDrawStyle.KDE)} (kernel density estimator), "
                                f"{str(DensityDrawStyle.EVAL)} (evaluate the correct value for each cell in a grid) and {str(DensityDrawStyle.HEATMAP)} (heatmap made of sampled points from the PHD).")

        group.add_argument("--n_samples_density_map", metavar="[1000-N]", type=GmPhdFilterSimulatorConfig.InRange(int, 1000, None),  default=1000,
                           help="Sets the number samples to draw from the PHD for drawing the density map. A good number might be 10000.")

        group.add_argument("--n_bins_density_map", metavar="[100-N]", type=GmPhdFilterSimulatorConfig.InRange(int, 100, None),  default=100,
                           help="Sets the number bins for drawing the PHD density map. A good number might be 100.")

        group.add_argument("--draw_layers", metavar=f"[{{{  ','.join([str(t) for t in DrawLayer]) }}}*]", action=self._EvalListAction, comptype=DrawLayer,
                           default=[ly for ly in DrawLayer if ly not in[DrawLayer.ALL_TRAJ_LINE, DrawLayer.ALL_TRAJ_POS, DrawLayer.ALL_DET, DrawLayer.ALL_DET_CONN,
                                                                        DrawLayer.CUR_GMM_COV_ELL, DrawLayer.CUR_GMM_COV_MEAN]],
                           help=f"Sets the list of drawing layers. Allows to draw only the required layers and in the desired order. If not set, a fixes set of layers are drawn in a fixed order. "
                           f"Example 1: [{str(DrawLayer.DENSITY_MAP)}, {str(DrawLayer.UNTIL_EST_STATE)}]\n"
                           f"Example 2: [layer for layer in DrawLayer if not layer == {str(DrawLayer.CUR_GMM_COV_ELL)} and not layer == {str(DrawLayer.CUR_GMM_COV_MEAN)}]")

        group.add_argument("--show_legend", action=self._IntOrWhiteSpaceStringAction, nargs="+", default="lower right",
                           help="If set, the legend will be shown. SHOW_LEGEND itself specifies the legend's location. The location can be specified with a number of the corresponding string from "
                                "the following possibilities: 0 = 'best', 1 = 'upper right', 2 = 'upper left', 3 = 'lower left', 4 = 'lower right', 5 = 'right', 6 = 'center left', "
                                "7 = 'center right', 8 = 'lower center', 9 = 'upper center', 10 = 'center'. Default: 4.")

        group.add_argument("--show_colorbar", type=GmPhdFilterSimulatorConfig.IsBool, nargs="?", default=True, const=False, choices=[True, False, 1, 0],
                           help="Specifies, if the colorbar should be shown.")

        group.add_argument("--start_window_max", type=GmPhdFilterSimulatorConfig.IsBool, nargs="?", default=False, const=True, choices=[True, False, 1, 0],
                           help="Specifies, if the plotting window will be maximized at program start. Works only if the parameter --output_video is not set.")
    # end def __init__

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
        # Read all measurements from file
        file_reader: FileReader = FileReader(args.input)
        line_handler: InputLineHandlerLatLonIdx = InputLineHandlerLatLonIdx()
        file_reader.read(line_handler)
        data_provider = line_handler

    else:  # data_provider == DataProviderType.SIMULATOR
        data_provider = DataProvider(f=args.f, q=args.q, dt=args.dt, t_max=args.sim_t_max, n_birth=args.n_birth, var_birth=args.var_birth, n_fa=args.n_fa, var_fa=args.var_fa,
                                     fov=args.fov, birth_area=args.birth_area,
                                     p_survival=args.p_survival, p_detection=args.p_detection, birth_dist=args.birth_dist, sigma_vel_x=args.sigma_vel_x, sigma_vel_y=args.sigma_vel_y,
                                     birth_gmm=args.birth_gmm)
    # end if

    # Convert data from certain coordinate systems to ENU, which is used internally
    if args.input_coord_system_conversion == CoordSysConv.WGS84:
        data_provider = Wgs84ToEnuConverter(data_provider.frame_list, args.observer)
    # end if
    sim: GmPhdFilterSimulator = GmPhdFilterSimulator(data_provider=data_provider, output_coord_system_conversion=args.output_coord_system_conversion, fn_out=args.output,
                                                     fn_out_video=args.output_video,
                                                     auto_step_interval=args.auto_step_interval, auto_step_autostart=args.auto_step_autostart, fov=args.fov, birth_area=args.birth_area,
                                                     limits_mode=args.limits_mode, observer=args.observer, logging=args.verbosity,
                                                     birth_gmm=args.birth_gmm, p_survival=args.p_survival, p_detection=args.p_detection,
                                                     f=args.f, q=args.q, h=args.h, r=args.r, rho_fa=args.rho_fa, gate_thresh=args.gate_thresh,
                                                     trunc_thresh=args.trunc_thresh, merge_dist_measure=args.merge_dist_measure, merge_thresh=args.merge_thresh, max_components=args.max_components,
                                                     ext_states_bias=args.ext_states_bias, ext_states_use_integral=args.ext_states_use_integral,
                                                     density_draw_style=args.density_draw_style, n_samples_density_map=args.n_samples_density_map, n_bins_density_map=args.n_bins_density_map,
                                                     draw_layers=args.draw_layers, sim_loop_step_parts=args.sim_loop_step_parts, show_legend=args.show_legend, show_colorbar=args.show_colorbar,
                                                     start_window_max=args.start_window_max, init_kbd_cmds=args.init_kbd_cmds)

    sim.fn_out_seq_max = args.output_seq_max
    sim.fn_out_fill_gaps = args.output_fill_gaps

    sim.run()
# end def main


if __name__ == "__main__":
    main(sys.argv)
