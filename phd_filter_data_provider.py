from __future__ import annotations
from typing import List, Optional

from filter_simulator.common import FrameList, Detection, Limits
from filter_simulator.data_provider_interface import IDataProvider


# ToDos:
# for-Schleifen mit numpy-Vektoren umsetzen, also einzige Operation statt Schleifendurchlauf

import numpy as np
import random
import time
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime


class Pos:
    def __init__(self, pos_x, pos_y):
        self.pos_x = pos_x
        self.pos_y = pos_y
    # end def

    def __str__(self):
        return "Pos: x={:.04f}, y={:.04f}".format(self.pos_x, self.pos_y)

    def __repr__(self):
        return str(self)
# end class


class Obj:
    def __init__(self, state, frame_id):
        self.state = state
        self.frame_id = frame_id
    # end def

    @property
    def pos_x(self):
        return self.state[0]

    @property
    def pos_y(self):
        return self.state[1]

    @property
    def vel_x(self):
        return self.state[2]

    @property
    def vel_y(self):
        return self.state[3]

    def __str__(self):
        return "Obj: pos=[{:.04f}, {:.04f}], vel=[{:.04f}, {:.04f}]".format(self.pos_x, self.pos_y,
                                                                            self.vel_x, self.vel_y)

    def __repr__(self):
        return str(self)
# end class


class Simulator:
    def __init__(self, f: np.ndarray, q: np.ndarray, t: float, t_max: int, n_birth: int, var_birth: int, n_fa: int, var_fa: int, limits: Limits, p_survival: float, p_detection: float, sigma_vel_x: float, sigma_vel_y: float, seed=None, show_visu=True):
        """
        :param fn_data: Name of file where simulation is saved.
        :param cst: Configuration object with already set values.
        :param displ: Display current states.
        :param fn_final_figure: Name of file where the final state of the plot figure is saved.
        :return Tuple:
        data: List containing simulated targets per frame.
        gt: List with  ground truth per targets.
        all_objects: List with  all targets.
        """
        self.f: np.ndarray = f
        self.T = t
        self.t_max = t_max
        self.n_birth = n_birth
        self.var_birth = var_birth

        self.limits: Limits = limits  # We don't need to calulate the FoV in pixels, do we? I think it's only for the discretization of the false alarms, e.g. for a camera

        self.p_s = p_survival  # Probability of survival
        self.p_d = p_detection  # Probability of detection

        self.q = q

        self.sigma_vel_x = sigma_vel_x  # Std deviation of initial velocity, x
        self.sigma_vel_y = sigma_vel_y  # Std deviation of initial velocity, y

        self.n_fa = n_fa  # Mean no. of false alarms / frame
        self.var_fa = var_fa

        # --

        self.__fig = None
        self.__ax = None
        self.__step_interval = .02  # [s]

        self.__is_drawing = False
        self.__tt = 0
        self.__objects = []                                  # Living objects
        self.__all_objects = []                              # All objects (over time)
        self.__data = [[] for _ in range(self.t_max)]  # Data storage (per frame)
        self.__gt = [[] for _ in range(self.t_max)]    # GT storage (per frame)

        self.__show_visu = show_visu

        # Set random generator for reproducible results
        if seed is not None:
            random.seed(seed)  # Not working - different results anyway

    def __update_window(self, _frame):
        if self.__is_drawing:
            self.__ax.set_aspect('equal', 'box')

            # Targets (trajectory)
            for ii in range(len(self.__objects)):
                self.__ax.plot([o.pos_x for o in self.__objects[ii]], [o.pos_y for o in self.__objects[ii]],
                               color="blue", linewidth=.5)
            # end for

            # Targets (position history)
            if len(self.__gt[self.__tt]) > 0:
                self.__ax.scatter([o.pos_x for o in self.__gt[self.__tt]], [o.pos_y for o in self.__gt[self.__tt]],
                                  edgecolor="blue", marker="o", facecolors="none", linewidth=.5)
            # end if

            # Measurements (incl. false measurements)
            self.__ax.scatter([o.pos_x for o in self.__data[self.__tt]], [o.pos_y for o in self.__data[self.__tt]],
                              s=3, marker="o", color="red")

            self.__is_drawing = False
        # end if

    @staticmethod
    def __sample_from_discrete_distribution(num: int, var: int, alpha=None, beta=None):
        if num == 0:
            return 0

        c_is_set = alpha is not None and beta is not None

        if num == var:  # Poisson
            res = np.random.poisson(num)

        elif num < var:  # Negative binomial
            if not c_is_set:
                beta = num / (var - num)
                alpha = beta * num
            # end if

            res = np.random.negative_binomial(alpha, beta / (1 + beta))

        else:  # Binomial
            p = 1 - var / num
            n = num / p
            res = np.random.binomial(n, p)
        # end if

        return res
    # end def

    @staticmethod
    def __sample_random_from_range(min_val: float, max_val: float):
        return min_val + (max_val - min_val) * random.random()
    # end def

    def run(self):
        if self.__show_visu:
            # Processing thread
            t_proc: threading.Thread = threading.Thread(target=self.__processing)
            t_proc.daemon = True
            t_proc.start()

            # Prepare GUI
            self.__fig: plt.Figure = plt.figure()
            self.__fig.canvas.set_window_title("State Space")
            self.__ax = self.__fig.add_subplot(1, 1, 1)

            # Cyclic update check (but only draws, if there's something new)
            _anim: animation.Animation = animation.FuncAnimation(self.__fig, self.__update_window, interval=100)

            # Show blocking window which draws the current state and handles mouse clicks
            plt.show()

        else:
            self.__processing()
        # end if

        return self.__data, self.__gt, self.__all_objects

    def remove_dead_objects(self, ind_deaths):
        if len(ind_deaths) > 0:
            self.__all_objects.append([])

        for ii in ind_deaths:
            self.__all_objects[-1].append(self.__objects[ii])

        for ii in reversed(range(len(ind_deaths))):
            self.__objects.pop(ii)

    # end def

    def __processing(self):
        for tt in range(self.t_max):
            self.__tt = tt

            # Remove dead (not survived) objects
            ####################################
            ind_deaths = [ii for ii in range(len(self.__objects)) if random.random() > self.p_s]
            self.remove_dead_objects(ind_deaths)

            # Prediction of persisting targets
            ##################################
            ind_deaths = []
            for ii, obj in enumerate(self.__objects):
                # Sampling from Q is equal to sampling from G*N(o, sigma^2), which was used before, but we want to use Q as we use it also for the filter itselt
                pred_obj = Obj(np.dot(self.f, obj[-1].state) + np.random.multivariate_normal(mean=np.zeros(self.q.shape[0]), cov=self.q), tt)

                # If target is outside FoV, kill it, otherwise update trajectories with current state
                if not (self.limits.x_min <= pred_obj.pos_x <= self.limits.x_max) or not (self.limits.y_min <= pred_obj.pos_y <= self.limits.x_max):
                    ind_deaths.append(ii)
                else:
                    obj.append(pred_obj)
                # end if
            # end if

            self.remove_dead_objects(ind_deaths)

            # Create birth according to the chosen model
            ############################################
            # Determine number of newly born targets
            n_births = self.__sample_from_discrete_distribution(num=self.n_birth, var=self.var_birth)

            for ii in range(n_births):
                self.__objects.append([])
                self.__objects[-1].append(Obj(np.array([self.__sample_random_from_range(self.limits.x_min, self.limits.x_max),
                                                        self.__sample_random_from_range(self.limits.y_min, self.limits.y_max),
                                                        self.sigma_vel_x * np.random.randn(),
                                                        self.sigma_vel_y * np.random.randn()]),
                                              tt))
            # end for

            # Create measurement data (simulated by p_d and added noise)
            ############################################################
            for ii in range(len(self.__objects)):
                obj = self.__objects[ii][-1]  # Get current objet state (therefore the last element in the objects history)

                self.__gt[tt].append(Pos(obj.pos_x, obj.pos_y))

                if random.random() < self.p_d:  # Target is detected
                    x = np.random.multivariate_normal(mean=np.zeros(self.q.shape[0]), cov=self.q, size=1)
                    x = x[0]  # Get first element
                    self.__data[tt].append(Pos(obj.pos_x + x[0], obj.pos_y + x[1]))
                # end if
            # end if

            # Clutter
            #########
            # Determine number of clutter
            n_clu = self.__sample_from_discrete_distribution(num=self.n_fa, var=self.var_fa, alpha=1, beta=1)  # XXX No idea, what alpha and beta are for

            # Add clutter (as false measurements) to measurements
            for _ in range(n_clu):
                self.__data[tt].append(Pos(self.__sample_random_from_range(self.limits.x_min, self.limits.x_max),
                                           self.__sample_random_from_range(self.limits.y_min, self.limits.y_max)))
            # end for

            # Update visualization
            ######################
            if self.__show_visu:
                # Start drawing
                self.__is_drawing = True
                first_time = datetime.datetime.now()

                # Wait for drawing is finished
                while self.__is_drawing:
                    time.sleep(0.01)

                # Wait for interval time passed
                while True:
                    later_time = datetime.datetime.now()
                    time_diff = later_time - first_time

                    if time_diff.seconds + time_diff.microseconds / 1.e6 > self.__step_interval:
                        break

                    time.sleep(0.01)
                # end while
            # end if

        # end for

        # Add all remaining targets to the list of all targets
        if len(self.__objects) > 0:
            self.__all_objects.append([])

            for obj in self.__objects:
                self.__all_objects[-1].append(obj)
        # end if
    # end def
# end class


class PhdFilterDataProvider(IDataProvider):
    def __init__(self, f: np.ndarray, q: np.ndarray, dt: float, t_max: int, n_birth: int, var_birth: int, n_fa: int, var_fa: int, limits: Limits, p_survival: float, p_detection: float, sigma_vel_x: float, sigma_vel_y: float):
        self.__frame_list: FrameList = FrameList()
        show_visu = False
        #show_visu = True
        meas_data, ground_truth, _all_objects = Simulator(f, q, dt, t_max, n_birth, var_birth, n_fa, var_fa, limits, p_survival, p_detection, sigma_vel_x, sigma_vel_y, seed=None, show_visu=show_visu).run()

        for data in meas_data:
            self.__frame_list.add_empty_frame()

            for data_point in data:
                self.__frame_list.get_current_frame().add_detection(Detection(data_point.pos_x, data_point.pos_y))
            # end for
        # end for
    # end def

    @property
    def frame_list(self) -> FrameList:
        return self.__frame_list
    # end def
# end class
