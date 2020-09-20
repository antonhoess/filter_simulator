# ToDos:
# for-Schleifen mit numpy-Vektoren umsetzen, also einzige Operation statt Schleifendurchlauf
import numpy as np
import random
import time
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime
from enum import Enum

from filter_simulator.common import Limits
from gm_phd_filter import Gmm
from scenario_data.scenario_data import *
from scenario_data.scenario_data_converter import CoordSysConv


class BirthDistribution(Enum):
    UNIFORM_AREA = 0  # Uniformly distributed over the set birth area (which might be or not the FoV)
    GMM_FILTER = 1  # Use the Gaussian Mixture used for the filter (which means, that the filter 'assumes' exactly the same distribution as the simulated data, which is almost too perfect)
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
        return "Obj: pos=[{:.04f}, {:.04f}], vel=[{:.04f}, {:.04f}]".format(self.pos_x, self.pos_y, self.vel_x, self.vel_y)

    def __repr__(self):
        return str(self)
# end class


class ScenarioDataSimulator:
    def __init__(self, f: np.ndarray, q: np.ndarray, dt: float, t_max: int, n_birth: float, var_birth: float, n_fa: float, var_fa: float, fov: Limits, birth_area: Limits,
                 p_survival: float, p_detection: float, birth_dist: BirthDistribution, sigma_vel_x: float, sigma_vel_y: float, birth_gmm: Gmm):
        # Store parameters
        self.__f: np.ndarray = f
        self.__q: np.ndarray = q
        self.__dt: float = dt
        self.__t_max: int = t_max
        self.__n_birth = n_birth
        self.__var_birth: float = var_birth

        self.__fov: Limits = fov
        self.__birth_area: Limits = birth_area

        self.__p_s: float = p_survival
        self.__p_d: float = p_detection

        self.__birth_dist: BirthDistribution = birth_dist

        self.__sigma_vel_x = sigma_vel_x
        self.__sigma_vel_y = sigma_vel_y

        self.__birth_gmm: Gmm = birth_gmm

        self.__n_fa: float = n_fa
        self.__var_fa: float = var_fa

        # Drawing
        self.__fig = None
        self.__ax = None
        self.__is_drawing = False
        self.__show_visu = False  # Indicates if the plotting window is shows during creating the simulation data (to check the created data)
        self.__step_interval = .02  # Interval [s] between showing two subsequent steps in the plotting window.

        # Simulation data
        self.__d = None
    # end def

    def __update_window(self, _frame):
        if self.__is_drawing:
            self.__ax.clear()
            self.__ax.set_aspect('equal', 'box')

            # Target trajectories
            for gtt in self.__d.gtts:
                self.__ax.plot([point.x for point in gtt.points], [point.y for point in gtt.points], markersize=5, marker="o", markerfacecolor="none", markeredgewidth=.5, color="blue", linewidth=.5)
            # end for

            # Detections
            for frame in self.__d.ds:
                self.__ax.scatter([det.x for det in frame], [det.y for det in frame], s=8, marker="o", color="forestgreen", edgecolors="darkgreen", linewidths=.5)

            # False measurements
            for frame in self.__d.fas:
                self.__ax.scatter([det.x for det in frame], [det.y for det in frame], s=8, marker="o", color="red", edgecolors="darkred", linewidths=.5)

            # Missed detections
            for frame in self.__d.mds:
                self.__ax.scatter([det.x for det in frame], [det.y for det in frame], s=12, marker="x", color="black", linewidths=.5)

            self.__is_drawing = False
        # end if

    @staticmethod
    def __panjer(num: float, var: float):
        if num == 0:
            return 0

        if num == var:  # Poisson
            res = np.random.poisson(num)

        elif num < var:  # Negative binomial
            # The values alpha and beta need to be calculated automatically for not only the Poisson and Binomial distribution
            # are smooth at the borders, but also the Negative binomial one (which wourld ne not the case if both values each had a value of e.g. 1)
            beta = num / (var - num)
            alpha = beta * num

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

    def run(self) -> ScenarioData:
        # Simulation data
        self.__d = ScenarioData()
        self.__d.meta = MetaInformation()
        self.__d.meta.version = "1.0"
        self.__d.meta.coordinate_system = CoordSysConv.ENU.value
        self.__d.meta.number_steps = self.__t_max
        self.__d.meta.time_delta = self.__dt
        self.__d.ds = FrameList()
        self.__d.fas = FrameList()
        self.__d.mds = FrameList()
        self.__d.gtts = list()
        self.__d.tds = None  # Will not be set here

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

        return self.__d
    # end def

    def __processing(self):
        for tt in range(self.__t_max):
            # Mark dead (not survived) objects as such
            ##########################################
            for track in [_ for _ in self.__d.gtts if _.survived]:
                if random.random() > self.__p_s:
                    track.survived = False
            # end for

            # Prediction of persisting targets
            ##################################
            for track in [_ for _ in self.__d.gtts if _.survived]:
                # Sampling from Q is equal to sampling from G*N(o, sigma^2), which was used before, but we want to use Q as we use it also for the filter itself
                pred_obj = Obj(np.dot(self.__f, track.states[-1]) + np.random.multivariate_normal(mean=np.zeros(self.__q.shape[0]), cov=self.__q), tt)

                # If target is outside FoV, kill it, otherwise update trajectories with current state
                if not (self.__fov.x_min <= pred_obj.pos_x <= self.__fov.x_max) or not (self.__fov.y_min <= pred_obj.pos_y <= self.__fov.x_max):
                    track.survived = False
                else:
                    track.points.append(Position(pred_obj.pos_x, pred_obj.pos_y))
                    track.states.append(pred_obj.state)
                # end if
            # end if

            # Create birth according to chosen model
            ########################################
            # Determine number of newly born targets
            n_births = self.__panjer(num=self.__n_birth, var=self.__var_birth)
            # XXX Temp. for testing purposes
            # if not hasattr(self, "setxx"):
            #     self.setxx = True
            #     n_births = 1
            # else:
            #     n_births = 0
            # # end if

            for _ in range(n_births):
                if self.__birth_dist is BirthDistribution.UNIFORM_AREA:
                    obj = np.array([self.__sample_random_from_range(self.__birth_area.x_min, self.__birth_area.x_max),
                                    self.__sample_random_from_range(self.__birth_area.y_min, self.__birth_area.y_max),
                                    self.__sigma_vel_x * np.random.randn(),
                                    self.__sigma_vel_y * np.random.randn()])
                else:  # self.__birth_dist is BirthDistribution.GMM_FILTER:
                    obj = self.__birth_gmm.sample()
                # end if

                gtt = GroundTruthTrack()
                gtt.begin_step = tt
                gtt.points.append(Position(obj[0], obj[1]))
                gtt.states = list()  # Added dynamically
                gtt.states.append(obj)
                gtt.survived = True  # Added dynamically
                self.__d.gtts.append(gtt)
            # end for

            # Create measurement data (simulated by p_d and added noise)
            ############################################################
            f_ds = Frame()
            f_mds = Frame()
            for track in [_ for _ in self.__d.gtts if _.survived]:
                obj = track.states[-1]  # Get current object state (therefore the last element in the objects history)

                if random.random() < self.__p_d:  # Target is detected
                    x = np.random.multivariate_normal(mean=np.zeros(self.__q.shape[0]), cov=self.__q, size=1)[0]  # Get first (and only) element
                    f_ds.add_detection(Position(obj[0] + x[0], obj[1] + x[1]))
                else:
                    f_mds.add_detection(Position(obj[0], obj[1]))
                # end if
            # end if

            self.__d.ds.add_frame(f_ds)
            self.__d.mds.add_frame(f_mds)

            # Clutter
            #########
            # Determine number of clutter
            n_clu = self.__panjer(num=self.__n_fa, var=self.__var_fa)

            # Add clutter (as false measurements) to measurements
            f_fas = Frame()
            for _ in range(n_clu):
                f_fas.add_detection(Position(self.__sample_random_from_range(self.__fov.x_min, self.__fov.x_max),
                                             self.__sample_random_from_range(self.__fov.y_min, self.__fov.y_max)))
            # end for
            self.__d.fas.add_frame(f_fas)

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
    # end def
# end class
