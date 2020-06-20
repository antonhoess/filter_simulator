from __future__ import annotations
from typing import List, Tuple, Optional
import bisect
import copy
import math
import random
import numpy as np

from filter_simulator.common import Logging, Limits, Frame


class WeightedDistribution:
    def __init__(self, state: List[Particle]) -> None:
        self.__state: List[Particle] = [p for p in state if p.w > 0]
        self.__distribution: List[float] = []

        accum: float = .0
        for x in self.__state:
            accum += x.w
            self.__distribution.append(accum)

    def pick(self) -> Optional[Particle]:
        try:
            # Due to numeric problems, the weight don't sum up to 1.0 after normalization,
            # so we can't pick from a uniform distribution in range [0, 1]
            return self.__state[bisect.bisect_left(self.__distribution, random.uniform(0, self.__distribution[-1]))]

        except IndexError:
            # Happens when all particles are improbable w=0
            return None
    # end def
# end class


class Particle:
    def __init__(self, x: float, y: float, w: float = 1.) -> None:
        self.x: float = x
        self.y: float = y
        self.vx: float = .0
        self.vy: float = .0
        self.w: float = w

    def __repr__(self) -> str:
        return "(%f, %f, w=%f)" % (self.x, self.y, self.w)

    @property
    def xy(self) -> Tuple[float, float]:
        return self.x, self.y

    @classmethod
    def create_random(cls, count: int, limits: Limits) -> List[Particle]:
        return [cls(random.uniform(limits.x_min, limits.x_max), random.uniform(limits.y_min, limits.y_max))
                for _ in range(0, count)]

    def move_by(self, x, y) -> None:
        self.x += x
        self.y += y
# end class


class ParticleFilter:
    def __init__(self, n_part: int, s_gauss: float, noise: float, speed: float, fov: Limits, logging: Logging):
        self.__n_part: int = n_part
        self.__s_gauss: float = s_gauss
        self.__noise: float = noise
        self.__use_speed: bool = True  # XXX Param? # If we use speed to update the particle's position, the particle
        # might fly out of the curve, since their value of the gaussian kernel for calculating the importance
        # weight will be less
        self.__speed: float = speed
        self.__resampling: bool = True
        self.__det_borders: Limits = fov
        self.__logging: Logging = logging
        self.__cur_frame: Optional[Frame] = None
        self.__particles: List[Particle] = Particle.create_random(self.__n_part, self.__det_borders)

    @property
    def particles(self) -> List[Particle]:
        return self.__particles

    @property
    def s_gauss(self) -> float:
        return self.__s_gauss

    @property
    def cur_frame(self) -> Optional[Frame]:
        return self.__cur_frame

    @cur_frame.setter
    def cur_frame(self, value: Optional[Frame]) -> None:
        self.__cur_frame = value

    def update(self):
        # Update particle weight according to how good every particle matches the nearest detection
        if len(self.__cur_frame) > 0:
            for p in self.__particles:
                # Use the detection position nearest to the current particle
                p_d_min: Optional[float] = None

                for det in self.__cur_frame:
                    d_x: float = p.x - det.x
                    d_y: float = p.y - det.y
                    p_d: float = math.sqrt(d_x * d_x + d_y * d_y)

                    if p_d_min is None or p_d < p_d_min:
                        p_d_min = p_d
                    # end if
                # end for

                p.w = self._w_gauss(p_d_min, self.__s_gauss)
            # end for
        # end if
    # end def

    def predict(self):
        # Move all particles according to belief of movement
        if len(self.__cur_frame) > 0:
            for p in self.__particles:
                p_x = p.x
                p_y = p.y

                idx: Optional[int] = None
                p_d_min: float = .0

                # Determine detection nearest to the current particle
                for _idx, det in enumerate(self.__cur_frame):
                    d_x: float = (det.x - p.x)
                    d_y: float = (det.y - p.y)
                    p_d: float = math.sqrt(d_x * d_x + d_y * d_y)

                    if idx is None or p_d < p_d_min:
                        idx = _idx
                        p_d_min = p_d
                    # end if
                # end for

                # Calc distance and angle to nearest detection
                det = self.__cur_frame[idx]
                d_x: float = (det.x - p.x)
                d_y: float = (det.y - p.y)

                p_d: float = math.sqrt(d_x * d_x + d_y * d_y)
                angle: float = math.atan2(d_y, d_x)

                # Use a convex combination of...
                # .. the particles speed
                d_x = self.__speed * p_d * math.cos(angle)
                d_y = self.__speed * p_d * math.sin(angle)

                # .. and the 'speed', the particle progresses towards the new detection
                if self.__use_speed:
                    d_x += (1 - self.__speed) * p.vx
                    d_y += (1 - self.__speed) * p.vy
                # end if

                # Move particle towards nearest detection
                p.move_by(d_x, d_y)

                # Calc particle speed for next time step
                p.vx = p.x - p_x
                p.vy = p.y - p_y

                # Add some noise for more stability
                p.move_by(*self._create_gaussian_noise(self.__noise, 0, 0))
            # end for
        # end if
    # end def

    def resample(self):
        # Resampling
        if self.__resampling:
            new_particles: List[Particle] = []

            # Normalize weights
            nu: float = sum(p.w for p in self.__particles)
            self.__logging.print_verbose(Logging.DEBUG, "nu = {}".format(nu))

            if nu > 0:
                for p in self.__particles:
                    p.w = p.w / nu
            # end if

            # Create a weighted distribution, for fast picking
            dist: WeightedDistribution = WeightedDistribution(self.__particles)
            self.__logging.print_verbose(Logging.INFO, "# particles: {}".format(len(self.__particles)))

            cnt: int = 0
            for _ in range(len(self.__particles)):
                p: Particle = dist.pick()

                if p is None:  # No pick b/c all totally improbable
                    new_particle: Particle = Particle.create_random(1, self.__det_borders)[0]
                    cnt += 1
                else:
                    new_particle = copy.deepcopy(p)
                    new_particle.w = 1.
                # end if

                new_particle.move_by(*self._create_gaussian_noise(self.__noise, 0, 0))
                new_particles.append(new_particle)
            # end for

            self.__logging.print_verbose(Logging.INFO, "# particles newly created: {}".format(cnt))
            self.__particles = new_particles
        # end if
    # end def

    @staticmethod
    def _create_noise(level: float, *coords) -> List[float]:
        return [x + random.uniform(-level, level) for x in coords]

    @staticmethod
    def _create_gaussian_noise(level: float, *coords) -> List[float]:
        return [x + np.random.normal(.0, level) for x in coords]

    # Gaussian kernel to transform values near the particle => 1, further away => 0
    @staticmethod
    def _w_gauss(x: float, sigma: float) -> float:
        g = math.e ** -(x * x / (2 * sigma * sigma))

        return g
# end class
