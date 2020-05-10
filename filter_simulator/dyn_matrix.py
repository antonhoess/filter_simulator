from abc import ABC, abstractmethod
import numpy as np
from enum import Enum
from typing import Iterable, Union, List, Sequence


class TransitionModel(Enum):
    INDIVIDUAL = 0
    PCW_CONST_WHITE_ACC_MODEL_2xND = 1
# end class


class DynMatrix(ABC):
    @abstractmethod
    def eval_f(self, dt: float) -> np.ndarray:
        pass
    # end dev

    @abstractmethod
    def eval_q(self, dt: float) -> np.ndarray:
        pass
    # end dev
# end class


class PcwConstWhiteAccelModelNd(DynMatrix):
    """Piecewise Constant White Acceleration Model"""
    def __init__(self, dim: int, sigma: Union[float, Iterable[float]]):
        self.__dim: int = dim

        if not isinstance(sigma, Sequence):
            sigma = [sigma]

        self.__sigma: List[float] = list(sigma)
        assert len(sigma) in (1, dim), "Parameter sigma is not 1 or equal to dim."

        self.__i: np.ndarray = np.eye(self.__dim)
        self.__o: np.ndarray = np.zeros((self.__dim, self.__dim))
    # end def

    def eval_f(self, dt: float = 1.) -> np.ndarray:
        f = np.block([[self.__i, dt * self.__i],
                      [self.__o, self.__i]])

        return f
    # end dev

    def eval_q(self, dt: float = 1.) -> np.ndarray:
        g = np.block([[0.5 * dt ** 2 * self.__i],
                      [dt * self.__i]])

        if len(self.__sigma) == 1:
            sigma_squared = self.__sigma[0] ** 2

        else:
            v_var = np.array(self.__sigma * 2, ndmin=2)
            sigma_squared = np.dot(v_var.T, v_var)
        # end if

        q = np.dot(g, g.T) * sigma_squared

        return q
    # end dev
# end class
