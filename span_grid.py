from __future__ import annotations
from typing import List, Optional
import numpy as np


class SpanDim:
    def __init__(self, range_min: float, range_max: float):
        self.min: float = range_min
        self.max: float = range_max
    # end def

    @property
    def min_int(self):
        return int(self.min)

    @property
    def max_int(self):
        return int(self.max)
# end class


class Span(list):
    def __init__(self, span_dims: Optional[List[SpanDim]] = None):
        super().__init__()

        if span_dims is not None:
            self.extend(span_dims)
    # end def

    def add_dim(self, range_min: float, range_max: float) -> bool:
        if range_min != range_max:
            self.append(SpanDim(range_min, range_max))

            return True
        else:
            return False
        # end if
    # end def

    @property
    def n_dims(self):
        return len(self)
    # end def
# end class


class Grid:
    def __init__(self, n_dims: int, span: Span, grid_size: List[int]):
        self.__grid = []
        self.__n_dims = n_dims

        if n_dims == 1 and span.n_dims == 1 and len(grid_size) == 1:
            # Shortuts
            s = span[0]
            gs = grid_size[0]

            self.__grid.append((np.arange(gs, dtype=float) / (gs - 1)) * (s.max - s.min) + s.min)
        # end if

        if n_dims == 2 and span.n_dims == 2 and len(grid_size) == 2:
            # Shortuts
            s0 = span[0]
            s1 = span[1]
            gs0 = grid_size[0]
            gs1 = grid_size[1]

            step_dim0 = (s0.max - s0.min) / (gs0 - 1)
            step_dim1 = (s1.max - s1.min) / (gs1 - 1)
            x, y = np.mgrid[s0.min:s0.max:step_dim0, s1.min:s1.max:step_dim1]
            self.__grid.append(x)
            self.__grid.append(y)
        # end if
    # end def

    @property
    def n_dims(self):
        return self.__n_dims

    @property
    def grid(self):
        return self.__grid
    # end def
# end class
