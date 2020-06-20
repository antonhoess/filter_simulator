from __future__ import annotations
from typing import List, Union, Optional
import numpy as np
import re
import random
from collections import Counter
from enum import Enum

from span_grid import Span


class DistMeasure(Enum):
    MAHALANOBIS_MOD = 0
    HELLINGER = 1
    BHATTACHARYYA = 2
# end class


class GmComponent:
    """Represents a single Gaussian component,
    with a float weight, vector location, matrix covariance.
    Note that we don't require a GM to sum to 1, since not always about probability densities."""
    def __init__(self, weight, loc, cov):
        self.weight = np.float64(weight)
        self.__loc = np.array(loc, dtype=np.float64)

        if cov is not None:
            self.__cov = np.array(cov, dtype=np.float64, ndmin=2)
            self.__cov = np.reshape(self.cov, (np.size(self.loc), np.size(self.loc)))  # Ensure shape matches loc shape

            # Precalculated values for evaluating Gaussian:
            k = self.loc.size
            self.__dmv_part1 = (2.0 * np.pi) ** (-k * 0.5)
            self.__dmv_part2 = np.power(np.linalg.det(self.cov), -0.5)
            self.__invcov = np.linalg.inv(self.cov)  # Use np.linalg.pinv if problems with singular matrix
        # end if
    # end def

    def __str__(self) -> str:
        return "GM component with w={:.6f}, loc=[{:.4f}, {:.4f}]".format(float(self.weight), float(self.loc[0]), float(self.loc[1]))

    def __repr__(self) -> str:
        return "GmComponent(weight=" +\
               repr(self.weight) +\
               ", loc=" + re.sub(r"\s+", "", repr(self.loc)).replace("array", "np.array") +\
               ", cov=" + re.sub(r"\s+", "", repr(self.cov)).replace("array", "np.array") + ")"

    @property
    def loc(self):
        return self.__loc

    @property
    def cov(self):
        return self.__cov

    def eval_at(self, x, ignore_weight=False):
        """Evaluate this multivariate normal component, at a location x.
        NB this does NOT APPLY THE WEIGHTING, simply for API similarity to the other method with this name."""
        x = np.array(x, dtype=np.float64)
        dev = x - self.loc
        dmv_part3 = np.exp(-0.5 * np.dot(np.dot(dev.T, self.__invcov), dev))

        density = self.__dmv_part1 * self.__dmv_part2 * dmv_part3

        if not ignore_weight:
            density *= self.weight

        return density

    # Caluclate the distance measure between two distributions
    # Alternatively use package 'dictances'? Maybe this does not handle the covariances?
    # See https://jneuroengrehab.biomedcentral.com/articles/10.1186/s12984-017-0283-5/tables/2
    def calc_dist_to(self, comp: GmComponent, dist_measure: DistMeasure = DistMeasure.MAHALANOBIS_MOD):
        s1 = self.cov
        s2 = comp.cov
        s = .5 * (s1 + s2)
        diff = self.loc - comp.loc
        diff = np.dot(np.dot(diff.T, np.linalg.inv(s)), diff)
        dist = None

        if dist_measure is DistMeasure.MAHALANOBIS_MOD:  # Modified Mahalanobis
            dist = .5 * np.sqrt(diff)

        elif dist_measure is DistMeasure.HELLINGER:
            # dist = 1 - (np.power(np.linalg.det(s1), .25) * np.power(np.linalg.det(s2), .25)) / np.sqrt(np.linalg.det(s)) * np.exp(-0.125 * diff)
            dist = 1 - np.sqrt(np.sqrt(np.linalg.det(np.dot(s1, s2))) / np.linalg.det(s)) * np.exp(-0.125 * diff)

        elif dist_measure is DistMeasure.BHATTACHARYYA:
            dist = np.sqrt(.125 * diff - .5 * np.log(np.linalg.det(s) / np.sqrt(np.linalg.det(s1) * np.linalg.det(s2))))
        # end if

        return float(dist)
    # end def

    def get_with_reduced_dims(self, which_dims: List[int]):
        dim_size = len(self.loc)

        which_dims.sort()

        if not (0 <= min(which_dims) <= max(which_dims) < dim_size):
            return None

        loc = np.take(self.loc, which_dims)
        cov = self.cov[np.ix_(which_dims, which_dims)]

        c = GmComponent(self.weight, loc, cov)

        return c
    # end def
# end class


class Gmm:
    def __init__(self, gm_comps: Optional[List[GmComponent]] = None) -> None:
        self.__gm_comp_set: List[GmComponent] = []

        if gm_comps is not None:
            self.add_comps(gm_comps)

    def __iter__(self) -> GmmIterator:
        return GmmIterator(self)

    def __getitem__(self, index: Union[int, slice]) -> Union[GmComponent, List[GmComponent]]:
        return self.__gm_comp_set[index]

    def __setitem__(self, index: int, value: GmComponent) -> None:
        self.__gm_comp_set[index] = value

    def __len__(self) -> int:
        return len(self.__gm_comp_set)

    def __str__(self) -> str:
        return "Gmm with {} components".format(len(self))

    def __repr__(self) -> str:
        return "Gmm(gm_comps=" + repr(self.__gm_comp_set) + ")"

    def __add__(self, gmm) -> Gmm:
        res_gmm = Gmm()

        for comp in self:
            res_gmm.add_comp(comp)

        for comp in gmm:
            res_gmm.add_comp(comp)

        return res_gmm

    def __radd__(self, gmm) -> Gmm:
        if gmm == 0:
            return self
        else:
            return self.__add__(gmm)

    def add_comp(self, gm_comp: GmComponent) -> None:
        self.__gm_comp_set.append(gm_comp)

    def add_comps(self, gm_comps):
        for gm_comp in gm_comps:
            self.__gm_comp_set.append(gm_comp)

    def get_comps(self) -> List[GmComponent]:
        return self.__gm_comp_set

    def get_total_weight(self) -> float:
        return sum(comp.weight for comp in self.__gm_comp_set)

    def sort(self, **kwargs) -> None:
        self.__gm_comp_set.sort(**kwargs)

    def reverse(self) -> None:
        self.__gm_comp_set.reverse()

    def pop(self, index=-1):
        return self.__gm_comp_set.pop(index)

    def eval_at(self, x, ignore_weight=False, which_dims=None):
        """Evaluate all multivariate normal components, at a location x."""
        x = np.array(x, dtype=np.float64)

        if which_dims is None:
            density = float(sum(comp.eval_at(x, ignore_weight=ignore_weight) for comp in self))
        else:
            x = np.take(x, which_dims)
            density = float(sum(comp.get_with_reduced_dims(which_dims).eval_at(x, ignore_weight=ignore_weight) for comp in self))
        # end if

        return density
    # end def

    def eval_list(self, points, which_dims=None):
        """Evaluates the GMM at a supplied list of points"""
        if not isinstance(points, np.ndarray):
            points = np.array(points)

        if which_dims is None:
            return [self.eval_at(p) for p in points]

        else:
            dim_size = points.shape[-1]

            which_dims = self.__get_which_dims(dim_size, which_dims)

            vals = np.zeros(np.prod(points.shape[:-1]))

            for comp in self:
                c = comp.get_with_reduced_dims(which_dims)

                for i in range(len(vals)):
                    p = np.take(np.array(points[i]), which_dims)
                    vals[i] += float(c.eval_at(p))
                # end for
            # end for

            return vals
        # end if
    # end def

    def eval_1d_list(self, points, which_dim=0):
        """Evaluates the GMM at a supplied list of points (1D only)"""
        vals = [0.] * len(points)

        for comp in self:
            c = GmComponent(comp.weight, comp.loc[which_dim], comp.cov[which_dim][which_dim])

            for i, p in enumerate(points):
                vals[i] += float(c.eval_at(p))
            # end for
        # end for

        return vals
    # end def

    def eval_1d_grid(self, grid, which_dim=0):
        """Evaluates the GMM on a uniformly-space grid of points (1D only)"""
        return self.eval_1d_list(grid.grid[0], which_dim)

    def eval_2d_grid(self, grid, which_dims=None):
        """Evaluates the GMM at a supplied list of points"""
        x, y = grid.grid

        points = np.stack((x, y), axis=-1).reshape(-1, 2)

        vals = self.eval_list(points, which_dims)

        return np.array(vals).reshape(x.shape)
    # end def

    def eval_along_line(self, span=None, gridsize=200, onlydims=None):
        """Evaluates the GMM on a uniformly-spaced line of points (i.e. a 1D line, though can be angled).
        'span' must be a list of (min, max) for each dimension, over which the line will iterate.
        'onlydims' if not nil, marginalises out (well, ignores) the nonlisted dims. All dims must still be listed in the spans, so put zeroes in."""
        if span is None:
            locs = np.array([comp.loc for comp in self]).T  # Note transpose - locs not a list of locations but a list of dimensions
            span = np.array([map(min, locs), map(max, locs)]).T  # Note transpose - span is an array of (min, max) for each dim
        else:
            span = np.array(span)
        steps = (np.arange(gridsize, dtype=float) / (gridsize-1))
        points = np.array(map(lambda aspan: steps * (aspan[1] - aspan[0]) + aspan[0], span)).T  # Transpose back to list of state-space points

        return self.eval_list(points, onlydims)
    # end def

    @staticmethod
    def __get_which_dims(dim_size, which_dims=None):
        which_dims = list(which_dims)

        if which_dims is None:
            which_dims = [d for d in range(dim_size)]
        else:
            which_dims.sort()
        # end if

        if not (0 <= min(which_dims) <= max(which_dims) < dim_size):
            return None

        return which_dims
    # end def

    def calc_span(self):
        n_dims = len(self[0].loc.shape)

        span = Span()
        for d in range(n_dims):
            locs = np.array([comp.loc[d] for comp in self])
            span.add_dim(np.min(locs), np.max(locs))
        # end for

        return span

    def sample(self) -> Optional[np.ndarray]:
        """Given a list of GmphdComponents, randomly samples a value from the density they represent"""
        samples = self.samples(1)

        if len(samples) > 0:
            return samples[0]
        else:
            return None
    # end def

    def samples(self, n) -> List[np.ndarray]:
        """Given a list of GmphdComponents, randomly samples n values from the density they represent"""
        samples = []

        if len(self.__gm_comp_set) > 0:
            rnd_gms = random.choices(range(len(self.__gm_comp_set)), [x.weight for x in self.__gm_comp_set], k=n)
            counter = Counter(rnd_gms)

            for g in counter.keys():
                cnt = counter.get(g)
                comp = self.__gm_comp_set[g]

                for s in np.random.multivariate_normal(comp.loc.flat, comp.cov, cnt):
                    samples.append(s)
                # end for
            # end for
        # end if

        return samples
    # end def

    def mult_comp_weight(self, factor: float):
        for comp in self:
            comp.weight *= factor
    # end def

    def add_comp_weight(self, summand: float):
        for comp in self:
            comp.weight += summand
    # end def

    def get_unified_comp(self, norm_cov=True):
        # Create unified new component from subsumed ones
        agg_weight = self.get_total_weight()

        loc = np.sum(np.array([comp.weight * comp.loc for comp in self]), 0) / agg_weight

        cov = 0
        for comp in self:
            dist = loc - comp.loc  # The original implementation used the weightiest component instead of the new location calculated just above
            cov += comp.weight * (comp.cov + np.dot(dist, dist.T))  # Why is no sqrt() used? # Wrong implementation in the original code using "dist * dist.T" instead of np.dot()
        # end for

        if norm_cov:
            cov /= agg_weight
        # end if

        return GmComponent(agg_weight, loc, cov)
# end class


class GmmIterator:
    def __init__(self, gmm: Gmm) -> None:
        self.__gmm: Gmm = gmm
        self.__index: int = 0

    def __next__(self) -> GmComponent:
        if self.__index < len(self.__gmm):
            result: GmComponent = self.__gmm[self.__index]
            self.__index += 1

            return result

        # End of iteration
        raise StopIteration
# end class
