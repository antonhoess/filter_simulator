#!/usr/bin/env python

# GM-PHD implementation in python by Dan Stowell.
# Based on the description in Vo and Ma (2006).
# (c) 2012 Dan Stowell and Queen Mary University of London.
# All rights reserved.
#
# NOTE: I AM NOT IMPLEMENTING SPAWNING, since I don't need it.
#   It would be straightforward to add it - see the original paper for how-to.
"""
This file is part of gmphd, GM-PHD filter in python by Dan Stowell.

    gmphd is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    gmphd is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with gmphd.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import annotations
from typing import List, Union, Optional
import numpy as np
from copy import deepcopy
import re
import random
from collections import Counter
from scipy.stats.distributions import chi2

from filter_simulator.common import Logging, Frame


class GmComponent:
    """Represents a single Gaussian component, 
    with a float weight, vector location, matrix covariance.
    Note that we don't require a GM to sum to 1, since not always about proby densities."""
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

    # Mahalanobis distance
    def calc_dist_to(self, comp: GmComponent):
        return float(np.dot(np.dot((comp.loc - self.loc).T, comp.__invcov), comp.loc - self.loc))

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

    def get_unified_comp(self):
        # Create unified new component from subsumed ones
        agg_weight = self.get_total_weight()

        loc = np.sum(np.array([comp.weight * comp.loc for comp in self]), 0) / agg_weight

        cov = 0
        for comp in self:
            dist = loc - comp.loc  # The original implementation used the weightiest component instead of the new location calculated just above
            cov += comp.weight * (comp.cov + np.dot(dist, dist.T))  # Why is no sqrt() used? # Wrong implementation in the original code using "dist * dist.T" instead of np.dot()
        # end for
        cov /= agg_weight

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


class GmPhdFilter:
    """Represents a set of modelling parameters and the latest frame's
       GMM estimate, for a GM-PHD model without spawning.

       Typical usage would be, for each frame of input data, to run:
          g.update(obs)
          g.prune()
          estimate = g.extract_states()

      'gmm' is an array of GmComponent items which makes up
           the latest GMM, and updated by the update() call. 
           It is initialised as empty.

    Test code example (1D data, with new trails expected at around 100):
from gmphd import *
g = Gmphd([GmphdComponent(1, [100], [[10]])], 0.9, 0.9, [[1]], [[1]], [[1]], [[1]], 0.000002)
g.update([[30], [67.5]])
g.gmmplot1d()
g.prune()
g.gmmplot1d()

g.gmm

[(float(comp.loc), comp.weight) for comp in g.gmm]
    """

    def __init__(self, birth_gmm: List[GmComponent], survival: float, detection: float, f: np.ndarray, q: np.ndarray, h: np.ndarray, r: np.ndarray,
                 rho_fa: float, gate_thresh: Optional[float], logging: Logging = Logging.INFO):
        """
          'birthgmm' is an array of GmphdComponent items which makes up
               the GMM of birth probabilities.
          'survival' is survival probability.
          'detection' is detection probability.
          'f' is state transition matrix F.
          'q' is the process noise covariance Q.
          'h' is the observation matrix H.
          'r' is the observation noise covariance R.
          'p_fa' is the p_fa intensity.
          """
        self.__gmm: Gmm = Gmm()                   # Empty - things will need to be born before we observe them
        self.__birth_gmm: Gmm = Gmm(birth_gmm)
        self.__survival = np.float64(survival)    # p_{s,k}(x) in paper
        self.__detection = np.float64(detection)  # p_{d,k}(x) in paper
        self.__f = np.array(f, dtype=np.float64)  # State transition matrix      (F_k-1 in paper)
        self.__q = np.array(q, dtype=np.float64)  # Process noise covariance     (Q_k-1 in paper)
        self.__h = np.array(h, dtype=np.float64)  # Observation matrix           (H_k in paper)
        self.__r = np.array(r, dtype=np.float64)  # Observation noise covariance (R_k in paper)
        self.__rho_fa = np.float64(rho_fa)        # Clutter intensity (KAU in paper)
        self.__gate_thresh = chi2.ppf(gate_thresh, df=f.shape[0]) if gate_thresh else None  # Calculate the inverse chi^2

        self.__logging = logging
        self.__cur_frame: Optional[Frame] = None
    # end def

    @property
    def _gmm(self) -> Gmm:
        return self.__gmm
    # end def

    @_gmm.setter
    def _gmm(self, value: Gmm) -> None:
        self.__gmm = value
    # end def

    @property
    def _birth_gmm(self) -> Gmm:
        return self.__birth_gmm
    # end def

    # @_birth_gmm.setter
    # def _birth_gmm(self, value: Gmm) -> None:
    #     self.__birth_gmm = value
    # # end def

    @property
    def _p_s(self) -> float:
        return self.__survival
    # end def

    @property
    def _p_d(self) -> float:
        return self.__detection
    # end def

    @property
    def _f(self) -> np.ndarray:
        return self.__f
    # end def

    @property
    def _q(self) -> np.ndarray:
        return self.__q
    # end def

    @property
    def _h(self) -> np.ndarray:
        return self.__h
    # end def

    @property
    def _r(self) -> np.ndarray:
        return self.__r
    # end def

    @property
    def _cur_frame(self) -> Optional[Frame]:
        return self.__cur_frame
    # end def

    @_cur_frame.setter
    def _cur_frame(self, value: Optional[Frame]) -> None:
        self.__cur_frame = value
    # end def

    def _predict_and_update(self, observations):
        """Run a single GM-PHD step given a new frame of observations.
          'obs' is an array (a set) of this frame's observations.
          Based on Table 1 from Vo and Ma paper."""
        # Step 1 - prediction for birth targets
        #######################################
        born: Gmm = Gmm([deepcopy(comp) for comp in self.__birth_gmm])
        # The original paper would do a spawning iteration as part of step 1.
        spawned = Gmm()  # XXX not implemented

        # Step 2 - prediction for existing targets
        ##########################################
        updated: Gmm = Gmm()
        for comp in self.__gmm:
            updated.add_comp(GmComponent(weight=self.__survival * comp.weight,
                                         loc=np.dot(self.__f, comp.loc),  # Motion model: x = F * x
                                         cov=np.dot(np.dot(self.__f, comp.cov), self.__f.T) + self.__q))  # Covariance matrix: P = F * P * F.T + Q
        # end for

        predicted: Gmm = born + spawned + updated

        # Step 3 - construction of PHD update components (using Kalman update rules)
        ############################################################################
        # These two are the mean and covariance of the expected observation
        nu = [np.dot(self.__h, comp.loc) for comp in predicted]  # nu = H * x # H (observation model) maps the true state space into the observed space
        s = [np.dot(np.dot(self.__h, comp.cov), self.__h.T) + self.__r for comp in predicted]  # Innovation covariance: S = H * P * H.T + R
        s_inv = [np.linalg.inv(s_) for s_ in s] if self.__gate_thresh else None  # Computationally expensive and will get used many times below

        # Not sure about any physical interpretation of these two...
        k = [np.dot(np.dot(comp.cov, self.__h.T), np.linalg.inv(s[index])) for index, comp in enumerate(predicted)]  # Kalman Gain: K = P * H.T * S^{-1}
        p = [np.dot(np.eye(len(k[index])) - np.dot(k[index], self.__h), comp.cov) for index, comp in enumerate(predicted)]  # Updated (a posteriori) estimate covariance: P = (I - K * H) * P
        # If there's numeric instability, the Joseph's form might get used: P = (I - K * H) * P * (I - K * H).T + K * R * K.T

        # Step 4 - update using observations
        ####################################
        # The 'predicted' components are kept, with a decay
        new_gmm = predicted
        new_gmm.mult_comp_weight(1. - self.__detection)

        # Then more components are added caused by each observations interaction with existing component
        for obs in observations:
            obs = np.array(obs, dtype=np.float64)
            new_gmm_partial = Gmm()

            for j, comp in enumerate(predicted):
                y = obs - nu[j]  # Kalman Innovation Residual: y = z - H * x

                # Gating
                if self.__gate_thresh:
                    gate = np.dot(np.dot(y, s_inv[j]), y.T)

                    if gate > self.__gate_thresh:
                        continue  # Discard the combination if the hypothesis location x is outside of the threshold around the measurement z
                    # end if
                # end if

                # weight: Depending on how good the measurement hits the GM component's mean
                # loc: Updated (a posteriori) state estimate
                new_gmm_partial.add_comp(GmComponent(
                    weight=self.__detection * GmComponent(comp.weight, loc=nu[j], cov=s[j]).eval_at(obs),
                    loc=comp.loc + np.dot(k[j], y),
                    cov=p[j])
                )
            # end for

            # The Kappa thing (clutter and reweight)
            weight_sum = new_gmm_partial.get_total_weight()
            reweighter = 1. / (self.__rho_fa + weight_sum)
            new_gmm_partial.mult_comp_weight(reweighter)

            new_gmm += new_gmm_partial
        # end for

        self.__gmm = new_gmm

    def _prune(self, trunc_thresh=1e-6, merge_thresh=0.01, max_components=100):
        """Prune the GMM. Alters model state.
          Based on Table 2 from Vo and Ma paper."""
        weight_sums: List[float] = list()
        orig_len: int = 0
        trunc_len: int = 0

        # Truncation is easy
        source_gmm: Gmm = Gmm([comp for comp in list(filter(lambda comp: comp.weight > trunc_thresh, self.__gmm))])

        weight_sums.append(self.__gmm.get_total_weight())  # Diagnostic
        weight_sums.append(source_gmm.get_total_weight())
        if self.__logging >= Logging.INFO:
            orig_len = len(self.__gmm)
            trunc_len = len(source_gmm)
        # end if

        # Iterate to build the new GMM
        new_gmm: Gmm = Gmm()

        while len(source_gmm) > 0:
            # Find weightiest old component and pull it out
            w_index = int(np.argmax([comp.weight for comp in source_gmm]))
            weightiest: GmComponent = source_gmm.pop(w_index)

            # Find all nearby ones and pull them out
            distances = [weightiest.calc_dist_to(comp) for comp in source_gmm]
            do_subsume = np.array([dist <= merge_thresh for dist in distances])
            subsumed_gmm: Gmm = Gmm([weightiest])

            if np.any(do_subsume):
                self.__logging.print_verbose(Logging.DEBUG, "Subsuming the following locations into weightest with loc %s and weight %g (cov %s):" %
                                             (','.join([str(x) for x in weightiest.loc.flat]), weightiest.weight, ','.join([str(x) for x in weightiest.cov.flat])))
                self.__logging.print_verbose(Logging.DEBUG, str(list([comp.loc[0] for comp in list(np.array(source_gmm.get_comps())[do_subsume])])))
                subsumed_gmm.add_comps(list(np.array(source_gmm.get_comps())[do_subsume]))
                source_gmm = Gmm(list(np.array(source_gmm.get_comps())[~do_subsume]))
            # end if

            # Create unified new component from subsumed ones
            new_gmm.add_comp(subsumed_gmm.get_unified_comp())
        # end while

        # Now ensure the number of components is within the limit, keeping the weightiest
        new_gmm.sort(key=lambda comp: comp.weight, reverse=True)
        self.__gmm = Gmm(new_gmm[:max_components])

        weight_sums.append(new_gmm.get_total_weight())
        weight_sums.append(self.__gmm.get_total_weight())
        if self.__logging >= Logging.DEBUG:
            self.__logging.print("prune(): %i -> %i -> %i -> %i" % (orig_len, trunc_len, len(new_gmm), len(self.__gmm)))
            self.__logging.print("prune(): weight_sums %g -> %g -> %g -> %g" % (weight_sums[0], weight_sums[1], weight_sums[2], weight_sums[3]))
        # end if

        # Pruning should not alter the total weightsum (which relates to total num items) - so we renormalize
        weight_norm = weight_sums[0] / weight_sums[3]  # Not in the original paper
        self.__gmm.mult_comp_weight(weight_norm)

    def _extract_states(self, bias: float = 1.0, use_integral: bool = False):
        if not use_integral:
            items = self.__extract_states(bias)
        else:
            items = self.__extract_states_using_integral(bias)
        # end if

        for x in items:
            self.__logging.print_verbose(Logging.DEBUG, x.T)
        # end for

        return items
    # end def

    def __extract_states(self, bias: float = 1.0):
        """Extract the multiple-target states from the GMM.
          Returns a list of target states; doesn't alter model state.
          Based on Table 3 from Vo and Ma paper.
          I added the 'bias' factor, by analogy with the other method below."""
        items = []

        self.__logging.print_verbose(Logging.DEBUG, "weights:")
        self.__logging.print_verbose(Logging.DEBUG, str([round(comp.weight, 7) for comp in self.__gmm]))

        for comp in self.__gmm:
            val = comp.weight * bias

            if val > .5:
                for _ in range(int(round(val))):
                    items.append(deepcopy(comp.loc))
            # end if
        # end for

        return items

    def __extract_states_using_integral(self, bias: float = 1.0):
        """Extract states based on the expected number of states from the integral of the intensity.
        This is NOT in the GMPHD paper; added by Dan.
        "bias" is a multiplier for the est number of items.
        """
        num_to_add = int(round(bias * self.__gmm.get_total_weight()))
        self.__logging.print_verbose(Logging.DEBUG, "bias is %g, num_to_add is %i" % (bias, num_to_add))

        # A temporary list of peaks p will gradually be decimated as we steal from its highest peaks
        peaks = [GmComponent(comp.weight, comp.loc, None) for comp in self.__gmm]

        items = []
        while num_to_add > 0:
            # Find weightiest peak
            p_index = 0
            p_weight = 0
            for p, peak in enumerate(peaks):
                if peak.weight > p_weight:
                    p_index = p
                    p_weight = peak.weight
                # end if
            # end for

            # Add the winner
            items.append(deepcopy(peaks[p_index].loc))
            peaks[p_index].weight -= 1.0
            num_to_add -= 1
        # end while

        return items
    # end def
# end class


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

