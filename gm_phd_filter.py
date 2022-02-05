# Heavily modified version of the GM-PHD implementation in python by Dan Stowell.
# Based on the description in Vo and Ma (2006): The Gaussian mixture probability hypothesis density filter.
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
from typing import List, Optional
import numpy as np
from copy import deepcopy
from scipy.stats.distributions import chi2

from filter_simulator.common import Logging, Frame
from gm import GmComponent, Gmm, DistMeasure


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
        self._gmm: Gmm = Gmm()                   # Empty - things will need to be born before we observe them
        self._birth_gmm: Gmm = Gmm(birth_gmm)
        self._survival = np.float64(survival)    # p_{s,k}(x) in paper
        self._detection = np.float64(detection)  # p_{d,k}(x) in paper
        self._f = np.array(f, dtype=np.float64)  # State transition matrix      (F_k-1 in paper)
        self._q = np.array(q, dtype=np.float64)  # Process noise covariance     (Q_k-1 in paper)
        self._h = np.array(h, dtype=np.float64)  # Observation matrix           (H_k in paper)
        self._r = np.array(r, dtype=np.float64)  # Observation noise covariance (R_k in paper)
        self._rho_fa = np.float64(rho_fa)        # Clutter intensity (KAU in paper)
        self._gate_thresh = chi2.ppf(gate_thresh, df=f.shape[0]) if gate_thresh else None  # Calculate the inverse chi^2

        self._logging = logging
        self._cur_frame: Optional[Frame] = None
    # end def

    @property
    def gmm(self) -> Gmm:
        return self._gmm
    # end def

    @gmm.setter
    def gmm(self, value: Gmm) -> None:
        self._gmm = value
    # end def

    @property
    def birth_gmm(self) -> Gmm:
        return self._birth_gmm
    # end def

    @property
    def p_s(self) -> float:
        return self._survival
    # end def

    @property
    def p_d(self) -> float:
        return self._detection
    # end def

    @property
    def f(self) -> np.ndarray:
        return self._f
    # end def

    @property
    def q(self) -> np.ndarray:
        return self._q
    # end def

    @property
    def h(self) -> np.ndarray:
        return self._h
    # end def

    @property
    def r(self) -> np.ndarray:
        return self._r
    # end def

    @property
    def cur_frame(self) -> Optional[Frame]:
        return self._cur_frame
    # end def

    @cur_frame.setter
    def cur_frame(self, value: Optional[Frame]) -> None:
        self._cur_frame = value
    # end def

    def predict_and_update(self, observations):
        """Run a single GM-PHD step given a new frame of observations.
          'obs' is an array (a set) of this frame's observations.
          Based on Table 1 from Vo and Ma paper."""
        # Step 1 - prediction for birth targets
        #######################################
        born: Gmm = Gmm([deepcopy(comp) for comp in self.birth_gmm])
        # The original paper would do a spawning iteration as part of step 1.
        spawned = Gmm()  # Not implemented

        # Step 2 - prediction for existing targets
        ##########################################
        updated: Gmm = Gmm()
        for comp in self.gmm:
            updated.add_comp(GmComponent(weight=self._survival * comp.weight,
                                         loc=np.dot(self.f, comp.loc),  # Motion model: x = F * x
                                         cov=np.dot(np.dot(self.f, comp.cov), self.f.T) + self.q))  # Covariance matrix: P = F * P * F.T + Q
        # end for

        predicted: Gmm = born + spawned + updated

        # Step 3 - construction of PHD update components (using Kalman update rules)
        ############################################################################
        # These two are the mean and covariance of the expected observation
        nu = [np.dot(self._h, comp.loc) for comp in predicted]  # nu = H * x # H (observation model) maps the true state space into the observed space
        s = [np.dot(np.dot(self._h, comp.cov), self._h.T) + self._r for comp in predicted]  # Innovation covariance: S = H * P * H.T + R
        s_inv = [np.linalg.inv(s_) for s_ in s] if self._gate_thresh else None  # Computationally expensive and will get used many times below

        # Not sure about any physical interpretation of these two...
        k = [np.dot(np.dot(comp.cov, self._h.T), np.linalg.inv(s[index])) for index, comp in enumerate(predicted)]  # Kalman Gain: K = P * H.T * S^{-1}
        p = [np.dot(np.eye(len(k[index])) - np.dot(k[index], self._h), comp.cov) for index, comp in enumerate(predicted)]  # Updated (a posteriori) estimate covariance: P = (I - K * H) * P
        # If there's numeric instability, the Joseph's form might get used: P = (I - K * H) * P * (I - K * H).T + K * R * K.T

        # Step 4 - update using observations
        ####################################
        new_gmm = Gmm()

        # Then more components are added caused by each observations interaction with existing component
        for obs in observations:
            obs = np.array(obs, dtype=np.float64)
            new_gmm_partial = Gmm()

            for j, comp in enumerate(predicted):
                y = obs - nu[j]  # Kalman Innovation Residual: y = z - H * x

                # Gating
                if self._gate_thresh:
                    gate = np.dot(np.dot(y.T, s_inv[j]), y)

                    if gate > self._gate_thresh:
                        continue  # Discard the combination if the hypothesis location x is outside of the threshold around the measurement z
                    # end if
                # end if

                # weight: Depending on how good the measurement hits the GM component's mean
                # loc: Updated (a posteriori) state estimate
                new_gmm_partial.add_comp(GmComponent(
                    weight=self._detection * GmComponent(comp.weight, loc=nu[j], cov=s[j]).eval_at(obs),
                    loc=comp.loc + np.dot(k[j], y),
                    cov=p[j])
                )
            # end for

            # The Kappa thing (clutter and reweight)
            weight_sum = new_gmm_partial.get_total_weight()
            reweighter = 1. / (self._rho_fa + weight_sum)
            new_gmm_partial.mult_comp_weight(reweighter)

            new_gmm += new_gmm_partial
        # end for

        # The 'predicted' components are kept, with a decay (= missed detections) - this needs to be done after the loop above, since the loop needs the original values
        predicted.mult_comp_weight(1. - self._detection)

        # Build new GMM
        self.gmm = new_gmm + predicted

    def prune(self, trunc_thresh: float = 1e-6, merge_dist_measure: DistMeasure = DistMeasure.MAHALANOBIS_MOD, merge_thresh: float = 0.01, max_components: int = 100):
        """Prune the GMM. Alters model state.
          Based on Table 2 from Vo and Ma paper."""
        weight_sums: List[float] = list()
        orig_len: int = 0
        trunc_len: int = 0

        # Truncation is easy
        source_gmm: Gmm = Gmm([comp for comp in list(filter(lambda comp: comp.weight > trunc_thresh, self.gmm))])

        weight_sums.append(self.gmm.get_total_weight())  # Diagnostic
        weight_sums.append(source_gmm.get_total_weight())
        if self._logging >= Logging.INFO:
            orig_len = len(self.gmm)
            trunc_len = len(source_gmm)
        # end if

        # Iterate to build the new GMM
        new_gmm: Gmm = Gmm()
        while len(source_gmm) > 0:
            # Find weightiest old component and pull it out
            w_index = int(np.argmax([comp.weight for comp in source_gmm]))
            weightiest: GmComponent = source_gmm.pop(w_index)

            # Find all nearby ones and pull them out
            distances = [weightiest.calc_dist_to(comp, merge_dist_measure) for comp in source_gmm]
            do_subsume = np.array([dist <= merge_thresh for dist in distances])
            subsumed_gmm: Gmm = Gmm([weightiest])

            if np.any(do_subsume):
                self._logging.print_verbose(Logging.DEBUG, f"Subsuming the following locations into weightest with loc {','.join([str(x) for x in weightiest.loc.flat])} and weight "
                                                            f"{weightiest.weight} (cov {','.join([str(x) for x in weightiest.cov.flat])}):")
                self._logging.print_verbose(Logging.DEBUG, str(list([comp.loc[0] for comp in list(np.array(source_gmm.get_comps())[do_subsume])])))
                subsumed_gmm.add_comps(list(np.array(source_gmm.get_comps())[do_subsume]))
                source_gmm = Gmm(list(np.array(source_gmm.get_comps())[~do_subsume]))
            # end if

            # Create unified new component from subsumed ones
            new_gmm.add_comp(subsumed_gmm.get_unified_comp())
        # end while

        # Now ensure the number of components is within the limit, keeping the weightiest
        new_gmm.sort(key=lambda comp: comp.weight, reverse=True)
        self.gmm = Gmm(new_gmm[:max_components])

        weight_sums.append(new_gmm.get_total_weight())
        weight_sums.append(self.gmm.get_total_weight())
        if self._logging >= Logging.DEBUG:
            self._logging.print("prune(): %i -> %i -> %i -> %i" % (orig_len, trunc_len, len(new_gmm), len(self.gmm)))
            self._logging.print("prune(): weight_sums %g -> %g -> %g -> %g" % (weight_sums[0], weight_sums[1], weight_sums[2], weight_sums[3]))
        # end if

        # Pruning should not alter the total weightsum (which relates to total num items) - so we renormalize
        weight_norm = weight_sums[0] / weight_sums[3]  # Not in the original paper
        self.gmm.mult_comp_weight(weight_norm)

    def extract_states(self, bias: float = 1.0, use_integral: bool = False):
        if not use_integral:
            items = self._extract_states(bias)
        else:
            items = self._extract_states_using_integral(bias)
        # end if

        for x in items:
            self._logging.print_verbose(Logging.DEBUG, x.T)
        # end for

        return items
    # end def

    def _extract_states(self, bias: float = 1.0):
        """Extract the multiple-target states from the GMM.
          Returns a list of target states; doesn't alter model state.
          Based on Table 3 from Vo and Ma paper.
          I added the 'bias' factor, by analogy with the other method below."""
        items = list()

        self._logging.print_verbose(Logging.DEBUG, "weights:")
        self._logging.print_verbose(Logging.DEBUG, str([round(comp.weight, 7) for comp in self.gmm]))

        for comp in self.gmm:
            val = comp.weight * bias

            if val > .5:
                for _ in range(int(round(val))):
                    items.append(deepcopy(comp.loc))
            # end if
        # end for

        return items

    def _extract_states_using_integral(self, bias: float = 1.0):
        """Extract states based on the expected number of states from the integral of the intensity.
        This is NOT in the GMPHD paper; added by Dan.
        "bias" is a multiplier for the est number of items.
        """
        num_to_add = int(round(bias * self._gmm.get_total_weight()))
        self._logging.print_verbose(Logging.DEBUG, "bias is %g, num_to_add is %i" % (bias, num_to_add))

        # A temporary list of peaks p will gradually be decimated as we steal from its highest peaks
        peaks = [GmComponent(comp.weight, comp.loc, None) for comp in self._gmm]

        items = list()
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
