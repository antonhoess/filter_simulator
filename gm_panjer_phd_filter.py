# Author: Anton Höß
# Heavily modified code from the Python PHD-Filter of Dan Stowell (2012) extended by
# Code of PHD-Panjer-Filter from Isabel Schlangen [is] (2016) translated from MATLAB to Python:
# A second-order PHD filter with mean and variance in target number

from __future__ import annotations
from typing import List, Optional
import numpy as np
from copy import deepcopy
from scipy.stats.distributions import chi2
import math

from filter_simulator.common import Logging, Frame
from gm import GmComponent, Gmm, DistMeasure


class GmPanjerPhdFilter:
    def __init__(self, birth_gmm: List[GmComponent], var_birth: float, survival: float, detection: float, f: np.ndarray, q: np.ndarray, h: np.ndarray, r: np.ndarray,
                 rho_fa: float, gate_thresh: Optional[float], logging: Logging = Logging.INFO):
        self._gmm: Gmm = Gmm()                   # Empty - things will need to be born before we observe them
        self._variance = .0                      # Initial variance
        self._birth_gmm: Gmm = Gmm(birth_gmm)
        self._var_birth = var_birth
        self._survival = np.float64(survival)    # p_{s,k}(x)
        self._detection = np.float64(detection)  # p_{d,k}(x)
        self._f = np.array(f, dtype=np.float64)  # State transition matrix
        self._q = np.array(q, dtype=np.float64)  # Process noise covariance
        self._h = np.array(h, dtype=np.float64)  # Observation matrix
        self._r = np.array(r, dtype=np.float64)  # Observation noise covariance
        self._rho_fa = np.float64(rho_fa)        # Clutter intensity
        self._gate_thresh = chi2.ppf(gate_thresh, df=f.shape[0]) if gate_thresh else None  # Calculate the inverse chi^2

        self._logging = logging
        self._cur_frame: Optional[Frame] = None
    # end def

    @property
    def gmm(self) -> Gmm:
        return self._gmm
    # end def

    @property
    def variance(self) -> float:
        return self._variance
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

    def predict(self):
        # Prediction for birth targets
        ##############################
        born: Gmm = Gmm([deepcopy(comp) for comp in self.birth_gmm])

        # Prediction for existing targets
        #################################
        mean_posterior = self.gmm.get_total_weight()
        nu_x = self._variance + mean_posterior ** 2 - mean_posterior

        # Predict components
        updated: Gmm = Gmm()
        for comp in self.gmm:
            updated.add_comp(GmComponent(weight=self._survival * comp.weight,
                                         loc=np.dot(self.f, comp.loc),  # Motion model: x = F * x
                                         cov=np.dot(np.dot(self.f, comp.cov), self.f.T) + self.q))  # Covariance matrix: P = F * P * F.T + Q
        # end for

        # Calculate predicted variance
        mu_pred = updated.get_total_weight()  # Predicted intensity
        # XXX On the second run here is already a first error, because here results another sum as in Matlab and in fact my GMM components have different weights.
        var_pred = mu_pred + self._survival ** 2 * nu_x - mu_pred ** 2  # Prediction of variance without birth
        self._variance = var_pred + self._var_birth  # Full variance including birth

        # Assemble the total GMM
        self.gmm = updated + born
    # end def

    def update(self, observations):
        """Run a single update step given a new frame of observations.
          'observations' is an array (a set) of this frame's observations."""
        n_meas = len(observations)  # Number of measurements

        # Predicted alpha and beta
        ##########################
        mean_pred = self.gmm.get_total_weight()  # Predicted intensity
        var_new = self._variance

        # The binomial case
        if var_new < mean_pred:
            n = math.ceil(mean_pred)  # n >= mu
            dist = self._variance  # Doesn't work with negative variance!

            while True:
                var_tmp = mean_pred * (1 - mean_pred / n)

                if abs(var_tmp - self._variance) < dist:
                    dist = abs(var_tmp - self._variance)
                    var_new = var_tmp

                else:
                    break
                # end if

                n += 1
            # end if
        # end if

        # % check if the values are okay in the (positive) binomial case
        # if cst.nBirth>cst.varBirth
        #     alphaval = cst.nBirth^2/(cst.varBirth-cst.nBirth);
        #     alphaval = floor(alphaval); % must not be a non-integer!
        #     cst.varBirth = cst.nBirth + cst.nBirth^2/alphaval;
        #     fprintf('Warning: binomial birth. Adjusted variance to %g\n',cst.varBirth);
        # end

        beta_pred = mean_pred / (var_new - mean_pred) if var_new - mean_pred != 0 else np.inf  # Avoid error on division by zero
        alpha_pred = mean_pred * beta_pred

        # Construction of PHD update components (using Kalman update rules)
        ###################################################################
        # These two are the mean and covariance of the expected observation
        nu = [np.dot(self._h, comp.loc) for comp in self.gmm]  # nu = H * x # H (observation model) maps the true state space into the observed space
        s = [np.dot(np.dot(self._h, comp.cov), self._h.T) + self._r for comp in self.gmm]  # Innovation covariance: S = H * P * H.T + R
        s_inv = [np.linalg.inv(s_) for s_ in s] if self._gate_thresh else None  # Computationally expensive and will get used many times below - calculate only when gating is active

        # Not sure about any physical interpretation of these two...
        k = [np.dot(np.dot(comp.cov, self._h.T), np.linalg.inv(s[index])) for index, comp in enumerate(self.gmm)]  # Kalman Gain: K = P * H.T * S^{-1}
        p = [np.dot(np.eye(len(k[index])) - np.dot(k[index], self._h), comp.cov) for index, comp in enumerate(self.gmm)]  # Updated (a posteriori) estimate covariance: P = (I - K * H) * P
        # If there's numeric instability, the Joseph's form might get used: P = (I - K * H) * P * (I - K * H).T + K * R * K.T

        # Update using observations
        ###########################
        new_gmm_per_measurement = list()

        # Then more components are added caused by each observations interaction with existing component
        mu_z = np.zeros(len(observations))
        for o, obs in enumerate(observations):
            obs = np.array(obs, dtype=np.float64)
            new_gmm_partial = Gmm()

            for j, comp in enumerate(self.gmm):
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
                    weight=self._detection * GmComponent(comp.weight, loc=nu[j], cov=s[j]).eval_at(obs) / (mean_pred * self._rho_fa),
                    loc=comp.loc + np.dot(k[j], y),
                    cov=p[j])
                )
            # end for

            # Add partial GMM to the GMM list
            mu_z[o] = new_gmm_partial.get_total_weight()
            new_gmm_per_measurement.append(new_gmm_partial)
        # end for

        # Missed detections
        ###################
        # Update missed detections weights (without l_1)
        self.gmm.mult_comp_weight((1 - self._detection) / mean_pred)

        # Compute l_i
        l1, l2, l1z, l2z, l2zz = self._compute_l_factors(mu_z, alpha_pred, beta_pred, n_meas, mean_pred)

        # Mean and variance
        ###################
        mu_phi = self.gmm.get_total_weight()

        if len(mu_z) == 0:
            sum_muz_l1z = 0
            sum_muz_l2z = 0
            sumsum = 0
        else:
            sum_muz_l1z = np.sum(mu_z * l1z)
            sum_muz_l2z = np.sum(mu_z * (l2z - l1 * l1z))
            _mu_z = mu_z.reshape((1, -1))
            _l1z = l1z.reshape((1, -1))
            sumsum = np.sum(np.sum((np.dot(_mu_z.T, _mu_z)) * (l2zz - np.dot(_l1z.T, _l1z))))
        # end if

        mean_update = mu_phi * l1 + sum_muz_l1z
        self._variance = mean_update + mu_phi ** 2 * (l2 - l1 ** 2) + 2 * mu_phi * sum_muz_l2z + sumsum

        assert self._variance >= 0, f"Variance ({self._variance}) < 0!"

        # Final update of the weights (var had to be computed first)
        ############################################################
        self.gmm.mult_comp_weight(l1)

        # Associations
        for jj in range(n_meas):
            new_gmm_per_measurement[jj].mult_comp_weight(l1z[jj])
        # end for

        # Built final GMM
        #################
        self.gmm += sum([new_gmm_per_measurement[i] for i in range(n_meas)])
    # end def

    def prune(self, trunc_thresh: float = 1e-6):
        # Prune components with negligible weight
        #########################################
        pruned_weight = .0
        for c in reversed(range(len(self.gmm))):
            if self.gmm[c].weight < trunc_thresh:
                pruned_weight += self.gmm[c].weight  # Save weight
                self.gmm.pop(c)
            # end if
        # end for

        # Distribute the pruned weight among the remaining components
        #############################################################
        if len(self.gmm) > 0 and pruned_weight > 0:
            self.gmm.add_comp_weight(pruned_weight / len(self.gmm))
        # end if
    # end def

    def merge(self, merge_dist_measure: DistMeasure, merge_thresh: float, max_components: int = 100):
        # Start with first GMM component
        ii = 0
        while ii < len(self.gmm):
            collect_gmm_comps = [self.gmm[ii]]

            # Add all nearby GMM components
            jj = ii + 1
            while jj < len(self.gmm):
                dist = self.gmm[ii].calc_dist_to(self.gmm[jj], merge_dist_measure)

                if dist < merge_thresh:
                    collect_gmm_comps.append(self.gmm.pop(jj))  # Move element from gmm to collect list
                # end if

                jj += 1  # Check next GMM component
            # end while

            # Subsume all collected GMM components
            if len(collect_gmm_comps) > 1:
                self.gmm[ii] = Gmm(collect_gmm_comps).get_unified_comp()
            # end if

            ii += 1  # Just take next GMM component
        # end while

        # Now ensure the number of components is within the limit, keeping the weightiest
        # Adapted by (ah): taken from PHD filter -> instead of calculating max. e.g. 5000 combinations like [is] does, all combinations are calculated and only the weightiest are kept
        if len(self.gmm) > max_components:
            weight_before = self.gmm.get_total_weight()
            self.gmm.sort(key=lambda comp: comp.weight, reverse=True)
            self.gmm = Gmm(self.gmm[:max_components])

            # Pruning should not alter the total weightsum (which relates to total num items) - so we renormalize
            weight_after = self.gmm.get_total_weight()
            self.gmm.mult_comp_weight(weight_before / weight_after)
        # end if
    # end def

    def _compute_l_factors(self, mu_z, alpha_val, beta_val, n_meas, mean_val):
        """Computes the additional factors used in the Panjer PHD.

        Args:
            mu_z: ?
            alpha_val: alpha_{k|k-1}.
            beta_val: beta_{k|k-1}.
            n_meas: Number of measurements.
            mean_val: ?

        Returns:
            l1: l_1(phi)
            l2: l_2(phi)
            l0z: l_0(z) (array for all z)
            l1z: l_1(z) (array for all z)
            l0zz: l_0(z,z') (matrix of all combinations z,z')
        """

        def eps():
            return np.finfo(np.float64).eps
        # end def

        # Check measurement number. If n  == 0 or n == 1 compute terms manually
        if n_meas == 0:
            if math.isinf(alpha_val):
                y0 = 1
                y1 = mean_val
                y2 = mean_val ** 2
                y1z = np.empty(0)
                y2z = np.empty(0)
                y2zz = np.empty(0)

            else:
                y0 = 1
                y1 = alpha_val / (beta_val + self._detection)
                y2 = alpha_val * (alpha_val + 1) / (beta_val + self._detection) ** 2
                y1z = np.empty(0)
                y2z = np.empty(0)
                y2zz = np.empty(0)
            # end if

        elif n_meas == 1:
            if math.isinf(alpha_val):
                y0 = 1 + mean_val * mu_z[0]
                y1 = mean_val + mean_val ** 2 * mu_z[0]
                y2 = mean_val ** 2 + mean_val ** 3 * mu_z[0]
                y1z = np.asarray([mean_val])
                y2z = np.asarray([mean_val ** 2])
                y2zz = 0

            else:
                y0 = 1 + alpha_val / (beta_val + self._detection) * mu_z[0]
                y1 = alpha_val / (beta_val + self._detection) + alpha_val * (alpha_val + 1) / (beta_val + self._detection) ** 2 * mu_z[0]
                y2 = alpha_val * (alpha_val + 1) / (beta_val + self._detection) ** 2 + alpha_val * (alpha_val + 1) * (alpha_val + 2) / (beta_val + self._detection) ** 3 * mu_z[0]
                y1z = np.asarray([alpha_val / (beta_val + self._detection)])
                y2z = np.asarray([alpha_val * (alpha_val + 1) / (beta_val + self._detection) ** 2])
                y2zz = 0
            # end if

        else:
            # Pre-calculate (alpha * (alpha - 1) * ... * (alpha + k - 1) / beta^k) for the Y_d terms - do that with the log-trick to avoid precision issues)
            log_prefactor = np.zeros(n_meas + 3)
            signs = np.zeros(len(log_prefactor))

            if math.isinf(alpha_val):
                for ii in range(1, len(log_prefactor)):  # Leave out the first step because prefactor[0] = 1 per definition
                    log_prefactor[ii] = log_prefactor[ii - 1] + np.log(mean_val + eps())
                # end for

            else:
                for ii in range(1, len(log_prefactor)):  # Leave out the first step because prefactor[0] = 1 per definition
                    signs[ii] = signs[ii - 1] * ((np.sign(alpha_val) * np.sign(beta_val)) == -1)
                    log_prefactor[ii] = log_prefactor[ii - 1] + np.log(np.abs(alpha_val + [ii - 1] - 1 + eps())) - np.log(abs(beta_val + self._detection + eps()))
                    assert np.isreal(log_prefactor[ii]), f"log_prefactor[{ii}] (= {log_prefactor[ii]}) is complex!"
                # end for
            # end if

            # # Vieta's theorem trick (leads to sum of all possible products of length k):
            log_pol = np.log(np.abs(np.poly(mu_z)) + eps())  # Equivalent to iterative convolution of (1 - mu[z_i])

            # Y_i[Z]
            ########
            y0tmp = np.exp((-1) ** signs[n_meas + 0] * log_prefactor[0:n_meas + 1] + log_pol)
            y1tmp = np.exp((-1) ** signs[n_meas + 1] * log_prefactor[1:n_meas + 2] + log_pol)
            y2tmp = np.exp((-1) ** signs[n_meas + 2] * log_prefactor[2:n_meas + 3] + log_pol)

            y0 = np.sum(y0tmp)
            y1 = np.sum(y1tmp)
            y2 = np.sum(y2tmp)

            assert not math.isinf(y0), "y0 is infinity!"
            assert not math.isinf(y1), "y1 is infinity!"
            assert not math.isinf(y2), "y2 is infinity!"

            # Y_i[Z_m \ {z}], Y_2[Z_m \ {z,z'}]
            ###################################
            # Pre-allocation:
            y1z = np.zeros(n_meas)
            y2z = np.zeros(n_meas)
            y2zz = np.zeros([n_meas, n_meas])  # l^{\neq}_0 for the variance (only non-diagonal entries)

            for ii in range(n_meas):  # n_meas is >= 2 here (since we checked 0 and 1 already above)!
                # Y_1[Z_m \ {z}], Y_2[Z_m \ {z}]
                ################################
                ind = np.array(range(n_meas))
                ind = np.delete(ind, ii)  # Take all except that particular z

                # Vieta's theorem trick again:
                log_pol = np.log(np.abs(np.poly(mu_z[ind])) + eps())

                y1z_tmp = np.exp((-1) ** signs[n_meas + 0] * log_prefactor[1:n_meas + 1] + log_pol)
                y2z_tmp = np.exp((-1) ** signs[n_meas + 1] * log_prefactor[2:n_meas + 2] + log_pol)

                y1z[ii] = np.sum(y1z_tmp)
                y2z[ii] = np.sum(y2z_tmp)

                assert not math.isinf(y1z[ii]), f"y1z[{ii}] is infinity!"
                assert not math.isinf(y2z[ii]), f"y2z[{ii}] is infinity!"

                # Y_2[z, z']
                ############
                for jj in range(n_meas):
                    if ii != jj:
                        ind = np.array(range(n_meas))
                        ind = np.delete(ind, [ii, jj])  # Take all except those particular z, z'

                        # Vieta's theorem trick again:
                        log_pol = np.log(abs(np.poly(mu_z[ind])) + eps())

                        y2zz_tmp = np.exp((-1) ** signs[n_meas + 0] * log_prefactor[2:n_meas + 1] + log_pol)
                        y2zz[ii, jj] = np.sum(y2zz_tmp)

                        assert not math.isinf(y2zz[ii, jj]), f"y2zz[{ii}, {jj}] is infinity!"
                    # end if
                # end for
            # end for
        # end if

        l1 = y1 / y0      # l_1(phi)          --> for mean
        l2 = y2 / y0      # l_2(phi)          --> for variance

        l1z = y1z / y0    # l_1(z)            --> for the variance
        l2z = y2z / y0    # l_0(z)            --> for the mean
        l2zz = y2zz / y0  # l_1^{\neq}(z,z')  --> for the variance

        assert np.all(np.isreal([l1, l1z, l2, l2z, l2zz.T])), f"One or more of [l1, l1z, l2, l2z, l2zz.T] are complex!"

        return l1, l2, l1z, l2z, l2zz
    # end def

    # XXX This and the two more specific extract_states-functions are taken 1:1 from the PHD-Filter paper - check again:
    # * Does it make for the PDH filter?
    # * In case it is the same for both filter, is it possible to keep just in one place to avoid duplicates?
    # * Does it make sense to take these both function into GMM, or is it to specific, as each filter has its own logic? -> Check?
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
        items = []

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
    # end def

    def _extract_states_using_integral(self, bias: float = 1.0):
        """Extract states based on the expected number of states from the integral of the intensity.
        This is NOT in the GMPHD paper; added by Dan.
        "bias" is a multiplier for the est number of items.
        """
        num_to_add = int(round(bias * self.gmm.get_total_weight()))
        self._logging.print_verbose(Logging.DEBUG, "bias is %g, num_to_add is %i" % (bias, num_to_add))

        # A temporary list of peaks p will gradually be decimated as we steal from its highest peaks
        peaks = [GmComponent(comp.weight, comp.loc, None) for comp in self.gmm]

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


def main():
    f = np.array([[1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    q = np.array([[0.25, 0.00, 0.50, 0.00],
                  [0.00, 0.25, 0.00, 0.50],
                  [0.50, 0.00, 1.00, 0.00],
                  [0.00, 0.50, 0.00, 1.00]])

    h = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])

    r = np.array([[1, 0],
                  [0, 1]])
    birth_comp = GmComponent(weight=1,
                             loc=np.array([25, 25, 0, 0]),
                             cov=np.array([[625, 0, 0, 0],
                                           [0, 625, 0, 0],
                                           [0, 0, 0.09, 0],
                                           [0, 0, 0, 0.09]])
                             )

    z = list()
    z.append([38.295392824015780, 4.037063438243743])
    z.append([25.920899393647160, 36.922014809948500])
    z.append([14.840025078811097, 22.065461144797656])
    z.append([9.386061433062581, 7.915493385632560])

    f = GmPanjerPhdFilter(birth_gmm=[birth_comp],
                          var_birth=1., survival=.99, detection=.9, f=f, q=q, h=h, r=r,
                          rho_fa=0.0012, gate_thresh=9.2103, logging=Logging.INFO)

    for i in range(50):
        f.predict()
        f.update(observations=z)
        f.prune(trunc_thresh=1.e-10)
        f.merge(merge_dist_measure=DistMeasure.HELLINGER, merge_thresh=0.8, max_components=100)
# end def


# Execute only if run as a script
if __name__ == "__main__":
    main()
# end if
