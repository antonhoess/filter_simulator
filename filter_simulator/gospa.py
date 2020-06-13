# From https://github.com/ewilthil/gospapy

from __future__ import division
from __future__ import annotations
from typing import Optional, Tuple, Sequence, Callable, TypeVar, Generic, Dict, List
import numpy as np
from scipy.optimize import linear_sum_assignment


POINT = TypeVar('POINT')


class GospaResult:
    def __init__(self, gospa, target_to_track_assignments, gospa_localization, gospa_missed, gospa_false):
        self.gospa = gospa
        self.target_to_track_assignments = target_to_track_assignments
        self.gospa_localization = gospa_localization
        self.gospa_missed = gospa_missed
        self.gospa_false = gospa_false
    # end def
# end class


class Gospa:
    @staticmethod
    def _check_params(c: float, p: float, alpha: float):
        if alpha <= 0 or alpha > 2:
            raise ValueError("The value of alpha is outside the range (0, 2]")
        if c <= 0:
            raise ValueError("The cutoff distance c is outside the range (0, inf)")
        if p < 1:
            raise ValueError("The order p is outside the range [1, inf)")
    # end def

    @staticmethod
    def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
        return np.linalg.norm(x - y)
    # end def

    @staticmethod
    def calc(targets: Sequence[POINT], tracks: Sequence[POINT], c: float, p: float, alpha: Optional[float] = 2,
             assignment_cost_function: Callable[[POINT, POINT], float] = None) -> GospaResult:
        """ GOSPA metric for multitarget tracking filters

        Provide a detailed description of the method.

        Parameters
        ----------
        targets : iterable of elements
            Contains the elements of the first set.
        tracks : iterable of elements
            Contains the elements of the second set.
        c : float
            Defines the cost of a missed target or a false track, equal to c/2.
        p : float
            Briefly describe the parameter.
        alpha : float, optional
            Briefly describe the parameter.
        assignment_cost_function : function, optional
            Briefly describe the parameter.
            Default: Gospa.euclidean_distance

        Returns
        -------
        gospa : float
            Total gospa.
        assignment : dictionary
            Contains the assignments on the form {target_idx : track_idx}.
        gospa_localization : float
            Localization error contribution.
        gospa_missed : float
            Number of missed target contribution.
        gospa_false : float
            Number of false tracks contribution.

        References
        ----------

        - A. S. Rahmathullah, A. F. Garcia-Fernandez and L. Svensson, Generalized
          optimal sub-pattern assignment metric, 20th International Conference on
          Information Fusion, 2017.
        - L. Svensson, Generalized optimal sub-pattern assignment metric (GOSPA),
          presentation, 2017. Available online: https://youtu.be/M79GTTytvCM
        """
        if assignment_cost_function is None:
            assignment_cost_function = Gospa.euclidean_distance

        Gospa._check_params(c, p, alpha)
        num_targets = len(targets)
        num_tracks = len(tracks)
        mf_cost = c**p / alpha  # missed / false cost

        # Initial values
        target_to_track_assignments = dict()
        gospa_localization = 0

        if num_targets == 0:  # All the tracks are false tracks
            pass  # Will be calculated below implicitly

        elif num_tracks == 0:  # All the targets are missed
            pass  # Will be calculated below implicitly

        else:  # There are elements in both sets. Compute cost matrix
            # Create empty cost matrix
            cost_matrix = np.zeros((num_targets, num_tracks))

            # Fill cost matrix
            for n_target in range(num_targets):
                for n_track in range(num_tracks):
                    # Calculate cost function for current pair of elements from target and track
                    current_cost = assignment_cost_function(targets[n_target], tracks[n_track]) ** p

                    # Apply the limit to the previously calculated cost
                    cost_matrix[n_target, n_track] = np.min([current_cost, alpha * mf_cost])
                # end for
            # end for

            # Calculate the row- and column indices of the optimal assignment
            target_assignment, track_assignment = linear_sum_assignment(cost_matrix)

            for target_idx, track_idx in zip(target_assignment, track_assignment):
                if cost_matrix[target_idx, track_idx] < alpha * mf_cost:
                    gospa_localization += cost_matrix[target_idx, track_idx]
                    # Add dictionary entry with target/track assignment
                    target_to_track_assignments[target_idx] = track_idx
                # end if
            # end for
        # end if

        # Calculate output values and thereby apply things like error term factors
        num_assignments = len(target_to_track_assignments)
        num_missed = num_targets - num_assignments
        num_false = num_tracks - num_assignments
        gospa_missed = mf_cost * num_missed
        gospa_false = mf_cost * num_false

        # Calculate the p-th root of the sum of all error terms to get the final GOSPA value
        gospa = (gospa_localization + gospa_missed + gospa_false) ** (1 / p)

        return GospaResult(gospa, target_to_track_assignments, float(gospa_localization), gospa_missed, gospa_false)
    # end def
# end class
