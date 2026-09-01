# thrust_curve.py
# Synthesizes the engine thrust profile given nominal thrust and burn time.
# Called by SimulationHandler when building a rocket for each trade permutation.
#
# Replace the body of thrust_curve() with your actual curve equations.

import numpy as np


def thrust_curve(T, burn_time):
    """
    Returns (t_array, F_array) defining the thrust profile.

    Parameters
    ----------
    T         : nominal thrust (N)
    burn_time : total burn duration (s)

    Returns
    -------
    t : 1-D array of time points (s), strictly increasing, starting at 0
    F : 1-D array of thrust values (N) at each time point
    """
    t = np.array([0.0, burn_time])
    F = np.array([T,   T        ])  # TODO: REPLACE THIS — flat-top placeholder

    return t, F
