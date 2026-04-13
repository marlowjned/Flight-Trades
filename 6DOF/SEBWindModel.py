# SEBWindModel.py
# ERA5 EOF ensemble + Von Kármán turbulence wind model adapter for the 6DOF sim.
# Wraps the wind_model package into the WindModel interface expected by SimulationLoop.

import sys
import os
import numpy as np
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "wind_model"))

from wind_model.eof import EOFModel
from wind_model.von_karman import VonKarmanFilter
from WindModel import WindModel
from Vector3D import Vector3D


class SEBWindModel(WindModel):
    """
    Altitude-varying mean wind drawn from an EOF ensemble, with Von Kármán
    turbulence layered on top. Compatible with the 6DOF WindModel interface.

    The Von Kármán filter advances exactly once per simulation timestep via
    advance(t, z, V), which SimulationLoop calls before each RK4 step.
    windVector(altitude) uses the frozen turbulence increment from the last
    advance() call, so the filter state is consistent across all RK4 stages.
    """

    def __init__(self, eof_model: EOFModel, dt: float, seed: int, scale: float = 1.0):
        """
        Parameters
        ----------
        eof_model : EOFModel
            Fitted EOF model from one month of ERA5 data.
        dt : float
            Simulation timestep (s) — must match SimulationLoop.
        seed : int
            Random seed for reproducibility.
        scale : float
            Perturbation scale on the EOF ensemble draw (default 1.0).
        """
        rng = np.random.default_rng(seed)

        # Draw one mean wind profile from the EOF ensemble
        u_base, v_base = eof_model.sample(n_draws=1, rng=rng, scale=scale)
        self._u_interp = PchipInterpolator(eof_model.alt_grid, u_base[0], extrapolate=True)
        self._v_interp = PchipInterpolator(eof_model.alt_grid, v_base[0], extrapolate=True)

        # Von Kármán filter — initialised at ground level, updated each advance() call
        self._vk = VonKarmanFilter(dt=dt, airspeed=1.0, z_m=0.0)
        self._rng = rng

        # Cached turbulence increments (frozen across RK4 stages)
        self._du: float = 0.0
        self._dv: float = 0.0
        self._dw: float = 0.0

    def advance(self, t: float, z: float, V: float):
        """
        Advance the Von Kármán filter by one timestep and cache the result.
        Must be called exactly once per simulation timestep, before the RK4 step.

        Parameters
        ----------
        t : float  Simulation time (s)
        z : float  Altitude AGL (m)
        V : float  Rocket airspeed (m/s)
        """
        self._vk.update_altitude(z, V)
        self._du, self._dv, self._dw = self._vk.step(self._rng)

    def windVector(self, altitude: float) -> Vector3D:
        """
        Return the wind vector at the given altitude.
        Base wind varies with altitude; turbulence is frozen from the last advance().
        """
        u = float(self._u_interp(altitude)) + self._du
        v = float(self._v_interp(altitude)) + self._dv
        w = self._dw
        return Vector3D(np.array([u, v, w]))
