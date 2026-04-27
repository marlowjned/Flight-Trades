# seb_wind_model.py
# ERA5 EOF ensemble + Von Karman turbulence wind model adapter for the 6DOF sim.
# Wraps the seb_wind_model package into the WindModelBase interface.

import sys
import os
import numpy as np
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'SEB-windmodel'))

from seb_wind_model.eof import EOFModel
from seb_wind_model.von_karman import VonKarmanFilter
from flight_sim.wind.wind_model_base import WindModelBase
from flight_sim.data_helpers.vector3d import Vector3D


class SEBWindModel(WindModelBase):
    """
    Altitude-varying mean wind drawn from an EOF ensemble, with Von Karman
    turbulence layered on top. Compatible with the 6DOF WindModelBase interface.

    The Von Karman filter advances exactly once per simulation timestep via
    advance(t, z, V), which SimulationLoop calls before each RK4 step.
    wind_vector(altitude) uses the frozen turbulence increment from the last
    advance() call, so the filter state is consistent across all RK4 stages.
    """

    def __init__(self, eof_model: EOFModel, dt: float, seed: int, scale: float = 1.0):
        rng = np.random.default_rng(seed)

        u_base, v_base = eof_model.sample(n_draws=1, rng=rng, scale=scale)
        self._u_interp = PchipInterpolator(eof_model.alt_grid, u_base[0], extrapolate=True)
        self._v_interp = PchipInterpolator(eof_model.alt_grid, v_base[0], extrapolate=True)

        self._vk  = VonKarmanFilter(dt=dt, airspeed=1.0, z_m=0.0)
        self._rng = rng

        self._du: float = 0.0
        self._dv: float = 0.0
        self._dw: float = 0.0

    def advance(self, t: float, z: float, V: float):
        """
        Advance the Von Karman filter by one timestep and cache the result.
        Must be called exactly once per simulation timestep, before the RK4 step.
        """
        self._vk.update_altitude(z, V)
        self._du, self._dv, self._dw = self._vk.step(self._rng)

    def wind_vector(self, altitude: float) -> Vector3D:
        u = float(self._u_interp(altitude)) + self._du
        v = float(self._v_interp(altitude)) + self._dv
        w = self._dw
        return Vector3D(np.array([u, v, w]))
