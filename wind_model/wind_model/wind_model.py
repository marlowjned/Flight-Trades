# wind_model.py
# Top-level WindModel class: draws EOF ensemble members and layers Von Kármán turbulence

import numpy as np
from scipy.interpolate import PchipInterpolator

from .eof import EOFModel
from .von_karman import VonKarmanFilter


class WindModel:
    """
    Top-level interface for Monte Carlo wind realisations.

    Combines an EOF-based mean wind ensemble with altitude-varying Von Kármán
    turbulence. Each realisation is a callable that can be consumed by a 6DOF
    flight simulator.

    Usage
    -----
    eof = EOFModel(u_profiles, v_profiles, alt_grid)
    model = WindModel(eof, dt=0.05)
    wind_fn = model.realisation(seed=42, airspeed_profile=None)

    # At each simulation timestep (call exactly once per dt):
    u, v, w = wind_fn(t=1.0, z=500.0, V=150.0)
    """

    def __init__(self, eof_model: EOFModel, dt: float, scale: float = 1.0):
        """
        Parameters
        ----------
        eof_model : EOFModel
            Fitted EOF model from one month of ERA5 data.
        dt : float
            Simulation timestep (s). Must match the 6DOF integrator.
        scale : float
            Perturbation scale factor (default 1.0). Set < 1 for validation
            mode where ERA5 representativeness error is known.
        """
        self.eof_model = eof_model
        self.dt = dt
        self.scale = scale

    def realisation(self, seed: int, airspeed_profile=None):
        """
        Generate a single Monte Carlo wind realisation.

        Parameters
        ----------
        seed : int
            Random seed for reproducibility.
        airspeed_profile : callable, optional
            Unused at call time (V is supplied per-call). Reserved for future
            pre-computation.

        Returns
        -------
        wind : callable
            Signature: wind(t, z, V) -> (u_wind, v_wind, w_wind) in m/s.
            Must be called exactly once per simulation timestep in order.
            Also exposes wind.base_wind(z) -> (u_base, v_base) for diagnostics.
        """
        rng = np.random.default_rng(seed)

        # Draw one mean wind profile from the EOF ensemble
        u_base_arr, v_base_arr = self.eof_model.sample(n_draws=1, rng=rng, scale=self.scale)
        u_base_arr = u_base_arr[0]  # (K,)
        v_base_arr = v_base_arr[0]

        u_interp = PchipInterpolator(self.eof_model.alt_grid, u_base_arr, extrapolate=True)
        v_interp = PchipInterpolator(self.eof_model.alt_grid, v_base_arr, extrapolate=True)

        # Initialise Von Kármán filter at ground level
        vk_filter = VonKarmanFilter(dt=self.dt, airspeed=1.0, z_m=0.0)

        def wind(t: float, z: float, V: float):
            """
            Return (u_wind, v_wind, w_wind) at time t (s), altitude z (m AGL),
            rocket airspeed V (m/s). Advances the turbulence filter by one dt.
            """
            vk_filter.update_altitude(z, V)
            du, dv, dw = vk_filter.step(rng)

            u_total = float(u_interp(z)) + du
            v_total = float(v_interp(z)) + dv
            w_total = dw

            return u_total, v_total, w_total

        def base_wind(z: float):
            """Return the (u_base, v_base) mean profile at altitude z, without turbulence."""
            return float(u_interp(z)), float(v_interp(z))

        wind.base_wind = base_wind
        return wind

    def ensemble(self, n: int, airspeed_profile=None):
        """
        Return a list of n independent wind realisations with seeds 0..n-1.

        Each element is an independent closure with its own rng and filter state.
        """
        return [self.realisation(seed=i, airspeed_profile=airspeed_profile) for i in range(n)]
