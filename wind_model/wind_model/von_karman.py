# von_karman.py
# Von Kármán turbulence generator using Hoblit rational-approximation shaping filters
# Reference: MIL-HDBK-1797B / Hoblit (1988)

import numpy as np
from scipy.signal import bilinear_zpk, zpk2sos, sosfilt


# MIL-HDBK-1797B altitude-band turbulence intensity table
# Columns: [alt_lower_m, alt_upper_m, sigma_ms]
MILSPEC_SIGMA_TABLE = np.array([
    [0,      305,   1.0],
    [305,    610,   1.28],
    [610,    3048,  1.68],
    [3048,   6096,  1.83],
    [6096,   12192, 1.37],
    [12192,  30000, 0.91],
])

# Above 305 m, free-atmosphere scale length (m)
L_FREE_ATM = 533.0

# Minimum airspeed and scale length to prevent division-by-zero
_V_MIN = 0.1    # m/s
_L_MIN = 1.0    # m


def sigma_u(z_m: float) -> float:
    """
    Turbulence intensity σ_u (m/s) at altitude z_m (metres AGL).

    Below 305 m uses the MIL-HDBK-1797B surface-layer formula, scaled so
    that σ = 1.0 m/s at z = 305 m (matching the table boundary).
    Above 305 m uses the MILSPEC altitude-band table.
    """
    if z_m < 305.0:
        z_ft = z_m * 3.28084
        # Formula returns 1.0 at z_ft ≈ 1000 ft (z_m ≈ 305 m) naturally
        return 1.0 / (0.177 + 0.000823 * z_ft) ** 0.4
    else:
        # Find which band z_m falls in; use upper boundary convention
        upper_bounds = MILSPEC_SIGMA_TABLE[:, 1]
        idx = int(np.searchsorted(upper_bounds, z_m, side="left"))
        idx = min(idx, len(MILSPEC_SIGMA_TABLE) - 1)
        return float(MILSPEC_SIGMA_TABLE[idx, 2])


def scale_length(z_m: float) -> float:
    """
    Turbulence scale length L_u (m) at altitude z_m (metres AGL).

    Below 305 m uses the MIL-HDBK-1797B surface-layer formula.
    Above 305 m returns L_FREE_ATM = 533 m.
    Minimum return value is 1.0 m.
    """
    if z_m < 305.0:
        z_ft = z_m * 3.28084
        if z_ft < 1e-6:
            return _L_MIN
        L_ft = z_ft / (0.177 + 0.000823 * z_ft) ** 1.2
        return max(L_ft / 3.28084, _L_MIN)
    else:
        return L_FREE_ATM


def _analog_zpk_u(sigma: float, L: float, V: float):
    """
    Analog ZPK coefficients for the longitudinal (u) Hoblit filter.

    H_u(s) = sigma * sqrt(2L/(πV)) * (1 + sqrt(3)*(L/V)*s) / (1 + L/V*s)²
    """
    z = [-V / (np.sqrt(3.0) * L)]          # single zero
    p = [-V / L, -V / L]                   # double pole
    k = sigma * np.sqrt(2.0 * L / (np.pi * V)) * np.sqrt(3.0) * V / L
    return z, p, k


def _analog_zpk_v(sigma: float, L: float, V: float):
    """
    Analog ZPK coefficients for the lateral/vertical (v, w) Hoblit filter.

    H_v(s) = sigma * sqrt(L/(πV)) * (1 + 2.678*sqrt(8/3)*L/(3V)*s)
                                    / (1 + 2.678*L/V*s)²
    """
    c = 2.678
    numer_s_coeff = c * np.sqrt(8.0 / 3.0) * L / (3.0 * V)
    denom_s_coeff = c * L / V

    z = [-1.0 / numer_s_coeff]             # single zero
    p = [-1.0 / denom_s_coeff, -1.0 / denom_s_coeff]  # double pole
    k = sigma * np.sqrt(L / (np.pi * V)) * numer_s_coeff / denom_s_coeff ** 2
    return z, p, k


def _prewarped_sos(z_analog, p_analog, k_analog, dt: float, omega_c: float):
    """
    Discretise an analog ZPK filter using the bilinear transform with
    pre-warping at corner frequency omega_c (rad/s).

    Pre-warping scales the analog prototype so that the digital filter
    matches the analog filter exactly at omega_c.
    """
    fs = 1.0 / dt
    # Pre-warp: find the analog frequency that maps to omega_c after bilinear
    omega_prewarp = 2.0 * fs * np.tan(omega_c / (2.0 * fs))
    scale = omega_prewarp / omega_c  # > 1 for typical dt/omega_c combos

    z_scaled = [zi * scale for zi in z_analog]
    p_scaled = [pi * scale for pi in p_analog]
    # Relative degree = len(p) - len(z); gain scales as scale^(nz-np)
    rel_deg = len(p_analog) - len(z_analog)
    k_scaled = k_analog / (scale ** rel_deg)

    z_d, p_d, k_d = bilinear_zpk(z_scaled, p_scaled, k_scaled, fs=fs)
    return zpk2sos(z_d, p_d, k_d)


class VonKarmanFilter:
    """
    Hoblit rational-approximation shaping filters for Von Kármán turbulence.
    Generates correlated (u_t, v_t, w_t) turbulence increments at each timestep.
    """

    def __init__(self, dt: float, airspeed: float, z_m: float):
        """
        Parameters
        ----------
        dt : float
            Simulation timestep (s).
        airspeed : float
            Rocket airspeed (m/s).
        z_m : float
            Altitude AGL (m).
        """
        self.dt = dt
        self._sigma = None  # set by _build_filters
        self._build_filters(z_m, max(airspeed, _V_MIN))

    def _build_filters(self, z_m: float, V: float):
        sigma = sigma_u(z_m)
        L = scale_length(z_m)
        V = max(V, _V_MIN)
        L = max(L, _L_MIN)

        omega_c = V / L  # corner frequency (rad/s)

        z_u, p_u, k_u = _analog_zpk_u(sigma, L, V)
        z_v, p_v, k_v = _analog_zpk_v(sigma, L, V)

        self.sos_u = _prewarped_sos(z_u, p_u, k_u, self.dt, omega_c)
        self.sos_v = _prewarped_sos(z_v, p_v, k_v, self.dt, omega_c)

        self.zi_u = np.zeros((self.sos_u.shape[0], 2))
        self.zi_v = np.zeros((self.sos_v.shape[0], 2))
        self.zi_w = np.zeros((self.sos_v.shape[0], 2))

        self._sigma = sigma

    def step(self, rng: np.random.Generator):
        """
        Advance the filter by one timestep.

        Returns
        -------
        (u_t, v_t, w_t) : tuple of float
            Turbulence velocity components (m/s).
        """
        dt = self.dt
        eta_u = rng.standard_normal() / np.sqrt(dt)
        eta_v = rng.standard_normal() / np.sqrt(dt)
        eta_w = rng.standard_normal() / np.sqrt(dt)

        u_arr, self.zi_u = sosfilt(self.sos_u, [eta_u], zi=self.zi_u)
        v_arr, self.zi_v = sosfilt(self.sos_v, [eta_v], zi=self.zi_v)
        w_arr, self.zi_w = sosfilt(self.sos_v, [eta_w], zi=self.zi_w)

        return float(u_arr[0]), float(v_arr[0]), float(w_arr[0])

    def update_altitude(self, z_m: float, airspeed: float):
        """
        Recompute filter coefficients for a new altitude and airspeed.
        The filter state is rescaled proportionally to the new sigma to
        maintain continuity across altitude transitions.
        """
        old_sigma = self._sigma
        V = max(airspeed, _V_MIN)
        new_sigma = sigma_u(z_m)

        self._build_filters(z_m, V)

        if old_sigma is not None and old_sigma > 0:
            ratio = new_sigma / old_sigma
            self.zi_u *= ratio
            self.zi_v *= ratio
            self.zi_w *= ratio
