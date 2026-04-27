# utils.py
# Shared helpers: altitude grids, unit conversions, wind vector utilities

import numpy as np


def uniform_alt_grid(z_min: float = 0.0, z_max: float = 20000.0, dz: float = 50.0) -> np.ndarray:
    """Return a uniform altitude grid from z_min to z_max (inclusive) with step dz (metres)."""
    return np.arange(z_min, z_max + dz, dz)


def wind_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Element-wise horizontal wind speed (m/s)."""
    return np.sqrt(u ** 2 + v ** 2)


def wind_direction_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Meteorological wind direction in degrees (direction wind is coming FROM).
    0° = wind from North, 90° = wind from East.
    """
    return (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
