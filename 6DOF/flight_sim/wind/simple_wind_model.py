# simple_wind_model.py
# Basic wind model that adds turbulence to an otherwise constant wind vector

import numpy as np

from flight_sim.wind.wind_model_base import WindModelBase
from flight_sim.data_helpers.vector3d import Vector3D
from flight_sim.data_helpers.custom_interpolator import Interpolator1D


class SimpleWindModel(WindModelBase):

    TAU = 2000  # Decay scale length (m)

    def __init__(self,
                 magnitude: float,             # m/s
                 direction: float,             # rad (clockwise from north)
                 altitudes: np.ndarray,        # m
                 turbulence_seed: int = None,
                 turbulence_intensity: float = 1.0):

        wind_north = np.array([magnitude * np.cos(direction) for a in altitudes])
        wind_east  = np.array([magnitude * np.sin(direction) for a in altitudes])

        _seed = turbulence_seed if turbulence_seed is not None else np.random.randint(0, 99999)
        self.seed = _seed
        rng = np.random.default_rng(seed=_seed)
        n_samples = np.array(rng.standard_normal(len(altitudes))) * turbulence_intensity
        e_samples = np.array(rng.standard_normal(len(altitudes))) * turbulence_intensity

        LASTVAL = Interpolator1D.BoundaryBehavior.LASTVAL
        self.wind_north = Interpolator1D(altitudes, self.apply_turbulence(altitudes, wind_north, n_samples), LASTVAL)
        self.wind_east  = Interpolator1D(altitudes, self.apply_turbulence(altitudes, wind_east,  e_samples), LASTVAL)

    def apply_turbulence(self, altitudes: np.ndarray, wind: np.ndarray, rng_samples: np.ndarray):
        final_wind = np.zeros(len(altitudes))
        final_wind[0] = wind[0] + rng_samples[0]
        for i in range(1, len(wind)):
            dz = altitudes[i] - altitudes[i - 1]
            final_wind[i] = (np.exp(-dz / self.TAU) * final_wind[i - 1]
                             + np.sqrt(1 - np.exp(-2 * dz / self.TAU)) * rng_samples[i])
        return final_wind

    def wind_vector(self, altitude: float) -> Vector3D:
        return Vector3D([self.wind_east.query(altitude), self.wind_north.query(altitude), 0])
