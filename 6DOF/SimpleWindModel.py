# SimpleWindModel.py
# Basic wind model that adds turbulence to an otherwise constant wind vector

import numpy as np

from WindModel import WindModel
from Vector3D import Vector3D
import CustomInterpolator

class SimpleWindModel(WindModel):

    TAU = 2000 # DECAY SCALE LENGTH, m

    def __init__(self,
                 magnitude: float,        # m/s
                 direction: float,        # rad (clockwise from north)
                 altitudes: np.ndarray,   # m
                 turbulence_seed: int = None,      # traceable turbulence generation
                 turbulence_intensity: float = 1.0 # scales noise std (0 = no turbulence)
                 ):

        windNorth = np.array([magnitude * np.cos(direction) for a in altitudes])
        windEast  = np.array([magnitude * np.sin(direction) for a in altitudes])

        _seed = turbulence_seed if turbulence_seed is not None else np.random.randint(0, 99999)
        self.seed = _seed
        rng = np.random.default_rng(seed=_seed)
        nsamples = np.array(rng.standard_normal(len(altitudes))) * turbulence_intensity
        esamples = np.array(rng.standard_normal(len(altitudes))) * turbulence_intensity

        LASTVAL = CustomInterpolator.Interpolator1D.BoundaryBehavior.LASTVAL
        self.windNorth = CustomInterpolator.Interpolator1D(altitudes, self.applyTurbulence(altitudes, windNorth, nsamples), LASTVAL)
        self.windEast  = CustomInterpolator.Interpolator1D(altitudes, self.applyTurbulence(altitudes, windEast,  esamples), LASTVAL)

    def applyTurbulence(self, altitudes: np.ndarray, wind: np.ndarray, rng_samples: np.ndarray):
        final_wind = np.zeros(len(altitudes))
        final_wind[0] = wind[0] + rng_samples[0]
        for i in range(1, len(wind)):
            dz = altitudes[i] - altitudes[i-1]
            final_wind[i] = np.exp(-dz/self.TAU) * final_wind[i-1] + np.sqrt(1 - np.exp(-2*dz/self.TAU)) * rng_samples[i]
        return final_wind
    
    def windVector(self, altitude: float):
        return Vector3D([self.windEast.query(altitude), self.windNorth.query(altitude), 0])

