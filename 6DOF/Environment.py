# Environment.py
# Includes all wind, gust, and other environmental data

from __future__ import annotations
from abc import ABC, abstractmethod

from ambiance import Atmosphere
from typing import Union, Optional
import numpy as np

import Vector3D

# Base model for environment, then make another file for our current best wind model
class Environment:
    
    # make env data class
    # insert your altitude one time, and access all data you could ever need!
    
    # take in randomization seeds on instantiation
    def __init__(self):
        return
    

    class WindModel(ABC):

        @abstractmethod
        def windVector(self, altitude: float) -> Vector3D: ...
        # add any other necessary functions



    class SimpleWindModel:
        pass

    # Type alias for inputs that can be scalar or array
    Scalar_or_Array = Union[float, int, list, np.ndarray]

    @property
    def a(self): pass
    @property
    def pressure(self): pass