# WindModel.py
# Base wind model template to be inherited by wind models used in simulations

'''
Many wind models can be precomputed, altitude by altitude,
If for whatever reason your wind model includes time dependency
(i.e. time dependent white noise) you must (or at least should) 
design your wind model such that it can store the wind values 
throughtout the flight so they can be later referenced for data 
collection.
'''

from __future__ import annotations
from abc import ABC, abstractmethod

import Vector3D

class WindModel(ABC):

        @abstractmethod
        def windVector(self, altitude: float) -> Vector3D: ...
        # TODO: add any other necessary functions