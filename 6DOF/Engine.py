# Engine.py
# Direct and RPA implementations

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from dataclasses import dataclass

import CustomInterpolator
import Vector3D

if TYPE_CHECKING:
    import Rocket

# TEMPORARY, rpa implementations will flesh out this class significantly
class Engine:
    def __init__(self, 
                 massData: Rocket.Rocket.MassComponent,
                 thrust: CustomInterpolator.Interpolator1D):
        self.massData = massData
        self.thrust = thrust

    def thrustVector(self, time: float, DCM: np.ndarray):
        currentThrust = self.thrust.query(time)
        return Vector3D([0, 0, currentThrust], DCM, True)
    
    # engine should probably own the property "thrusting"

    