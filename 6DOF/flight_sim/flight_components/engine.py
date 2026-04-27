# engine.py
# Direct and RPA implementations

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from flight_sim.data_helpers.custom_interpolator import Interpolator1D
from flight_sim.data_helpers.vector3d import Vector3D

if TYPE_CHECKING:
    import flight_sim.flight_components.rocket as rocket_module


# TEMPORARY, rpa implementations will flesh out this class significantly
class Engine:
    def __init__(self,
                 mass_data: rocket_module.Rocket.MassComponent,
                 thrust: Interpolator1D,
                 thrust_scale: float = 1.0):
        self.mass_data   = mass_data
        self.thrust      = thrust
        self.thrust_scale = thrust_scale

    def thrust_vector(self, time: float, dcm: np.ndarray):
        current_thrust = self.thrust.query(time) * self.thrust_scale
        return Vector3D([0, 0, current_thrust], dcm, True)

    # engine should probably own the property "thrusting"
