# sim_conditions.py
# Computed once per timestep from FlightState + WindModelBase.
# Acts as the bridge between the physical world and aerodynamic calculations,
# so that q, Mach, alpha, and airflow are never recomputed in multiple places.

'''
In the future, it may be worthwhile to add the atmospheric model as a separate
module, passed to construct SimConditions. However, this should be good enough
for all we're trying to do as of S26.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from ambiance import Atmosphere

from flight_sim.data_helpers.vector3d import Vector3D

if TYPE_CHECKING:
    import flight_sim.core.sim_loop as sim_loop_module
    import flight_sim.wind.wind_model_base as wind_module


@dataclass
class SimConditions:
    rho:     float      # Air density (kg/m^3)
    a:       float      # Speed of sound (m/s)
    mach:    float      # Mach number
    alpha:   float      # Angle of attack (rad)
    q:       float      # Dynamic pressure (Pa)
    airflow: Vector3D   # Effective airflow vector (m/s), body-frame DCM attached

    @classmethod
    def compute(
        cls,
        state: sim_loop_module.FlightSim.FlightState,
        wind:  wind_module.WindModelBase,
    ) -> SimConditions:

        atm = Atmosphere(max(state.position.z, 0.0))
        rho = float(atm.density[0])
        a   = float(atm.speed_of_sound[0])

        u, v, w = wind.wind_vector(state.position.z).vector_world
        wind_world = np.array([u, v, w])
        airflow = Vector3D(wind_world - state.velocity.vector_world,
                           state.orientation.as_matrix())

        v_mag = np.linalg.norm(state.velocity.elements)
        q     = 0.5 * rho * v_mag ** 2
        mach  = v_mag / a

        body_z_world  = state.orientation.apply([0, 0, 1])
        airflow_world = airflow.vector_world
        airflow_norm  = np.linalg.norm(airflow_world)
        if airflow_norm > 0:
            alpha = np.arccos(
                np.clip(np.dot(body_z_world, -airflow_world) / airflow_norm, -1.0, 1.0))
        else:
            alpha = 0.0

        return cls(rho=rho, a=a, mach=mach, alpha=alpha, q=q, airflow=airflow)
