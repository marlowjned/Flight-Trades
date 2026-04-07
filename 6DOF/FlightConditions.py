# FlightConditions.py
# Computed once per timestep from FlightState + Environment.
# Acts as the bridge between the physical world and aerodynamic calculations,
# so that q, Mach, alpha, and airflow are never recomputed in multiple places.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from ambiance import Atmosphere

import Vector3D

if TYPE_CHECKING:
    import SimulationLoop
    import Environment


@dataclass
class FlightConditions:
    rho:     float          # air density (kg/m³)
    a:       float          # speed of sound (m/s)
    mach:    float          # Mach number
    alpha:   float          # angle of attack (rad)
    q:       float          # dynamic pressure (Pa)
    airflow: Vector3D.Vector3D  # effective airflow vector in world frame (wind - velocity)

    @classmethod
    def compute(
        cls,
        state:  SimulationLoop.FlightSim.FlightState,
        env:    Environment.Environment,
    ) -> FlightConditions:

        atm = Atmosphere(state.position.z)
        rho = float(atm.density)
        a   = float(atm.speed_of_sound)

        u, v    = env.windModel.at(state.position.z, seed=env.seed)
        wind_world = np.array([u, v, 0.0])
        airflow = Vector3D.Vector3D(
            wind_world - state.velocity.vectorWorld,
            state.orientation.as_matrix(),
        )

        v_mag = np.linalg.norm(state.velocity.elements)
        mach  = v_mag / a
        q     = 0.5 * rho * v_mag ** 2

        body_z_world = state.orientation.apply([0, 0, 1])
        airflow_world = airflow.vectorWorld
        airflow_norm  = np.linalg.norm(airflow_world)
        if airflow_norm > 0:
            alpha = np.arccos(
                np.clip(np.dot(body_z_world, airflow_world) / airflow_norm, -1.0, 1.0)
            )
        else:
            alpha = 0.0

        return cls(rho=rho, a=a, mach=mach, alpha=alpha, q=q, airflow=airflow)
