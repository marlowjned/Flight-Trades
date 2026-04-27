# sim_loop.py
# Handles individual full flight simulations

import numpy as np
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
import time

from flight_sim.flight_components.rocket import Rocket
from flight_sim.flight_components.gravity_model import GravityModel
from flight_sim.wind.wind_model_base import WindModelBase
from flight_sim.data_helpers.vector3d import Vector3D
from flight_sim.core.sim_snapshot import SimSnapshot
from flight_sim.core.sim_conditions import SimConditions


class FlightSim:

    class FlightState:
        def __init__(self,
                     time: float = 0,
                     position: Vector3D = Vector3D(),
                     velocity: Vector3D = Vector3D(),
                     orientation: Rotation = Rotation.from_quat([0, 0, 0, 1]),
                     omega: Vector3D = Vector3D()):
            self.time        = time
            self.position    = position
            self.velocity    = velocity
            self.orientation = orientation
            self.omega       = omega

        @property
        def state_vector(self):
            return np.concatenate((np.array([self.time]),
                                   self.position.elements,
                                   self.velocity.elements,
                                   self.orientation.as_quat(),
                                   self.omega.elements))

        @classmethod
        def from_vector(cls, state_vec: np.ndarray):
            _time        = state_vec[0]
            _orientation = Rotation.from_quat(state_vec[7:11])
            dcm          = _orientation.as_matrix()
            _position    = Vector3D(state_vec[1:4], dcm)
            _velocity    = Vector3D(state_vec[4:7], dcm)
            _omega       = Vector3D(state_vec[11:14], dcm, True)
            return cls(_time, _position, _velocity, _orientation, _omega)

    @dataclass
    class SimulationSettings:
        dt: float
        max_runtime: int

        recovery_sim: bool

        launch_rail_len: float
        launch_rail_orientation: Rotation = Rotation.from_quat([0, 0, 0, 1])

    def __init__(self, rocket: Rocket,
                 wind_model: WindModelBase,
                 gravity: GravityModel,
                 init_state: FlightState,
                 settings: SimulationSettings):

        self._rocket     = rocket
        self._wind_model = wind_model
        self._gravity    = gravity

        self.settings = settings
        self.state    = init_state

    def _acceleration(self, state: FlightState, sc: SimConditions) -> np.ndarray:
        mass      = self._rocket.mass(state.time)
        thrust    = self._rocket.engine.thrust_vector(state.time, state.orientation.as_matrix()).vector_world / mass
        thrusting = np.linalg.norm(thrust) > 0

        aero_force, _ = self._rocket.aero_force(state, sc, thrusting)
        aero    = aero_force.vector_world / mass
        gravity = self._gravity.g(state.position.z).elements

        accel = thrust + aero + gravity

        if self.settings.recovery_sim and self._rocket.recovery is not None:
            accel += self._rocket.recovery.recovery_force(state, sc) / mass

        return accel

    def _rotational_acceleration(self, state: FlightState, sc: SimConditions) -> np.ndarray:
        I    = self._rocket.inertia(state.time)
        Idot = self._rocket.inertia_dot(state.time)
        thrusting = np.linalg.norm(
            self._rocket.engine.thrust_vector(state.time, state.orientation.as_matrix()).vector_world
        ) > 0
        tau = self._rocket.aero_moments(state, sc, thrusting)

        return np.linalg.solve(
            I, tau - Idot @ state.omega.elements - np.cross(state.omega.elements, I @ state.omega.elements)
        )

    def _derivatives(self, state: FlightState, constraint: np.ndarray) -> np.ndarray:
        sc = SimConditions.compute(state, self._wind_model)

        d_position = state.velocity.elements
        d_velocity = self._acceleration(state, sc)
        q = state.orientation.as_quat()  # [qx, qy, qz, qw] scipy scalar-last
        Xi = np.array([
            [ q[3], -q[2],  q[1]],
            [ q[2],  q[3], -q[0]],
            [-q[1],  q[0],  q[3]],
            [-q[0], -q[1], -q[2]],
        ])
        d_orientation = 0.5 * Xi @ state.omega.elements
        d_omega       = self._rotational_acceleration(state, sc)

        # Launch rail: constrain motion to rail direction
        if constraint is not None:
            v_along = np.dot(d_position, constraint)
            a_along = np.dot(d_velocity, constraint)

            d_position = max(v_along, 0.0) * constraint

            if v_along <= 0.0 and a_along < 0.0:
                a_along = 0.0
            d_velocity = a_along * constraint

            d_orientation = np.zeros(4)
            d_omega       = np.zeros(3)

        return np.concatenate([[1], d_position, d_velocity, d_orientation, d_omega])

    def _rk4_step(self, state: FlightState, constraint: np.ndarray = None) -> FlightState:
        dt = self.settings.dt

        init_state = state.state_vector
        k1 = self._derivatives(state, constraint)

        state2 = self.FlightState.from_vector(init_state + dt / 2 * k1)
        k2 = self._derivatives(state2, constraint)

        state3 = self.FlightState.from_vector(init_state + dt / 2 * k2)
        k3 = self._derivatives(state3, constraint)

        state4 = self.FlightState.from_vector(init_state + dt * k3)
        k4 = self._derivatives(state4, constraint)

        step_vector = init_state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return self.FlightState.from_vector(step_vector)

    def run(self):
        on_launch_rail = True
        rail_dir = self.settings.launch_rail_orientation.apply([0, 0, 1])
        rail_len = self.settings.launch_rail_len
        runtime  = 0
        start_time = time.perf_counter()

        snapshots = []

        while runtime < self.settings.max_runtime:
            if self.state.position.vector_world[2] < 0:
                break
            if (not self.settings.recovery_sim) and self.state.velocity.vector_world[2] < 0:
                break

            sc = SimConditions.compute(self.state, self._wind_model)
            snapshots.append(SimSnapshot(self.state, sc, self._rocket))

            if hasattr(self._wind_model, 'advance'):
                V = np.linalg.norm(self.state.velocity.vector_world)
                self._wind_model.advance(self.state.time, self.state.position.z, V)

            if on_launch_rail:
                self.state = self._rk4_step(self.state, rail_dir)

                if np.dot(self.state.position.vector_world, rail_dir) >= rail_len:
                    on_launch_rail = False
                runtime = time.perf_counter() - start_time
                continue

            self.state = self._rk4_step(self.state)
            runtime = time.perf_counter() - start_time

        return snapshots
