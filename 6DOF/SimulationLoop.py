# SimulationLoop.py
# Handles individual full flight simulations

import numpy as np
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
import time

import Rocket
import WindModel
import GravityModel
from Vector3D import Vector3D
import SimSnapshot
from SimulationConditions import SimConditions

class FlightSim():

    class FlightState:
        def __init__(self,
                     time: float = 0,
                     position: Vector3D = Vector3D(),
                     velocity: Vector3D = Vector3D(),
                     orientation: Rotation = Rotation.from_quat([0, 0, 0, 1]),
                     omega: Vector3D = Vector3D()):
            self.time = time
            self.position = position
            self.velocity = velocity
            self.orientation = orientation
            self.omega = omega

        @property
        def stateVector(self):
            return np.concatenate((np.array([self.time]),
                                   self.position.elements,
                                   self.velocity.elements,
                                   self.orientation.as_quat(),
                                   self.omega.elements))

        @classmethod
        def fromVector(cls, stateVec: np.ndarray):
            _time = stateVec[0]
            _orientation = Rotation.from_quat(stateVec[7:11])
            dcm = _orientation.as_matrix()
            _position = Vector3D(stateVec[1:4], dcm)
            _velocity = Vector3D(stateVec[4:7], dcm)
            _omega    = Vector3D(stateVec[11:14], dcm, True)
            return cls(_time, _position, _velocity, _orientation, _omega)

        # TODO: Add DCM method

    @dataclass
    class SimulationSettings:
        dt: float
        maxRuntime: int

        recoverySim: bool

        launchRailLen: float
        launchRailOrientation: Rotation = Rotation.from_quat([0, 0, 0, 1])


    # FlightSim
    def __init__(self, rocket: Rocket,
                 windModel: WindModel.WindModel,
                 gravity: GravityModel,
                 init_state: FlightState,
                 settings: SimulationSettings):

        self._rocket = rocket
        self._windModel = windModel
        self._gravity = gravity

        self.settings = settings
        self.state = init_state


    def _acceleration(self, state: FlightState, sc: SimConditions) -> np.ndarray:
        mass = self._rocket.mass(state.time)
        thrust = self._rocket.engine.thrustVector(state.time, state.orientation.as_matrix()).vectorWorld / mass
        thrusting = np.linalg.norm(thrust) > 0

        aero_force, _ = self._rocket.aeroForce(state, sc, thrusting)
        aero = aero_force.vectorWorld / mass
        gravity = self._gravity.g(state.position.z).elements

        accel = thrust + aero + gravity

        if self.settings.recoverySim and self._rocket.recovery is not None:
            accel += self._rocket.recovery.recoveryForce(state, sc) / mass

        return accel

    def _rotationalAcceleration(self, state: FlightState, sc: SimConditions) -> np.ndarray:
        I    = self._rocket.inertia(state.time)  # In body frame, okay because extra term below
        Idot = self._rocket.inertia_dot(state.time)
        thrusting = np.linalg.norm(self._rocket.engine.thrustVector(state.time, state.orientation.as_matrix()).vectorWorld) > 0
        tau  = self._rocket.aeroMoments(state, sc, thrusting)

        return np.linalg.solve(I, tau - Idot @ state.omega.elements - np.cross(state.omega.elements, I @ state.omega.elements))

    def _derivatives(self, state: FlightState, constraint: np.ndarray) -> np.ndarray:
        sc = SimConditions.compute(state, self._windModel)

        _dposition    = state.velocity.elements
        _dvelocity    = self._acceleration(state, sc)
        q = state.orientation.as_quat()  # [qx, qy, qz, qw] scipy scalar-last
        Xi = np.array([
            [ q[3], -q[2],  q[1]],
            [ q[2],  q[3], -q[0]],
            [-q[1],  q[0],  q[3]],
            [-q[0], -q[1], -q[2]],
        ])
        _dorientation = 0.5 * Xi @ state.omega.elements
        _domega       = self._rotationalAcceleration(state, sc)

        # Launch rail: constrain motion to rail direction (one-sided: can't slide backward)
        if constraint is not None:
            v_along = np.dot(_dposition, constraint)
            a_along = np.dot(_dvelocity, constraint)

            # Position derivative: project velocity onto rail, clamp to zero if backward
            _dposition = max(v_along, 0.0) * constraint

            # Velocity derivative: project acceleration onto rail.
            # Rail only blocks backward acceleration when rocket is stopped against the hold-down.
            # If already moving forward, allow deceleration freely.
            if v_along <= 0.0 and a_along < 0.0:
                a_along = 0.0  # hold-down: reaction force cancels backward pull
            _dvelocity = a_along * constraint

            _dorientation = np.zeros(4)
            _domega       = np.zeros(3)

        return np.concatenate([[1], _dposition, _dvelocity, _dorientation, _domega])

    # make this smarter
    # integrating position, velocity, rot position, rot velocity
    # derivatives are velocities and accelerations respectively
    def _rk4_step(self, state: FlightState, constraint: np.ndarray = None) -> FlightState:
        dt = self.settings.dt

        # TODO: probably a better way to do this
        init_state = state.stateVector
        k1 = self._derivatives(state, constraint)

        state2 = self.FlightState.fromVector(init_state + dt/2 * k1)
        k2 = self._derivatives(state2, constraint)

        state3 = self.FlightState.fromVector(init_state + dt/2 * k2)
        k3 = self._derivatives(state3, constraint)

        state4 = self.FlightState.fromVector(init_state + dt * k3)
        k4 = self._derivatives(state4, constraint)

        step_vector = init_state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        return self.FlightState.fromVector(step_vector)


    def run(self):
        onLaunchRail = True
        railDir = self.settings.launchRailOrientation.apply([0, 0, 1])
        railLen = self.settings.launchRailLen
        runtime = 0
        startTime = time.perf_counter()

        snapshots = []

        while (runtime < self.settings.maxRuntime):
            if (self.state.position.vectorWorld[2] < 0): break
            if ((self.settings.recoverySim is False) and (self.state.velocity.vectorWorld[2] < 0)): break

            sc = SimConditions.compute(self.state, self._windModel)
            snapshots.append(SimSnapshot.SimSnapshot(self.state, sc, self._rocket))

            # Advance time-dependent wind models exactly once before RK4
            if hasattr(self._windModel, 'advance'):
                V = np.linalg.norm(self.state.velocity.vectorWorld)
                self._windModel.advance(self.state.time, self.state.position.z, V)

            if onLaunchRail:
                self.state = self._rk4_step(self.state, railDir)

                # Exit rail when distance traveled along rail direction exceeds rail length
                if np.dot(self.state.position.vectorWorld, railDir) >= railLen:
                    onLaunchRail = False
                runtime = time.perf_counter() - startTime

                continue

            self.state = self._rk4_step(self.state)
            runtime = time.perf_counter() - startTime

        return snapshots
