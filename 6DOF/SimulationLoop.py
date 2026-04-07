# SimulationLoop.py
# Handles individual full flight simulations

import numpy as np
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
import time

import Rocket
import Environment
import GravityModel
import Vector3D
import SimSnapshot

class FlightSim():

    # TODO: rework this class?
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
            _position = Vector3D.Vector3D(stateVec[1:4], dcm)
            _velocity = Vector3D.Vector3D(stateVec[4:7], dcm)
            _omega    = Vector3D.Vector3D(stateVec[11:14], dcm, True)
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
                 environment: Environment,
                 gravity: GravityModel, 
                 init_state: FlightState,
                 settings: SimulationSettings):

        self._rocket = rocket
        self._environment = environment
        self._gravity = gravity

        self.settings = settings
        self.state = init_state

    def getDCM(self):
        return self.state.orientation.as_matrix()


    def _acceleration(self, state: FlightState) -> np.ndarray: # TODO: change to stateVec
        # From forces: rocket aero, rocket thrust, gravity
        thrust = self._rocket.engine.thrustVector(state.time, self.getDCM()).vectorWorld / self._rocket.mass(state.time)
        thrusting = True if np.linalg.norm(thrust) > 0 else False

        aero = self._rocket.aeroForce(state, self._environment, thrusting).vectorWorld / self._rocket.mass(state.time)
        gravity = self._gravity.g(state.position.z)
        
        return thrust + aero + gravity

    def _rotationalAcceleration(self, state: FlightState) -> np.ndarray:
        I = self._rocket.inertia(state.time) # In body frame, okay because extra term below
        Idot = self._rocket.inertia_tensor_dot(state.time)
        tau = self._rocket.aeroMoments(state, self._environment, np.linalg.norm(self._rocket.engine.thrustVector(state.time, self.getDCM()).vectorWorld) > 0)

        return np.linalg.solve(I, tau - Idot @ state.omega.elements - np.cross(state.omega.elements, I @ state.omega.elements))
    
    def _derivatives(self, state: FlightState, constraint: np.ndarray) -> np.ndarray:
        _dposition    = state.velocity.elements
        _dvelocity    = self._acceleration(state)
        q = state.orientation.as_quat()  # [qx, qy, qz, qw] scipy scalar-last
        Xi = np.array([
            [ q[3], -q[2],  q[1]],
            [ q[2],  q[3], -q[0]],
            [-q[1],  q[0],  q[3]],
            [-q[0], -q[1], -q[2]],
        ])
        _dorientation = 0.5 * Xi @ state.omega.elements
        _domega       = self._rotationalAcceleration(state)

        # Launch rail TODO: FIX TS
        if constraint is not None:
            _dposition    = np.dot(_dposition, constraint) * constraint if np.dot(_dposition, constraint) > 0 else np.zeros(3)
            _dvelocity    = np.dot(_dvelocity, constraint) * constraint if np.dot(_dvelocity, constraint) > 0 else np.zeros(3)
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
        launchRailHeight = self.settings.launchRailLen * self.settings.launchRailOrientation.apply([0, 0, 1])[2]
        runtime = 0
        startTime = time.perf_counter()

        snapshots = []

        while (runtime < self.settings.maxRuntime):
            if (self.state.position.vectorWorld[2] < 0): break
            if ((self.settings.recoverySim is False) and (self.state.velocity.vectorWorld[2] < 0)): break

            snapshots.append(SimSnapshot.SimSnapshot(self.state, self._rocket, self._environment))

            if onLaunchRail:
                railDir = self.settings.launchRailOrientation.apply([0, 0, 1])
                self.state = self._rk4_step(self.state, railDir)

                if self.state.position.z > launchRailHeight: onLaunchRail = False
                runtime = time.perf_counter() - startTime

                continue

            self.state = self._rk4_step(self.state)
            runtime = time.perf_counter() - startTime

        return snapshots


