# SimulationLoop.py
# Handles individual full flight simulations

import numpy as np
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
import time

import Rocket
import Environment
import GravityModel
import Observer
import Vector3D

class FlightSim():

    # TODO: rework this class?
    class FlightState:
        def __init__(self, 
                     time: float,
                     position: Vector3D,
                     velocity: Vector3D,
                     orientation: Rotation,
                     omega: Vector3D):
            self.time = time
            self.position = position
            self.velocity = velocity
            self.orientation = orientation
            self.omega = omega

        @property
        def stateVector(self):
            return np.concatenate((self.time,
                                   self.position.elements, 
                                   self.velocity.elements,
                                   self.orientation.as_quat(),
                                   self.omega.elements))
        
        @classmethod
        def fromVector(cls, stateVec: np.ndarray):
            _time = stateVec[0]
            _position = stateVec[1:3]
            _velocity = stateVec[4:6]
            _orientation = stateVec[7:10]
            _omega = stateVec[11:13]
            return cls(_time, _position, _velocity, _orientation, _omega)

        # TODO: Add DCM method

    @dataclass
    class SimulationSettings:
        dt: float
        maxRuntime: int

        recoverySim: bool

        launchRailLen: float
        launRailOrientation: Rotation = Rotation.from_quat([0, 0, 0, 1])

    
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
        self._state = init_state


    def _acceleration(self, state: FlightState) -> np.ndarray: # TODO: change to stateVec
        # From forces: rocket aero, rocket thrust, gravity
        effectiveAirflow = self.Environment.wind - self.currentFlightState.velocity.vectorWorld

        aero = self._rocket.aeroForce(state, effectiveAirflow).vectorWorld / self._rocket.mass(state.time)
        thrust = self._rocket.thrust(state.time).vectorWorld / self._rocket.mass(state.time)
        gravity = self._gravity.g(state.position[3])
        
        return aero + thrust + gravity

    def _rotationalAcceleration(self, state: FlightState) -> np.ndarray:
        I = self._rocket.inertia_tensor(state.time)
        Idot = self._rocket.inertia_tensor_dot(state.time)
        tau = self._aero_torque(state)

        return np.linalg.solve(I, tau - Idot @ state.omega - np.cross(state.omega, I @ state.omega))
    
    def _derivatives(self, state: FlightState) -> np.ndarray:
        _dposition    = state.stateVec[4:6]
        _dvelocity    = self._acceleration(state) 
        _dorientation = np.concatenate([0], state.stateVec[11:13])
        _domega       = self._rotationalAcceleration(state)

        return np.concatenate([1], _dposition, _dvelocity, _dorientation, _domega) # choice of one is clear in the rk4 step function

    # make this smarter
    # integrating position, velocity, rot position, rot velocity
    # derivatives are velocities and accelerations respectively
    def _rk4_step(self, state: FlightState) -> FlightState:
        dt = self.settings.dt

        # TODO: probably a better way to do this
        init_state = state.stateVector
        k1 = self._derivatives(init_state)

        state2 = self.FlightState.fromVector(init_state + dt/2 * k1)
        k2 = self._derivatives(state2)

        state3 = self.FlightState.fromVector(init_state + dt/2 * k2)
        k3 = self._derivatives(state3)

        state4 = self.FlightState.fromVector(init_state + dt * k3)
        k4 = self._derivatives(state4)
        
        step_vector = init_state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        return self.FlightState.fromVector(step_vector)


    def run(self):
        # update own state at every time step

        onLaunchRail = True
        simulationRunning = True
        runtime = 0
        startTime = time.perf_counter()

        observer = Observer()
        observer.subscribe("State", lambda v: history["State"].append(v))

        while (simulationRunning and runtime < self.settings.maxRuntime):
            # check stopping logic

            # record data
            # OBSERVERS!!

            if onLaunchRail:
                # dot forces with launch rail orientation
                # take a step
                continue
            
            # take a step

            runtime = time.perf_counter() - startTime

        return


