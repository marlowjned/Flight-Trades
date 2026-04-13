# SimSnapshot.py
# Organizes individual simulation data collection
# Other parser handles data collection for trades

import numpy as np
import pandas as pd
from functools import cached_property

import Rocket
from SimulationConditions import SimConditions

class SimSnapshot:
    def __init__(self, fs, sc: SimConditions, rocket: Rocket.Rocket):
        self._state = fs
        self._sc = sc
        self._rocket = rocket

    # --- Time / Position ---
    @cached_property
    def time(self): return self._state.time

    @cached_property
    def altitude(self): return float(self._state.position.vectorWorld[2])

    @cached_property
    def pos_x(self): return float(self._state.position.vectorWorld[0])

    @cached_property
    def pos_y(self): return float(self._state.position.vectorWorld[1])

    # --- Velocity ---
    @cached_property
    def vel_x(self): return float(self._state.velocity.vectorWorld[0])

    @cached_property
    def vel_y(self): return float(self._state.velocity.vectorWorld[1])

    @cached_property
    def vel_z(self): return float(self._state.velocity.vectorWorld[2])

    @cached_property
    def speed(self): return float(np.linalg.norm(self._state.velocity.vectorWorld))

    # --- Aerodynamics ---
    @cached_property
    def mach(self): return self._sc.mach

    @cached_property
    def alpha_deg(self): return float(np.degrees(self._sc.alpha))

    @cached_property
    def dynamic_pressure(self): return self._sc.q

    @cached_property
    def air_density(self): return self._sc.rho

    # --- Attitude (ZYX Euler: yaw, pitch, roll) ---
    @cached_property
    def _euler(self):
        return self._state.orientation.as_euler('ZYX', degrees=True)

    @cached_property
    def yaw(self): return float(self._euler[0])

    @cached_property
    def pitch(self): return float(self._euler[1])

    @cached_property
    def roll(self): return float(self._euler[2])

    # --- Angular velocity (body frame) ---
    @cached_property
    def omega_x(self): return float(self._state.omega.elements[0])

    @cached_property
    def omega_y(self): return float(self._state.omega.elements[1])

    @cached_property
    def omega_z(self): return float(self._state.omega.elements[2])

    # --- Rocket ---
    @cached_property
    def mass(self): return float(self._rocket.mass(self._state.time))

    @cached_property
    def thrust(self): return float(self._rocket.engine.thrust.query(self._state.time))

    # --- Export ---
    def to_dict(self) -> dict:
        return {
            "time":             self.time,
            "altitude":         self.altitude,
            "pos_x":            self.pos_x,
            "pos_y":            self.pos_y,
            "vel_x":            self.vel_x,
            "vel_y":            self.vel_y,
            "vel_z":            self.vel_z,
            "speed":            self.speed,
            "mach":             self.mach,
            "alpha_deg":        self.alpha_deg,
            "dynamic_pressure": self.dynamic_pressure,
            "air_density":      self.air_density,
            "pitch":            self.pitch,
            "yaw":              self.yaw,
            "roll":             self.roll,
            "omega_x":          self.omega_x,
            "omega_y":          self.omega_y,
            "omega_z":          self.omega_z,
            "mass":             self.mass,
            "thrust":           self.thrust,
        }


def export_snapshots_csv(snapshots: list, path: str):
    pd.DataFrame([s.to_dict() for s in snapshots]).to_csv(path, index=False)
