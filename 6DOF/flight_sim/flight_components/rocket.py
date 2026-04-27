# rocket.py
# Creates rocket object that stores all mass, aero, orientation, engine data

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.spatial.transform import Rotation

from flight_sim.data_helpers.custom_interpolator import Interpolator1D
from flight_sim.data_helpers.vector3d import Vector3D
from flight_sim.data_helpers.rasaero_loader import RasAeroLoader
from flight_sim.core.sim_conditions import SimConditions
from flight_sim.flight_components.engine import Engine
from flight_sim.flight_components.recovery import Recovery

if TYPE_CHECKING:
    import flight_sim.core.sim_loop as sim_loop_module


def _constant_curve(value: float, ref_curve: Interpolator1D) -> Interpolator1D:
    """Create a constant-value interpolator spanning the same time domain as ref_curve."""
    t0, t1 = ref_curve.x_bounds
    return Interpolator1D([t0, t1], [value, value], Interpolator1D.BoundaryBehavior.LASTVAL)


def _normalize_ork_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Replace non-ASCII characters in ORK column names."""
    df.columns = (df.columns
                  .str.replace('\u00b7', '*', regex=False)   # middle dot -> *
                  .str.replace('\u00b2', '2', regex=False))  # superscript 2 -> 2
    return df


class Rocket:

    @dataclass
    class MassComponent:
        mass:  Interpolator1D
        cg:    Interpolator1D
        i_long: Interpolator1D
        i_rot:  Interpolator1D

        override: bool = False  # Overrides subcomponents

    def __init__(self,
                 mass_data: MassComponent,
                 ras_csv: str,
                 ref_area: float,
                 engine: Engine = None,
                 recovery: Recovery = None):
        self.mass_data = mass_data
        self.ras_csv   = ras_csv
        self.ref_area  = ref_area
        self.engine    = engine
        self.recovery  = recovery

        self.rasaero = RasAeroLoader(ras_csv)

    @classmethod
    def from_ork(cls, ork_csv: str, ras_csv: str, ref_area: float,
                 engine: Engine = None, recovery: Recovery = None):
        mass_curve, cg_curve, il_curve, ir_curve, thrust_curve = cls._parse_ork(ork_csv)
        mass_comp = cls.MassComponent(mass_curve, cg_curve, il_curve, ir_curve, bool(engine))

        eng = engine if engine else Engine(None, thrust_curve)

        return cls(mass_comp, ras_csv, ref_area, eng, recovery)

    @staticmethod
    def _parse_ork(ork_csv: str):
        ork_data = _normalize_ork_columns(pd.read_csv(ork_csv))
        t = ork_data["# Time (s)"]
        LASTVAL = Interpolator1D.BoundaryBehavior.LASTVAL

        mass_kg = ork_data["Mass (g)"] / 1000.0
        cg_m    = ork_data["CG location (cm)"] / 100.0
        i_long  = ork_data["Longitudinal moment of inertia (kg*m2)"]
        i_rot   = ork_data["Rotational moment of inertia (kg*m2)"]
        thrust  = ork_data["Thrust (N)"]

        result = [
            Interpolator1D(t, mass_kg, LASTVAL),
            Interpolator1D(t, cg_m,    LASTVAL),
            Interpolator1D(t, i_long,  LASTVAL),
            Interpolator1D(t, i_rot,   LASTVAL),
        ]
        ZEROVAL = Interpolator1D.BoundaryBehavior.ZEROVAL
        result.append(Interpolator1D(t, thrust, ZEROVAL))

        return result

    def q(self, rho: float, v: float) -> float:
        return 0.5 * rho * (v ** 2)

    def aero_force(self, fs: sim_loop_module.FlightSim.FlightState,
                   sc: SimConditions, thrusting: bool):
        cd_off, cd_on, cl, cp = self.rasaero.get_coeffs(sc.mach, sc.alpha)

        drag = (cd_on if thrusting else cd_off) * sc.q * self.ref_area
        drag_vector = drag * sc.airflow.normalized.vector_world

        lift = cl * sc.q * self.ref_area
        body_axis = fs.orientation.apply([0, 0, 1])
        airflow_world = sc.airflow.vector_world
        lift_unit_dir = airflow_world - np.dot(airflow_world, body_axis) * body_axis
        lift_mag = np.linalg.norm(lift_unit_dir)
        lift_unit_dir = lift_unit_dir / lift_mag if lift_mag > 0 else np.zeros(3)
        lift_vector = lift * lift_unit_dir

        net_force = drag_vector + lift_vector
        return Vector3D(net_force), cp

    def aero_moments(self, fs: sim_loop_module.FlightSim.FlightState,
                     sc: SimConditions, thrusting: bool):
        force, cp = self.aero_force(fs, sc, thrusting)

        arm = cp - self.mass_data.cg.query(fs.time)
        arm_vector = -arm * fs.orientation.apply([0, 0, 1])

        moment_world = np.cross(arm_vector, force.elements)
        return fs.orientation.inv().apply(moment_world)

    def mass(self, time: float):
        return self.mass_data.mass.query(time)

    def cg(self, time: float):
        return self.mass_data.cg.query(time)

    def inertia(self, time: float) -> np.ndarray:
        i_long = self.mass_data.i_long.query(time)
        i_rot  = self.mass_data.i_rot.query(time)
        return np.array([[i_long, 0,      0    ],
                         [0,      i_long, 0    ],
                         [0,      0,      i_rot]])

    def apply_overrides(self, overrides: dict):
        """
        Apply config overrides to the rocket after construction.

        Supported keys:
          mass          (float, kg)  — constant total mass, replaces ORK time-varying curve
          cg            (float, m)   — constant CG from nose tip, replaces ORK curve
          thrust_scale  (float)      — multiplier on the ORK thrust curve
        """
        if 'mass' in overrides:
            self.mass_data.mass = _constant_curve(float(overrides['mass']),
                                                  self.mass_data.mass)
        if 'cg' in overrides:
            self.mass_data.cg = _constant_curve(float(overrides['cg']),
                                                self.mass_data.cg)
        if 'thrust_scale' in overrides:
            self.engine.thrust_scale = float(overrides['thrust_scale'])

    def inertia_dot(self, time: float) -> np.ndarray:
        i_long_dot = self.mass_data.i_long.derivative().query(time)
        i_rot_dot  = self.mass_data.i_rot.derivative().query(time)
        return np.array([[i_long_dot, 0,          0        ],
                         [0,          i_long_dot, 0        ],
                         [0,          0,          i_rot_dot]])
