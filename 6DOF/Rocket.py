# Rocket.py
# Creates rocket object that stores all mass, aero, orientation, engine data



# rocketMass or overrideMass (lastval interpolator)
# Ixx, Iyy (direct assignment often from ORK), dI/dt
# CG
# Aero data (aero class)
# TODO: Recovery system (recovery class)
# TODO: Engine (class)

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.spatial.transform import Rotation

import ORKDataGrabber
import RasAeroDataGrabber
import CustomInterpolator
from Engine import Engine
from Recovery import Recovery
from Vector3D import Vector3D
from SimulationConditions import SimConditions

if TYPE_CHECKING:
    import SimulationLoop


class Rocket:

        # either: provided ork and eng, provided one, or none
        # rasaero always needed
        # engine, ork, or direct
        # ork provided -> rocket (+ ork engine if no direct engine given)
        # no ork -> direct rocket + direct engine
        # does it make sense to consider rocket and engine separately then?

        # TODO: Make own file? May be worthwhile to allow mass component subsystems for easy implementations of variations
        @dataclass
        class MassComponent:
                mass:  CustomInterpolator.Interpolator1D
                CG:    CustomInterpolator.Interpolator1D
                Ilong: CustomInterpolator.Interpolator1D
                Irot:  CustomInterpolator.Interpolator1D

                override: bool = False # Overrides subcomponents


        # TODO: some of these variables could be better organized
        def __init__(self,
                     massData: MassComponent,
                     rasCSV: str,
                     refArea: float,
                     engine: Engine = None,
                     recovery: Recovery = None):
                self.massData = massData
                self.rasCSV = rasCSV
                self.refArea = refArea
                self.engine = engine
                self.recovery = recovery

                self.rasaero = RasAeroDataGrabber.RasAero(rasCSV)


        # RasAero and ORK (and optional Engine) eng: Engine = None
        @classmethod
        def from_ork(cls, orkCSV: str, rasCSV: str, refArea: float, engine: Engine = None, recovery: Recovery = None):
                massCurve, CGCurve, ILCurve, IRCurve, thrustCurve = cls._parse_ork(orkCSV)
                massComp = cls.MassComponent(massCurve, CGCurve, ILCurve, IRCurve, bool(engine))

                eng = engine if engine else Engine(None, thrustCurve)

                return cls(massComp, rasCSV, refArea, eng, recovery)

        @staticmethod
        def _parse_ork(orkCSV: str):
                orkData = pd.read_csv(orkCSV)
                t = orkData["# Time (s)"]
                LASTVAL = CustomInterpolator.Interpolator1D.BoundaryBehavior.LASTVAL

                mass_kg  = orkData["Mass (g)"] / 1000.0
                cg_m     = orkData["CG location (cm)"] / 100.0
                ilong    = orkData["Longitudinal moment of inertia (kg·m²)"]
                irot     = orkData["Rotational moment of inertia (kg·m²)"]
                thrust   = orkData["Thrust (N)"]

                result = [
                    CustomInterpolator.Interpolator1D(t, mass_kg, LASTVAL),
                    CustomInterpolator.Interpolator1D(t, cg_m,    LASTVAL),
                    CustomInterpolator.Interpolator1D(t, ilong,   LASTVAL),
                    CustomInterpolator.Interpolator1D(t, irot,    LASTVAL),
                ]
                ZEROVAL = CustomInterpolator.Interpolator1D.BoundaryBehavior.ZEROVAL
                result.append(CustomInterpolator.Interpolator1D(t, thrust, ZEROVAL))
                
                return result

        # RasAero, direct rocket data (net or engless), Engine

        # Dynamic Pressure
        def q(self, rho: float, v: float) -> float:
                return 0.5 * rho * (v ** 2)
        
        # TODO: include recovery force (if configured), maybe can be done within the simulation loop itself
        def aeroForce(self, fs: SimulationLoop.FlightSim.FlightState, sc: SimConditions, thrusting: bool):
                CDOFF, CDON, CL, CP = self.rasaero.getCoeffs(sc.mach, sc.alpha)

                drag = (CDON if thrusting else CDOFF) * sc.q * self.refArea
                dragVector = drag * sc.airflow.normalized.vectorWorld

                lift = CL * sc.q * self.refArea
                body_axis = fs.orientation.apply([0, 0, 1])
                airflow_world = sc.airflow.vectorWorld
                liftUnitDirection = airflow_world - np.dot(airflow_world, body_axis) * body_axis
                lift_mag = np.linalg.norm(liftUnitDirection)
                liftUnitDirection = liftUnitDirection / lift_mag if lift_mag > 0 else np.zeros(3)
                liftVector = lift * liftUnitDirection

                netForce = dragVector + liftVector
                return Vector3D(netForce), CP

        def aeroMoments(self, fs: SimulationLoop.FlightSim.FlightState, sc: SimConditions, thrusting: bool):
                force, cp = self.aeroForce(fs, sc, thrusting)

                arm = cp - self.massData.CG.query(fs.time)
                armVector = -arm * fs.orientation.apply([0, 0, 1])  # points from CG toward CP (aft)

                moment_world = np.cross(armVector, force.elements)
                return fs.orientation.inv().apply(moment_world)  # body frame for Euler equation

        def mass(self, time: float):
                # TODO: logic for override + adding subcomponenets
                return self.massData.mass.query(time)
        
        def CG(self, time: float):
                # TODO: logic for override + subcomponents
                return self.massData.CG.query(time)

        def inertia(self, time: float) -> np.ndarray:
                Ilong = self.massData.Ilong.query(time)
                Irot  = self.massData.Irot.query(time)
                return np.array([[Ilong, 0,     0    ],
                                 [0,     Ilong, 0    ],
                                 [0,     0,     Irot ]])

        def inertia_dot(self, time: float) -> np.ndarray:
                IlongDot = self.massData.Ilong.derivative().query(time)
                IrotDot  = self.massData.Irot.derivative().query(time)
                return np.array([[IlongDot, 0,        0       ],
                                 [0,        IlongDot, 0       ],
                                 [0,        0,        IrotDot ]])


