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
import Engine
import Recovery
import Vector3D
import Environment

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
                     engine: Engine = None,
                     recovery: Recovery.Recovery = None):
                self.massData = massData
                self.rasCSV = rasCSV
                self.engine = engine
                self.recovery = recovery

                self.rasaero = RasAeroDataGrabber.RasAero(rasCSV)

                self.currFlightState = None
                self.currEnvironment = None

        # RasAero and ORK (and optional Engine) eng: Engine = None
        @classmethod
        def from_ork(cls, orkCSV: str, rasCSV: str, engine: Engine = None, recovery: Recovery.Recovery = None):
                massCurve, CGCurve, ILCurve, IRCurve, thrustCurve = cls._parse_ork(orkCSV)
                massComp = cls.MassComponent(massCurve, CGCurve, ILCurve, IRCurve, bool(engine))

                eng = engine if engine else Engine(None, thrustCurve)

                return cls(massComp, rasCSV, eng, recovery)

        @staticmethod
        def _parse_ork(orkCSV: str):
                orkData = pd.read_csv(orkCSV)
                massVars = ["Mass (g)", 
                            "CG location (cm)", 
                            "Longitudinal moment of inertia (kg·m²)", 
                            "Rotational moment of inertia (kg·m²)"]
                result = []
                for var in massVars:
                        result.append(CustomInterpolator.Interpolator1D(orkData["# Time (s)"], 
                                                                        orkData[var], 
                                                                        CustomInterpolator.Interpolator1D.BoundaryBehavior.LASTVAL))
                result.append(CustomInterpolator.Interpolator1D(orkData["# Time (s)"], 
                                                                        orkData['Thrust (N)'], 
                                                                        CustomInterpolator.Interpolator1D.BoundaryBehavior.LASTVAL))
                
                return result

        # RasAero, direct rocket data (net or engless), Engine

        # Dynamic Pressure
        def q(self, rho: float, v: float) -> float:
                return 0.5 * rho * (v ** 2)
        
        def machAlpha(self):
                _mach = self.currFlightState.velocity.magnitude / self.currEnvironment.a
                _alpha = self.angleBetweenVectors(self.currFlightState.orientation[0:2], self.effAirflow.vectorWorld)
                return _mach, _alpha

        def angleBetweenVectors(self, v1: np.ndarray, v2: np.ndarray) -> float:
                # sin(alpha) = v1 * v2 / |v1||v2|
                return np.arcsin(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        # TODO: include recovery force (if configured), maybe can be done within the simulation loop itself
        def aeroForce(self, fs: SimulationLoop.FlightSim.FlightState, env: Environment.Environment, thrusting: bool):
                self.updateStates(fs, env)

                mach, alpha = self.machAlpha()
                CDOFF, CDON, CL, CP = self.rasaero.getCoeffs(mach, alpha)

                drag = CDON * self.q if thrusting else CDOFF * self.q
                dragVector = drag * self.effAirflow.normalized.vectorWorld

                lift = CL * self.q
                liftUnitDirection = self.effAirflow.vectorWorld - np.dot(self.effAirflow, fs.orientation.as_quat()[1:3]) * fs.orientation.as_quat()[1:3]
                liftVector = lift * liftUnitDirection

                netForce = dragVector + liftVector
                return Vector3D(netForce), CP
        
        def aeroMoments(self, fs: SimulationLoop.FlightSim.FlightState, env: Environment.Environment, thrusting: bool):
                force, cp = self.aeroForce(fs, env, thrusting)

                arm = cp - self.massData.CG.query(fs.time)
                armVector = arm * fs.orientation.apply([0, 0, 1])

                moment = np.cross(armVector, force.elements)
                return moment

        def updateStates(self, fs: SimulationLoop.FlightSim.FlightState, env: Environment.Environment):
                if ((fs is not self.currFlightState) or (env is not self.currEnvironment)):
                        self.currFlightState = fs
                        self.currEnvironment = env
                        _effAirflow = env.windVector.vectorWorld - fs.velocity.vectorWorld
                        self.effAirflow = Vector3D(_effAirflow, fs.orientation.as_matrix())

        def moment(self):
                # cross (aero force) with (orientation * calipers)
                pass

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


