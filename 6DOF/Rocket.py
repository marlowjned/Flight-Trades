# Rocket.py
# Creates rocket object that stores all mass, aero, orientation, engine data



# rocketMass or overrideMass (lastval interpolator)
# Ixx, Iyy (direct assignment often from ORK), dI/dt
# CG
# Aero data (aero class)
# TODO: Recovery system (recovery class)
# TODO: Engine (class)

import ORKDataGrabber
import RasAeroDataGrabber
import SimulationLoop


class Rocket:

        # either: provided ork and eng, provided one, or none
        # rasaero always needed
        # engine, ork, or direct
        # ork provided -> rocket (+ ork engine if no direct engine given)
        # no ork -> direct rocket + direct engine
        # does it make sense to consider rocket and engine separately then?

        '''
	def __init__(self, orkFileName: str, rasFileName: str):
                odg = ORKDataGrabber(csvPath)
                
                self.mass() = odg.netMassCurve()
                self.CG() = odg.netCGCurve()
                self.longMOI() = odg.longMOICurve()
                self.rotMOI() = odg.rotMOICurve()

                self.rasaero = RasAero(rasFilePath, RasAero.Frame.WORLD)
                self.CP() = self.rasaero.coeffTable('CP')
        

        def inertia(self, time: float):
                l = longMOI.query(time)
                r = rotMOI.query(time)
                return np.array([[l, 0, 0],
                                 [0, r, 0],
                                 [0, 0, r]])
        '''
        # TODO: all this shit
        
        # TODO: Add derivative method to custom interpolator
        # Ultimate needs: Mass, Ijj, CG, Aero Data, recovery, engine
        # If a recovery system is not provided, assumed to stop at apogee
        
        def __init__(self):
                pass

        # RasAero and ORK (and optional Engine) eng: Engine = None
        @classmethod
        def from_csv(cls, path):
                return cls(mass)

        # RasAero, direct rocket data (net or engless), Engine


        def aeroForce(fs: SimulationLoop.FlightSim.FlightState):
                pass

        def moment():
                pass

        @property 
        def mass():
                pass

        


