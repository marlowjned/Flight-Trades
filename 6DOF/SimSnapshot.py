# SimSnapshot.py
# Organizes individual simulation data collection
# Other parser handles data collection for trades

from functools import cached_property

import SimulationLoop
import Rocket
import Environment

class SimSnapshot:
    def __init__(self, fs: SimulationLoop.FlightSim.FlightState, 
                 rocket: Rocket.Rocket, 
                 env: Environment.Environment):
        self._state = fs
        self._rocket = rocket
        self._environment = env

    @cached_property
    def time(self): return self._state.time
    @cached_property
    def altitude(self): return self._state.position.z
    # etc
