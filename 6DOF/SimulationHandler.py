# SimulationHandler.py
# Handles config file, creates actual trade framework, and packages data

import ConfigLoader
import Rocket

# TODO: in this order verify: 
#       - 1D single trial
#       - 6D single trial
#       - basic trade study

class SimulationHandler:

    def __init__(self, configPath: str):


        # TODO: assume every variable mentioned in the config loader is useful, even if it isn't conventionally used, may be used for the specific trade
        self.config = ConfigLoader.loadConfig(configPath)

        # process dimensions to iterate through

        # logic tree -> process trade v single trial
        # 1DOF v 6DOF 
        pass

    

    def sim1D(self, rocket: Rocket.Rocket):
        
        pass

