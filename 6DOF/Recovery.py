# Recovery.py
# Recovery is a process

from dataclasses import dataclass

#import numpy as np

import Vector3D
import SimulationLoop
import Rocket
import Environment

class Recovery:

    @dataclass
    class Device:
        CdA: float
        deployAlt: int = None
        deployed: bool = False

    def __init__(self, recovery_devices = None):
        self.recoveryDevices = recovery_devices
        pass

    def recoveryForce(self, fs: SimulationLoop.FlightSim.FlightState, env: Environment.Environment):
        self.updateDeployment(fs.velocity.vectorWorld[2], fs.position.vectorWorld[2])

        netForce = 0
        for device in self.recoveryDevices:
            if device.deployed:
                netForce += device.CdA * 0.5 * env.rho * fs.effAirflow # move variables around so this works, much more convenient structure
                # TODO: for the record, this doesn't work yet

    def updateDeployment(self, vz: float, altitude: float):
        if vz < 0:
            for device in self.recoveryDevices:
                if altitude <= device.deployAlt:
                    device.deployed = True



