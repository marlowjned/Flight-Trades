# Recovery.py
# Recovery is a process

from dataclasses import dataclass

import numpy as np

from SimulationConditions import SimConditions


class Recovery:

    @dataclass
    class Device:
        CdA: float
        deployAlt: int = None
        deployed: bool = False

    def __init__(self, recovery_devices=None):
        self.recoveryDevices = recovery_devices if recovery_devices is not None else []

    def recoveryForce(self, fs, sc: SimConditions) -> np.ndarray:
        altitude = fs.position.vectorWorld[2]
        velocity_world = fs.velocity.vectorWorld
        v_mag = np.linalg.norm(velocity_world)

        self.updateDeployment(velocity_world[2], altitude)

        net_force = np.zeros(3)
        for device in self.recoveryDevices:
            if device.deployed and v_mag > 0:
                drag_mag = device.CdA * 0.5 * sc.rho * v_mag ** 2
                net_force += -drag_mag * (velocity_world / v_mag)

        return net_force

    def updateDeployment(self, vz: float, altitude: float):
        if vz < 0:
            for device in self.recoveryDevices:
                if not device.deployed:
                    if device.deployAlt is None or altitude <= device.deployAlt:
                        device.deployed = True
