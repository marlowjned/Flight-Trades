# recovery.py
# Recovery is a process

from dataclasses import dataclass

import numpy as np

from flight_sim.core.sim_conditions import SimConditions


class Recovery:

    @dataclass
    class Device:
        cda: float
        deploy_alt: int = None
        deployed: bool = False

    def __init__(self, recovery_devices=None):
        self.recovery_devices = recovery_devices if recovery_devices is not None else []

    def recovery_force(self, fs, sc: SimConditions) -> np.ndarray:
        altitude = fs.position.vector_world[2]
        velocity_world = fs.velocity.vector_world
        v_mag = np.linalg.norm(velocity_world)

        self.update_deployment(velocity_world[2], altitude)

        net_force = np.zeros(3)
        for device in self.recovery_devices:
            if device.deployed and v_mag > 0:
                drag_mag = device.cda * 0.5 * sc.rho * v_mag ** 2
                net_force += -drag_mag * (velocity_world / v_mag)

        return net_force

    def update_deployment(self, vz: float, altitude: float):
        if vz < 0:
            for device in self.recovery_devices:
                if not device.deployed:
                    if device.deploy_alt is None or altitude <= device.deploy_alt:
                        device.deployed = True
