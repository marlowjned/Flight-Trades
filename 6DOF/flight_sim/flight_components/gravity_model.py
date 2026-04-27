# gravity_model.py
# WGS84 gravity model

import boule as bl

from flight_sim.data_helpers.vector3d import Vector3D


class GravityModel:
    def __init__(self):
        return

    def g(self, altitude) -> Vector3D:
        gravity = bl.WGS84.normal_gravity((0, 45, altitude), si_units=True)  # m/s^2
        return Vector3D([0, 0, -gravity])
