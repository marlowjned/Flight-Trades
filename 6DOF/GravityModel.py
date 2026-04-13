# GravityModel.py
# WGS84 gravity model

import boule as bl

from Vector3D import Vector3D

#def __init__(RefAltitude, RefLatitude):
#	refAltitude = RefAltitude
#	refLatitude = RefLatitude	

# can insert more specfics later

class GravityModel:
        def __init__(self):
                return

        # include pointer to altitude s.t. this can
        # be a property and not a function
        #@property
        def g(self, altitude) -> Vector3D:
                gravity = bl.WGS84.normal_gravity((0, 45, altitude), si_units=True) # m/s^2
                return Vector3D([0, 0, -gravity])
                # TODO: may need a input unit conversion
