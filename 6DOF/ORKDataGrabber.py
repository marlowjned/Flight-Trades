# ORKDataGrabber.py
# Initializes rocket data from an ORK csv


import pandas as pd

import CustomInterpolator


# make @ property ?

class ORKDataGrabber:
	def __init__(self, csvName: str):
                _sampleData = pd.read_csv(csvName)

	# Mass (g)
	def netMassCurve(self) -> 1DInterpolator:
		return 1DInterpolator(_sampleData["# Time (s)"],
                                      _sampleData["Mass (g)"],
                                      Interpolator1D.BoundaryBehavior.LASTVAL)

	def englessMass(self) -> float:
		return _sampleData["Mass (g)"][-1]

	# CG (from tip, cm)
	def netCGCurve(self) -> 1DInterpolator:
		return 1DInterpolator(_sampleData["# Time (s)"], 
                                      _sampleData["CG location (cm)"],
                                      Interpolator1D.BoundaryBehavior.LASTVAL)

	def englessCG(self) -> float: # TODO: check this
		return _sampleData["CG location (cm)"][-1]

	# Longitudinal and rotational MOI (kg m^2)
	# Construct inertia tensor in Aerodynamics.py
	# TODO: may need to preprocess by only permitting ASCII characters
	# TODO: final moi?
        def longMOICurve(self) -> 1DInterpolator:
                return 1DInterpolator(_sampleData["# Time (s)"],
                                      _sampleData["Longitudinal moment of inertia (kg·m²)"],
                                      Interpolator1D.BoundaryBehavior.LASTVAL)
        
        def englessLongMOI(self) -> float:
                return _sampleData["Longitudinal moment of inertia (kg·m²)"][-1]

        def rotMOICurve(self) -> 1DInterpolator:
                return 1DInterpolator(_sampleData["# Time (s)"],
                                      _sampleData["Rotational moment of inertia (kg·m²)"],
                                      Interpolator1D.BoundaryBehavior.LASTVAL)

        def englessRotMOI(self) -> float:
                return _sampleData["Rotational moment of inertia (kg·m²)"][-1]

        def thrustCurve(self) -> float:
                return 1DInterpolator(_sampleData["# Time (s)"],
                                      _sampleData["Thrust (N)"],
                                      Interpolator1D.BoundaryBehavior.ZEROVAL)



