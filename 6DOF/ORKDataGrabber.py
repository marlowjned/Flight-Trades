# ORKDataGrabber.py
# Initializes rocket data from an ORK csv


import pandas as pd

import CustomInterpolator


# make @ property ?

class ORKDataGrabber:
    def __init__(self, csvName: str):
        self._sampleData = pd.read_csv(csvName)

    # Mass (g)
    def netMassCurve(self) -> CustomInterpolator.Interpolator1D:
        return CustomInterpolator.Interpolator1D(self._sampleData["# Time (s)"],
                                                 self._sampleData["Mass (g)"],
                                                 CustomInterpolator.Interpolator1D.BoundaryBehavior.LASTVAL)

    def englessMass(self) -> float:
        return self._sampleData["Mass (g)"][-1]

    # CG (from tip, cm)
    def netCGCurve(self) -> CustomInterpolator.Interpolator1D:
        return CustomInterpolator.Interpolator1D(self._sampleData["# Time (s)"],
                                                 self._sampleData["CG location (cm)"],
                                                 CustomInterpolator.Interpolator1D.BoundaryBehavior.LASTVAL)

    def englessCG(self) -> float: # TODO: check this
        return self._sampleData["CG location (cm)"][-1]

    # Longitudinal and rotational MOI (kg m^2)
    # Construct inertia tensor in Aerodynamics.py
    # TODO: may need to preprocess by only permitting ASCII characters
    # TODO: final moi?
    def longMOICurve(self) -> CustomInterpolator.Interpolator1D:
        return CustomInterpolator.Interpolator1D(self._sampleData["# Time (s)"],
                                                 self._sampleData["Longitudinal moment of inertia (kg·m²)"],
                                                 CustomInterpolator.Interpolator1D.BoundaryBehavior.LASTVAL)

    def englessLongMOI(self) -> float:
        return self._sampleData["Longitudinal moment of inertia (kg·m²)"][-1]

    def rotMOICurve(self) -> CustomInterpolator.Interpolator1D:
        return CustomInterpolator.Interpolator1D(self._sampleData["# Time (s)"],
                                                 self._sampleData["Rotational moment of inertia (kg·m²)"],
                                                 CustomInterpolator.Interpolator1D.BoundaryBehavior.LASTVAL)

    def englessRotMOI(self) -> float:
        return self._sampleData["Rotational moment of inertia (kg·m²)"][-1]

    def thrustCurve(self) -> CustomInterpolator.Interpolator1D:
        return CustomInterpolator.Interpolator1D(self._sampleData["# Time (s)"],
                                                 self._sampleData["Thrust (N)"],
                                                 CustomInterpolator.Interpolator1D.BoundaryBehavior.ZEROVAL)
