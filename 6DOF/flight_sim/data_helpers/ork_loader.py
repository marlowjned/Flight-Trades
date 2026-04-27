# ork_loader.py
# Initializes rocket data from an ORK csv

import pandas as pd

from flight_sim.data_helpers.custom_interpolator import Interpolator1D


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Replace non-ASCII characters in column names."""
    df.columns = (df.columns
                  .str.replace('\u00b7', '*', regex=False)   # middle dot -> *
                  .str.replace('\u00b2', '2', regex=False))  # superscript 2 -> 2
    return df


class ORKLoader:
    def __init__(self, csv_name: str):
        self._data = _normalize_columns(pd.read_csv(csv_name))

    def net_mass_curve(self) -> Interpolator1D:
        return Interpolator1D(self._data["# Time (s)"],
                              self._data["Mass (g)"],
                              Interpolator1D.BoundaryBehavior.LASTVAL)

    def engineless_mass(self) -> float:
        return self._data["Mass (g)"].iloc[-1]

    def net_cg_curve(self) -> Interpolator1D:
        return Interpolator1D(self._data["# Time (s)"],
                              self._data["CG location (cm)"],
                              Interpolator1D.BoundaryBehavior.LASTVAL)

    def engineless_cg(self) -> float:
        return self._data["CG location (cm)"].iloc[-1]

    def long_moi_curve(self) -> Interpolator1D:
        return Interpolator1D(self._data["# Time (s)"],
                              self._data["Longitudinal moment of inertia (kg*m2)"],
                              Interpolator1D.BoundaryBehavior.LASTVAL)

    def engineless_long_moi(self) -> float:
        return self._data["Longitudinal moment of inertia (kg*m2)"].iloc[-1]

    def rot_moi_curve(self) -> Interpolator1D:
        return Interpolator1D(self._data["# Time (s)"],
                              self._data["Rotational moment of inertia (kg*m2)"],
                              Interpolator1D.BoundaryBehavior.LASTVAL)

    def engineless_rot_moi(self) -> float:
        return self._data["Rotational moment of inertia (kg*m2)"].iloc[-1]

    def thrust_curve(self) -> Interpolator1D:
        return Interpolator1D(self._data["# Time (s)"],
                              self._data["Thrust (N)"],
                              Interpolator1D.BoundaryBehavior.ZEROVAL)
