# rasaero_loader.py
# Aero coefficient tables from RasAero

import numpy as np
import pandas as pd
from enum import Enum, auto

from flight_sim.data_helpers.custom_interpolator import Interpolator2D


class RasAeroLoader:

    class Frame(Enum):
        WORLD = auto()
        BODY  = auto()

    def __init__(self, csv_name: str, frame: Frame = Frame.WORLD):
        self._ras_data = pd.read_csv(csv_name)
        if frame == self.Frame.WORLD:
            self._aero_vars = ['CD Power-Off', 'CD Power-On', 'CL', 'CP']
        else:
            self._aero_vars = ['CA Power-Off', 'CA Power-On', 'CN', 'CP']

        self.mach_vals  = sorted(set(self._ras_data['Mach']))
        self.alpha_vals = sorted(set(self._ras_data['Alpha']))

        self.interp_map = {}

        for var in self._aero_vars:
            _temp_array = np.zeros((len(self.mach_vals), len(self.alpha_vals)))

            for idx, coeff in enumerate(self._ras_data[var]):
                idx_mach, idx_alpha = self._value_position(var, idx)
                _temp_array[idx_mach][idx_alpha] = coeff

            self.interp_map[var] = Interpolator2D(
                self.mach_vals, self.alpha_vals, _temp_array,
                Interpolator2D.BoundaryBehavior.LASTVAL,
            )

    def _value_position(self, var: str, idx: int):
        mach  = self._ras_data['Mach'].iloc[idx]
        alpha = self._ras_data['Alpha'].iloc[idx]
        return [self.mach_vals.index(mach), self.alpha_vals.index(alpha)]

    def coeff_table(self, var_name: str) -> Interpolator2D:
        return self.interp_map[var_name]

    def get_coeffs(self, mach: float, alpha_rad: float):
        alpha_deg = np.degrees(alpha_rad)
        coeffs = [self.interp_map[var].query(mach, alpha_deg) for var in self._aero_vars]
        coeffs[-1] *= 0.0254  # CP: inches -> meters
        return coeffs
