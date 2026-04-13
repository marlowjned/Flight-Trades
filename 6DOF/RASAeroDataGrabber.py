# RasAeroDataGrabber.py
# Aero coefficient tables from RasAero

import numpy as np
import pandas as pd
from enum import Enum, auto
from typing import Optional

import CustomInterpolator


class RasAero:

    class Frame(Enum):
        WORLD = auto()
        BODY = auto()

    def __init__(self, csvName: str, frame: Frame = Frame.WORLD):
        self._rasData = pd.read_csv(csvName)
        if (frame == self.Frame.WORLD):
            self._aeroVars = ['CD Power-Off', 'CD Power-On', 'CL', 'CP']
        else:
            self._aeroVars = ['CA Power-Off', 'CA Power-On', 'CN', 'CP']
        
        # Find all unique Mach and Alpha values
        self.machVals = sorted(set(self._rasData['Mach']))
        self.alphaVals = sorted(set(self._rasData['Alpha']))
        
        self.interpMap = {}
        
        # Fill in array with values based on indices in Mach/Alpha
        for var in self._aeroVars:
            _tempArray = np.zeros((len(self.machVals), len(self.alphaVals)))
            
            for idx, coeff in enumerate(self._rasData[var]):
                # Find corresponding Mach Alpha location
                idxMach, idxAlpha = self._valuePosition(var, idx)

                # Place in 2D array
                _tempArray[idxMach][idxAlpha] = coeff

            self.interpMap[var] = CustomInterpolator.Interpolator2D(self.machVals, self.alphaVals, _tempArray,
                                                                      CustomInterpolator.Interpolator2D.BoundaryBehavior.LASTVAL)

    def _valuePosition(self, var: str, idx: int):
        mach = self._rasData['Mach'].iloc[idx]
        alpha = self._rasData['Alpha'].iloc[idx]

        return [self.machVals.index(mach), self.alphaVals.index(alpha)]


    def coeffTable(self, varName: str) -> CustomInterpolator.Interpolator2D:
        return self.interpMap[varName]
    
    def getCoeffs(self, mach: float, alpha_rad: float):
        alpha_deg = np.degrees(alpha_rad)
        coeffs = [self.interpMap[var].query(mach, alpha_deg) for var in self._aeroVars]
        coeffs[-1] *= 0.0254  # CP: inches -> meters
        return coeffs





