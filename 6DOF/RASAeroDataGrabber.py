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
        for var in _aeroVars:
            _tempArray = np.zeros(len(self.machVals), len(self.alphaVals))
            
            for coeff in self._rasData[var]:
                # Find corresponding Mach Alpha location
                idxMach, idxAlpha = self._valuePosition(coeff)
                
                # Place in 2D array
                _tempArray[idxMach][idxAlpha] = coeff

            interpMap[var] = CustomInterpolator.Interpolator2D(self.machVals, self.alphaVals, _tempArray)

    # TODO: fix these bottom two
    def _valuePosition(self, coeff):
        idx = self._rasData[var].index(coeff)
        mach = self._rasData['Mach'][idx]
        alpha = self._rasData['Alpha'][idx]

        return [self.machVals.index(mach), self.alphaVals.index(alpha)]


    def coeffTable(self, varName: str) -> CustomInterpolator.Interpolator2D:
        return self.interpMap[varName]
    
    def getCoeffs(self, mach: float, alpha: float):
        return [self.interpMap[var] for var in self._aeroVars]





