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

    def __init__(self, csvName: str, frame: Frame = self.Frame.WORLD):
        self._rasData = pd.read_csv(csvName)
        if (frame == Frame.WORLD):
            _aeroVars = ['CD Power-Off', 'CD Power-On', 'CL', 'CP']
        else:
            _aeroVars = ['CA Power-Off'	'CA Power-On', 'CN', 'CP']
        
        # Find all unique Mach and Alpha values
        self.machVals = sorted(set(_rasData['Mach']))
        self.alphaVals = sorted(set(_rasData['Alpha']))
        
        interpMap = {}
        
        # Fill in array with values based on indices in Mach/Alpha
        for var in _aeroVars:
            _tempArray = np.zeros(len(machVals), len(alphaVals))
            
            for coeff in _rasData[var]:
                # Find corresponding Mach Alpha location
                idxMach, idxAlpha = _valuePosition(coeff)
                
                # Place in 2D array
                _tempArray[idxMach][idxAlpha] = coeff

            interpMap[var] = Interpolator2D(machVals, alphaVals, _tempArray)

    def _valuePosition(self, value):
        idx = _rasData[var].index(coeff)
        mach = _rasData['Mach'][idx]
        alpha = _rasData['Alpha'][idx]

        return [machVals.index(mach), alphaVals.index(alpha)]


    def coeffTable(self, varName: str) -> Interpolator2D:
        return interpMap[varName]





