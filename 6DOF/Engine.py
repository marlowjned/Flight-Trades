# Engine.py
# Direct and RPA implementations

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from dataclasses import dataclass

import CustomInterpolator

if TYPE_CHECKING:
    import Rocket

# TEMPORARY, rpa implementations will flesh out this class significantly
@dataclass
class Engine:
    massData: Rocket.Rocket.MassComponent # if massData = None, just return 0 for any call
    thrust: CustomInterpolator.Interpolator1D