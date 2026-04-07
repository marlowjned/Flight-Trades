# CustomInterpolator.py
# Interpolator that handles out of bounds behavior
# (Defaults to null, zero, or the last value)

from enum import Enum, auto
from typing import Optional
import numpy as np

# TODO: add derivative method (Ilong_dot, Irot_dot)

class Interpolator1D:
    class BoundaryBehavior(Enum):
        NULLVAL = auto()
        ZEROVAL = auto()
        LASTVAL = auto()

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        boundary: "Interpolator1D.BoundaryBehavior" = None
    ):
        if len(x) != len(y):
            raise ValueError(f"x and y must have the same length (x length: {len(x)}, y length: {len(y)}")
        if len(x) < 2:
            raise ValueError("Need at least 2 data points")
        if not np.all(np.diff(x) > 0):
            raise ValueError("x must be strictly increasing")

        self._x = np.array(x, dtype=float)
        self._y = np.array(y, dtype=float)
        self._boundary = boundary if boundary is not None else self.BoundaryBehavior.NULLVAL

    def query(self, x_query: float) -> Optional[float]:
        if x_query < self._x[0] or x_query > self._x[-1]:
            return self._out_of_bounds(x_query)
        return float(np.interp(x_query, self._x, self._y))

    def query_array(self, x_queries: np.ndarray) -> np.ndarray:
        x_queries = np.asarray(x_queries, dtype=float)
        results = np.interp(x_queries, self._x, self._y)

        out_of_bounds = (x_queries < self._x[0]) | (x_queries > self._x[-1])
        if np.any(out_of_bounds):
            if self._boundary == self.BoundaryBehavior.NULLVAL:
                results = results.astype(object)
                results[out_of_bounds] = None
            elif self._boundary == self.BoundaryBehavior.ZEROVAL:
                results[out_of_bounds] = 0.0
            elif self._boundary == self.BoundaryBehavior.LASTVAL:
                results[x_queries < self._x[0]]  = self._y[0]
                results[x_queries > self._x[-1]] = self._y[-1]

        return results

    def _out_of_bounds(self, x_query: float) -> Optional[float]:
        if self._boundary == self.BoundaryBehavior.NULLVAL:
            return None
        elif self._boundary == self.BoundaryBehavior.ZEROVAL:
            return 0.0
        elif self._boundary == self.BoundaryBehavior.LASTVAL:
            return float(self._y[0] if x_query < self._x[0] else self._y[-1])

    def derivative(self) -> "Interpolator1D":
        x_mid = (self._x[:-1] + self._x[1:]) / 2
        y_deriv = np.diff(self._y) / np.diff(self._x)
        deriv_boundary = (self.BoundaryBehavior.ZEROVAL
                          if self._boundary == self.BoundaryBehavior.LASTVAL
                          else self._boundary)
        return Interpolator1D(x_mid, y_deriv, deriv_boundary)

    @property
    def x_bounds(self) -> tuple[float, float]:
        return (self._x[0], self._x[-1])

    @property
    def boundary(self) -> "Interpolator1D.BoundaryBehavior":
        return self._boundary
	

import numpy as np
from typing import Optional

class Interpolator2D:

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ):
        """
        Args:
            x: shape (m,) — first axis (e.g. Mach)
            y: shape (n,) — second axis (e.g. AoA)
            z: shape (m, n) — values at each (x, y) pair
        """
        if not np.all(np.diff(x) > 0):
            raise ValueError("x must be strictly increasing")
        if not np.all(np.diff(y) > 0):
            raise ValueError("y must be strictly increasing")
        if z.shape != (len(x), len(y)):
            raise ValueError(f"z must have shape ({len(x)}, {len(y)}), got {z.shape}")

        self._x = np.array(x, dtype=float)
        self._y = np.array(y, dtype=float)
        self._z = np.array(z, dtype=float)

    def query(self, x_query: float, y_query: float) -> Optional[float]:
        if (x_query < self._x[0] or x_query > self._x[-1] or
            y_query < self._y[0] or y_query > self._y[-1]):
            return None

        i = int(np.clip(np.searchsorted(self._x, x_query, side="right") - 1, 0, len(self._x) - 2))
        j = int(np.clip(np.searchsorted(self._y, y_query, side="right") - 1, 0, len(self._y) - 2))

        tx = (x_query - self._x[i]) / (self._x[i+1] - self._x[i])
        ty = (y_query - self._y[j]) / (self._y[j+1] - self._y[j])

        return (self._z[i,   j  ] * (1 - tx) * (1 - ty) +
                self._z[i+1, j  ] *      tx  * (1 - ty) +
                self._z[i,   j+1] * (1 - tx) *      ty  +
                self._z[i+1, j+1] *      tx  *      ty  )

    def query_array(self, x_queries: np.ndarray, y_queries: np.ndarray) -> np.ndarray:
        """Returns object array — entries are float or None for out-of-bounds."""
        x_queries = np.asarray(x_queries, dtype=float)
        y_queries = np.asarray(y_queries, dtype=float)
        if x_queries.shape != y_queries.shape:
            raise ValueError("x_queries and y_queries must have the same shape")

        results = np.empty(x_queries.shape, dtype=object)
        for idx in np.ndindex(x_queries.shape):
            results[idx] = self.query(x_queries[idx], y_queries[idx])

        return results

    @property
    def x_bounds(self) -> tuple[float, float]:
        return (self._x[0], self._x[-1])

    @property
    def y_bounds(self) -> tuple[float, float]:
        return (self._y[0], self._y[-1])    


