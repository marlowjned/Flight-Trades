# Environment.py
# Includes all wind, gust, and other environmental data

from __future__ import annotations
from abc import ABC, abstractmethod

from ambiance import Atmosphere
from typing import Union, Optional
import numpy as np

import Vector3D

# Base model for environment, then make another file for our current best wind model
class Environment:
	
	# make env data class
	
	
	# take in randomization seeds on instantiation
	def __init__(self):
		return
	

	"""
	simple_wind.py
	==============
	Minimal wind model for trajectory simulation.

	Three inputs per altitude level:
	- wind speed  (m/s)
	- wind direction (degrees from North, meteorological convention)
	- turbulence (% of wind speed, applied as white noise std dev)

	Usage
	-----
		from simple_wind import SimpleWindModel

		# Constant wind
		wm = SimpleWindModel(speed_ms=15.0, direction_deg=270.0, turbulence_pct=5.0)

		# Altitude-varying profile
		wm = SimpleWindModel(
			speed_ms    = [0, 10, 25, 40, 35, 20],   # m/s at each altitude
			direction_deg = [180, 210, 270, 280, 270, 260],
			turbulence_pct = 5.0,                      # scalar or per-altitude list
			altitudes_m  = [0, 2000, 5000, 10000, 15000, 20000],
		)

		# Get wind at a single altitude (one realisation)
		u, v = wm.at(altitude_m=10000)

		# Get a full profile for one Monte Carlo run (fixed seed = reproducible)
		u_arr, v_arr = wm.profile(altitudes_m=np.linspace(0, 20000, 200), seed=42)

		# Monte Carlo: draw N profiles
		profiles = wm.sample(n=500, altitudes_m=np.linspace(0, 20000, 200))
		# shape: (500, N_alt, 2)  — axis-2 is [u, v]
	"""

	class WindModel(ABC):

		@abstractmethod
		def windVector(self, altitude: float) -> Vector3D: ...
		# add any other necessary functions



	class SimpleWindModel:
		pass

	# Type alias for inputs that can be scalar or array
	Scalar_or_Array = Union[float, int, list, np.ndarray]


	class SimpleWindModel:
		"""
		Wind model defined by speed, direction, and turbulence percentage.

		Wind direction follows the **meteorological convention**:
		0° / 360° = wind coming FROM the North  (blows toward South → +u=0, -v)
		90°        = wind coming FROM the East   (blows toward West  → -u, +v=0)
		180°       = wind coming FROM the South  (blows toward North → +u=0, +v)
		270°       = wind coming FROM the West   (blows toward East  → +u, +v=0)

		The [u, v] components follow the NED (North-East-Down) convention:
		u = northward wind component  [m/s]
		v = eastward  wind component  [m/s]

		Turbulence is modelled as zero-mean Gaussian noise with
		std dev = (turbulence_pct / 100) * local_wind_speed

		Parameters
		----------
		speed_ms       : wind speed [m/s] — scalar or array aligned to altitudes_m
		direction_deg  : meteorological wind direction [deg] — scalar or array
		turbulence_pct : turbulence as % of wind speed — scalar or array
		altitudes_m    : altitude grid [m] — required if any input is an array;
						ignored if all inputs are scalars
		"""

		def __init__(
			self,
			speed_ms:       Scalar_or_Array,
			direction_deg:  Scalar_or_Array,
			turbulence_pct: Scalar_or_Array = 5.0,
			altitudes_m:    Optional[Scalar_or_Array] = None,
		):
			speed     = np.atleast_1d(np.asarray(speed_ms,       dtype=float))
			direction = np.atleast_1d(np.asarray(direction_deg,  dtype=float))
			turb      = np.atleast_1d(np.asarray(turbulence_pct, dtype=float))

			# If all scalars, work without an altitude grid
			scalar_mode = (
				speed.size == 1 and direction.size == 1 and turb.size == 1
				and altitudes_m is None
			)

			if scalar_mode:
				self._alts = np.array([0.0, 1e6])   # effectively constant everywhere
				self._speed = np.array([speed[0], speed[0]])
				self._dir   = np.array([direction[0], direction[0]])
				self._turb  = np.array([turb[0], turb[0]])
			else:
				if altitudes_m is None:
					raise ValueError(
						"altitudes_m is required when speed/direction/turbulence "
						"are arrays."
					)
				alts = np.asarray(altitudes_m, dtype=float)
				K = len(alts)
				assert np.all(np.diff(alts) > 0), "altitudes_m must be strictly ascending"

				# Broadcast scalars to full altitude grid
				self._alts  = alts
				self._speed = np.broadcast_to(speed,     (K,)).copy() if speed.size == 1     else speed
				self._dir   = np.broadcast_to(direction, (K,)).copy() if direction.size == 1 else direction
				self._turb  = np.broadcast_to(turb,      (K,)).copy() if turb.size == 1      else turb

			# Pre-compute mean [u, v] at every grid point
			# Meteorological convention: FROM direction → wind blows TOWARD (dir + 180)
			rad = np.deg2rad(self._dir + 180.0)
			self._mean_u = self._speed * np.cos(rad)   # northward
			self._mean_v = self._speed * np.sin(rad)   # eastward

		# ------------------------------------------------------------------
		# Query at a single altitude
		# ------------------------------------------------------------------

		def at(
			self,
			altitude_m: float,
			seed: Optional[int] = None,
		) -> tuple[float, float]:
			"""
			Return wind [u, v] at a single altitude, including turbulence.

			Parameters
			----------
			altitude_m : query altitude [m]
			seed       : optional RNG seed for reproducibility

			Returns
			-------
			u, v : northward and eastward wind [m/s]
			"""
			rng = np.random.default_rng(seed)

			u_mean = float(np.interp(altitude_m, self._alts, self._mean_u))
			v_mean = float(np.interp(altitude_m, self._alts, self._mean_v))
			sigma  = float(np.interp(altitude_m, self._alts,
									self._turb / 100.0 * self._speed))

			u = u_mean + rng.normal(0.0, sigma)
			v = v_mean + rng.normal(0.0, sigma)
			return u, v

		# ------------------------------------------------------------------
		# Full altitude profile for one Monte Carlo run
		# ------------------------------------------------------------------

		def profile(
			self,
			altitudes_m: np.ndarray,
			seed: Optional[int] = None,
		) -> tuple[np.ndarray, np.ndarray]:
			"""
			Return wind profile [u(z), v(z)] at requested altitudes.

			Turbulence is spatially correlated using a simple first-order
			Gauss-Markov process with a 2 km vertical correlation length —
			so the wind doesn't jump randomly at every altitude but varies
			smoothly, as real turbulence does.

			Parameters
			----------
			altitudes_m : (N,) altitudes to evaluate [m]
			seed        : optional RNG seed

			Returns
			-------
			u, v : (N,) wind components [m/s]
			"""
			rng  = np.random.default_rng(seed)
			alts = np.asarray(altitudes_m, dtype=float)
			N    = len(alts)

			# Interpolate mean profile onto query altitudes
			u_mean = np.interp(alts, self._alts, self._mean_u)
			v_mean = np.interp(alts, self._alts, self._mean_v)
			sigma  = np.interp(alts, self._alts, self._turb / 100.0 * self._speed)

			# Correlated turbulence via Ornstein-Uhlenbeck (altitude as the
			# independent variable instead of time).  Correlation length 2 km.
			L_vert = 2000.0   # m
			dz_arr = np.diff(alts, prepend=alts[0])   # altitude steps
			noise_u = rng.standard_normal(N)
			noise_v = rng.standard_normal(N)

			turb_u = np.zeros(N)
			turb_v = np.zeros(N)
			for i in range(N):
				dz   = abs(dz_arr[i]) if i > 0 else 0.0
				phi  = np.exp(-dz / L_vert)
				q    = np.sqrt(1.0 - phi**2)
				turb_u[i] = phi * (turb_u[i-1] if i > 0 else 0.0) + q * noise_u[i]
				turb_v[i] = phi * (turb_v[i-1] if i > 0 else 0.0) + q * noise_v[i]

			u = u_mean + sigma * turb_u
			v = v_mean + sigma * turb_v
			return u, v

		# ------------------------------------------------------------------
		# Monte Carlo: N independent profiles
		# ------------------------------------------------------------------

		def sample(
			self,
			n: int,
			altitudes_m: np.ndarray,
			seed: int = 0,
		) -> np.ndarray:
			"""
			Draw n independent wind profile realisations.

			Parameters
			----------
			n           : number of profiles
			altitudes_m : (N_alt,) altitude grid [m]
			seed        : master seed (each run gets a derived sub-seed)

			Returns
			-------
			profiles : (n, N_alt, 2)
					axis-0 = Monte Carlo index
					axis-1 = altitude index
					axis-2 = [u, v] [m/s]
			"""
			master = np.random.default_rng(seed)
			seeds  = master.integers(0, 2**31, size=n)
			alts   = np.asarray(altitudes_m, dtype=float)

			profiles = np.zeros((n, len(alts), 2))
			for i, s in enumerate(seeds):
				u, v = self.profile(alts, seed=int(s))
				profiles[i, :, 0] = u
				profiles[i, :, 1] = v
			return profiles
		

	def getBaseShear():
		pass

	@property
	def windVector(self) -> Vector3D.Vector3D:
		pass

	def step():
		pass

	@property
	def a(self):
		pass

	



