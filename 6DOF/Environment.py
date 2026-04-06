# Environment.py
# Includes all wind, gust, and other environmental data

from ambiance import Atmosphere

import Vector3D

# Base model for environment, then make another file for our current best wind model
class Environment:
	# take in randomization seeds on instantiation
	def __init__(self):
		return
	
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

	



