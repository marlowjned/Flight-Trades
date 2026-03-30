# Vector3D.py
# 3D vector with body/world frame conversions

import numpy as np

class Vector3D:

	def __init__(self, elements: np.ndarray,
                     dcm: np.ndarray = None, # TODO: add stuff s.t. transform just won't be allowed
                     isBodyFrame: bool = False):
		self.Vector3 = elements
		self.DCM = dcm # Body to world
		self.DCM_inv = np.linalg.inv(dcm) # World to body
		self.IsBodyFrame = isBodyFrame # Assumes world frame otherwise
        
	@property
	def vectorWorld(self):
		return ((self.Vector3 @ self.DCM) if isBodyFrame else self.Vector3)

	@property
	def vectorBody(self):
		return (self.Vector3 if isBodyFrame else (self.Vector3 @ self.DCM_inv))

