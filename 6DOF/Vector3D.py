# Vector3D.py
# 3D vector with body/world frame conversions

import numpy as np

class Vector3D:

	# Default vector is zero vector with orientation +z
	def __init__(self, elements: np.ndarray = np.zeros(3),
                     dcm: np.ndarray = None, # TODO: add stuff s.t. transform just won't be allowed
                     isBodyFrame: bool = False):
		self.elements = elements
		self.DCM = dcm # Body to world
		self.DCM_inv = np.linalg.inv(dcm) if dcm is not None else None
		self.isBodyFrame = isBodyFrame # Assumes world frame otherwise
        
	@property
	def vectorWorld(self):
		return ((self.elements @ self.DCM) if self.isBodyFrame else self.elements)

	@property
	def vectorBody(self):
		return (self.elements if self.isBodyFrame else (self.elements @ self.DCM_inv))
	
	@property
	def magnitude(self):
		return np.linalg.norm(self.elements)
	
	@property
	def normalized(self):
		mag = self.magnitude
		return Vector3D(self.elements / mag, self.DCM, self.isBodyFrame)
	
	# rework these
	@property
	def x(self): return self.elements[0]
	@property
	def y(self): return self.elements[1]
	@property
	def z(self): return self.elements[2]

