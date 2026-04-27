# vector3d.py
# 3D vector with body/world frame conversions

import numpy as np

class Vector3D:

	# Default vector is zero vector with orientation +z
	def __init__(self, elements: np.ndarray = np.zeros(3),
                     dcm: np.ndarray = None,
                     is_body_frame: bool = False):
		self.elements = elements
		self.dcm = dcm  # World to body
		self.dcm_inv = np.linalg.inv(dcm) if dcm is not None else None
		self.is_body_frame = is_body_frame  # Assumes world frame otherwise

	@property
	def vector_world(self):
		return ((self.elements @ self.dcm_inv) if self.is_body_frame else self.elements)

	@property
	def vector_body(self):
		return (self.elements if self.is_body_frame else (self.elements @ self.dcm))

	@property
	def magnitude(self):
		return np.linalg.norm(self.elements)

	@property
	def normalized(self):
		mag = self.magnitude
		if mag == 0:
			return Vector3D(np.zeros(3), self.dcm, self.is_body_frame)
		return Vector3D(self.elements / mag, self.dcm, self.is_body_frame)

	@property
	def x(self): return self.elements[0]
	@property
	def y(self): return self.elements[1]
	@property
	def z(self): return self.elements[2]
