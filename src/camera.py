import numpy as np

class AbstractCamera(object):
	"""docstring for AbstractCamera"""
	def __init__(self, width, height):
		self.width = width
		self.height = height

	def is_in_frame(self, uv, boundary=0):
		ll = np.array([boundary, boundary])  							# lower-left
		ur = np.array([self.width - boundary, self.height - boundary])  # upper-right
		return np.all(np.logical_and(ll <= uv, uv <= ur))

class PinholeCamera(AbstractCamera):
	"""docstring for PinholeCamera"""
	def __init__(self, width, height, fx, fy, cx, cy, k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
		super(PinholeCamera, self).__init__(width, height)
		self.fx = fx
		self.fy = fy
		self.cx = cx
		self.cy = cy
		
		self.is_distorted = abs(k1) > 0.0000001
		self.cvK = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1], dtype=float).reshape(3,3)
		self.cvD = np.array([k1, k2, p1, p2, k3], dtype=float)

	def get_focal_length(self):
		return abs(self.fx)

	def K(self):
		return np.array([
				[self.fx, 0, self.cx],
				[0, self.fy, self.cy],
				[0, 0, 1],
			])
		
	def cam2world(self, uv):
		if not self.is_distorted:
			x = (uv[0] - self.cx) / self.fx
			y = (uv[1] - self.cy) / self.fy
			z = 1.0
		else:
			# TODO: distorted
			pass
		v = np.array([x, y, z])
		return v / np.linalg.norm(v)

	def world2cam(self, pos):
		pos = np.asarray(pos, dtype=float)
		
		if len(pos) == 3:
			pos = pos[:2] / pos[2]
		
		if not self.is_distorted:
			x = self.fx * pos[0] + self.cx
			y = self.fy * pos[1] + self.cy
		else:
			# TODO: distorted
			pass

		return np.array([x, y])

	def compose_P(self, R, t):
	    RT = np.hstack((R, t))
	    return np.matmul(self.K(), RT)

	def undistort_image(self):
		# TODO
		pass