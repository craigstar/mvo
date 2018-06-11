import itertools

class Point3d(object):
	"""class of 3D Point"""
	id_generator = itertools.count(0)

	def __init__(self, pos):
		self.id = next(self.id_generator)   # unique id
		self.pos = pos                      # 3d [x, y, z] position
		self.obs = []                       # observed features that match this 3d point
		self.last_project_kf_id = 0;		# reprojection signiture, do not reproject twice

	def __str__(self):
		return str(self.pos)

	def add_frame_ref(self, feature):
		self.obs.append(feature)

	def get_close_view_obs(self, frame_pos):
		obs_dir = (frame_pos - self.pos)
		obs_dir /= np.linalg.norm(obs_dir)
		min_cos_angle = 0
		min_ft = self.obs[0]

		for ft in self.obs:
			direction = ft.frame.pos() - self.pos
			direction /= np.linalg.norm(direction)

			# dot product of unit vectors is cosine
			cos_angle = direction.dot(obs_dir)
			if cos_angle > min_cos_angle:
				min_cos_angle = cos_angle
				min_ft = ft

		if min_cos_angle < 0.5:				# angle greater than 60 degree is useless
			return (False, min_ft)
		return (True, min_ft)