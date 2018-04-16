import itertools

class Point3d(object):
	"""class of 3D Point"""
	id_generator = itertools.count(0)

	def __init__(self, pos):
		self.id = next(self.id_generator)     # unique id
		self.pos = pos                        # 3d [x, y, z] position
		self.fts = []                         # features that match this 3d point

	def add_frame_ref(self, feature):
		self.fts.append(feature)
		