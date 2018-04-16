import numpy as np

class Feature(object):
	"""docstring for Feature"""
	CORNER = 0

	FeatureType = {
		CORNER: 'corner features'
	}

	def __init__(self, frame, uv, level=0, pt3d=None, direction=np.zeros(0)):
		self.type = Feature.CORNER 	# feature type
		self.frame = frame			# reference frame where feature locates
		self.uv = uv				# (0, 0), type 'int', position of point in image
		self.level = level			# the pyrimid level of feature point
		self.point = pt3d			# Point3d object or None
		self.direction = direction if len(direction) else frame.c2f(uv) # feature point direction, (0, 0, 0)

		