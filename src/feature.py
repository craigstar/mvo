class Feature(object):
	"""docstring for Feature"""
	CORNER = 0

	FeatureType = {
		CORNER: 'corner features'
	}

	def __init__(self, frame, uv, level):
		self.type = Feature.CORNER 	# feature type
		self.frame = frame			# reference frame where feature locates
		self.uv = uv				# (0, 0), type 'int', position of point in image
		self.level = level			# the pyrimid level of feature point
		self.direction = frame.cam.cam2world(uv) # feature point direction, (0, 0, 0)

		