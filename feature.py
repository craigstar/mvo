class Feature(object):
	"""docstring for Feature"""
	CORNER = 0

	FeatureType = {
		CORNER: 'corner features'
	}

	def __init__(self, frame, xy, level):
		self.type = self.CORNER
		self.frame = frame
		self.xy = xy
		self.level = level
		self.point = None
		