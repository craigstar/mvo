from .detector import GoodFeaturesDetector

class Initialization(object):
	"""docstring for Initialization"""

	SUCCESS = 0
	FAILURE = 1
	NO_KEYFRAME = 2

	InitResult = {
		FAILURE: 'init failed',
		NO_KEYFRAME: 'no key frame',
		SUCCESS: 'init succeed'
	}

	def __init__(self):
		self.pos_ref = []		# position of reference feature points
		self.pos_cur = []		# position of current feature points
		self.dirs = []			# directions of feature points
		self.frame_ref = None	# reference frame

	def _reset(self):
		self.pos_cur = []
		self.frame_ref = None

	def add_first_frame(self, frame):
		self._reset()
		self.pos_ref, self.dirs = detect_features(frame);
		if (len(pos_ref) < 100):
			print('First image has less than 100 features.' \
				  ' Retry in more textured environment.')
			return Initialization.FAILURE

		self.frame_ref = frame
		self.pos_cur = list(self.pos_ref) # make a copy of pos_ref
		return Initialization.SUCCESS
		
	def detect_features(self, frame):
		rows, cols = frame.img.shape
		d = GoodFeaturesDetector(width=cols, height=rows,
								 cell_size=25, pyr_levels=3)
		fts = d.detect(frame, frame.img_pyr, 0.01)
		
		positions = []
		directions = []
		for ft in fts:
			positions.append(ft.uv)
			directions.append(ft.direction)
		return (positions, directions)

	def track_klt(self):
		pass
