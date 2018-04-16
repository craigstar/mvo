import numpy as np
import cv2
import itertools

import sophus as sp

class Frame(object):
	"""docstring for Frame"""
	id_generator = itertools.count(0)

	ERROR_IMG = "Frame: provided image has not the same size as the camera mod " \
				"el or image is not grayscale"

	def __init__(self, cam, img, timestamp):
		self.cam = cam
		self.img = img
		self.timestamp = timestamp
		self.img_pyr = []
		self.id = next(self.id_generator)
		self.features = []
		self.T_from_w = sp.SE3()
		self._init_frame(img)

	def _init_frame(self, img):
		height, width = img.shape
		if (not img.size or img.dtype != np.uint8 or
			width != self.cam.width or height != self.cam.height):
			# check image size and type
			raise Exception(self.ERROR_IMG)
		# TODO: use cv2.buildOpticalFlowPyramid instead
		self._create_img_pyramid(img, 5)

	def _create_img_pyramid(self, img_level_0, levels):
		self.img_pyr.append(img_level_0)
		for i in range(levels):
			self.img_pyr.append(self._half_sample(self.img_pyr[-1]))

	def _half_sample(self, img):
		rows, cols = img.shape
		rows = int(rows / 2)
		cols = int(cols / 2)
		return cv2.resize(img, (cols, rows))

	def add_feature(self, feature):
		self.features.append(feature)

	def pos(self):
		return self.T_from_w.inverse().translation()
		
	def c2f(self, uv):
		# convert camera 2d corrdinate to camera 3d
		return self.cam.cam2world(uv)

	def f2c(self, xyz):
		# convert camera 3d corrdinate to camera 2d
		return self.cam.world2cam(xyz)