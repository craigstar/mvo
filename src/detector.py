import numpy as np
import math
import cv2

from .feature import Feature

class AbstractDetector(object):
	"""docstring for AbstractDetector"""
	def __init__(self, width, height, cell_size, pyr_levels):
		super(AbstractDetector, self).__init__()
		self.cell_size = cell_size
		self.pyr_levels = pyr_levels
		self.grid_cols = math.floor(width / cell_size) + 1
		self.grid_rows = math.floor(height / cell_size) + 1
		self.grid_occupancy = np.full(self.grid_cols * self.grid_rows, False)

	def reset_grid(self):
		self.grid_occupancy.fill(False)


class Corner(object):
	"""docstring for Corner"""
	def __init__(self, x, y, score, level):
		self.x = x
		self.y = y
		self.score = score
		self.level = level
		

class FastDetector(AbstractDetector):
	"""docstring for FastDetector"""
	def __init__(self, width, height, cell_size, pyr_levels):
		super(FastDetector, self).__init__(width, height, 
										   cell_size, pyr_levels)
		self.fast = cv2.FastFeatureDetector_create(20)

	def detect(self, frame, img_pyr, quality_level):
		grid_shape = (self.grid_rows, self.grid_cols)
		n_grids = self.grid_cols * self.grid_rows
		corners = [Corner(0, 0, 0, 0) for i in range(n_grids)]
		best_eigs = []
		fts = []

		for level in range(self.pyr_levels):
			scale = 2 ** level
			eigs = cv2.cornerMinEigenVal(img_pyr[0], 3)
			best_eigs.append(eigs.max())
			kps = self.fast.detect(img_pyr[level], None)

			for kp in kps:
				x, y = kp.pt
				x, y = int(x), int(y)
				row_idx = math.floor(y * scale / self.cell_size)
				col_idx = math.floor(x * scale / self.cell_size)
				k = np.ravel_multi_index([row_idx, col_idx], grid_shape)
				if self.grid_occupancy[k]:
					continue
				score = eigs[y, x]
				if score > corners[k].score:
					corners[k] = Corner(x * scale, y * scale, score, level)

		for c in corners:
			if c.score > quality_level * best_eigs[c.level]:
				fts.append(Feature(frame, np.array([c.x, c.y]), c.level))
		self.reset_grid()
		return fts


class GoodFeaturesDetector(AbstractDetector):
	"""docstring for GoodFeaturesDetector"""
	def __init__(self, width, height, cell_size, pyr_levels):
		super(GoodFeaturesDetector, self).__init__(width, height, 
										   cell_size, pyr_levels)

	def detect(self, frame, img_pyr, quality_level):
		grid_shape = (self.grid_rows, self.grid_cols)
		n_grids = self.grid_cols * self.grid_rows
		corners = [Corner(0, 0, 0, 0) for i in range(n_grids)]
		best_eigs = []
		fts = []
		
		for level in range(self.pyr_levels):
			scale = 2 ** level
			eigs = cv2.cornerMinEigenVal(img_pyr[0], 3)
			best_eigs.append(eigs.max())
			kps = cv2.goodFeaturesToTrack(img_pyr[level], 1000, 0.01, 10)

			for kp in kps:
				x, y = kp[0].astype(int)
				row_idx = math.floor(y * scale / self.cell_size)
				col_idx = math.floor(x * scale / self.cell_size)
				k = np.ravel_multi_index([row_idx, col_idx], grid_shape)
				if self.grid_occupancy[k]:
					continue
				score = eigs[y, x]
				if score > corners[k].score:
					corners[k] = Corner(x * scale, y * scale, score, level)

		for c in corners:
			if c.score > quality_level * best_eigs[c.level]:
				fts.append(Feature(frame, np.array([c.x, c.y]), c.level))
		self.reset_grid()
		return fts


class SiftDetector(AbstractDetector):
	"""docstring for SiftDetector"""
	def __init__(self, width, height, cell_size, pyr_levels):
		super(SiftDetector, self).__init__(width, height, 
										   cell_size, pyr_levels)
		self.sift = cv2.xfeatures2d.SIFT_create()

	def detect(self, frame, img_pyr, quality_level):
		grid_shape = (self.grid_rows, self.grid_cols)
		n_grids = self.grid_cols * self.grid_rows
		corners = [Corner(0, 0, 0, 0) for i in range(n_grids)]
		best_eigs = []
		fts = []

		for level in range(self.pyr_levels):
			scale = 2 ** level
			eigs = cv2.cornerMinEigenVal(img_pyr[0], 3)
			best_eigs.append(eigs.max())
			kps = self.sift.detect(img_pyr[level], None)

			for kp in kps:
				x, y = kp.pt
				x, y = int(x), int(y)
				row_idx = math.floor(y * scale / self.cell_size)
				col_idx = math.floor(x * scale / self.cell_size)
				k = np.ravel_multi_index([row_idx, col_idx], grid_shape)
				if self.grid_occupancy[k]:
					continue
				score = eigs[y, x]
				if score > corners[k].score:
					corners[k] = Corner(x * scale, y * scale, score, level)

		for c in corners:
			if c.score > quality_level * best_eigs[c.level]:
				fts.append(Feature(frame, np.array([c.x, c.y]), c.level))
		self.reset_grid()
		return fts