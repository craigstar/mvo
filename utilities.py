import numpy as np
import cv2

def draw_grid(img_pyr, cell_size):
	''' draw grid on a list of images'''
	imgs = list(img_pyr)
	height, width = imgs[0].shape[0], imgs[0].shape[1]
	for level, img in enumerate(imgs):
		scale = 2 ** level
		n_rows = int(height / cell_size) + 1
		n_cols = int(width / cell_size) + 1

		for i in range(1, n_rows):
			start = (0, i * cell_size / scale)
			end = (width, i * cell_size / scale)
			cv2.line(img, start, end, (255, 0, 0), 1, 1)
		for j in range(1, n_cols):
			start = (j * cell_size / scale, 0)
			end = (j * cell_size / scale, height)
			cv2.line(img, start, end, (255, 0, 0), 1, 1)
	return imgs


def create_img_pyramid(img_level_0, levels):
	def _half_sample(img):
		rows, cols = img.shape[0], img.shape[1]
		rows = int(rows / 2)
		cols = int(cols / 2)
		return cv2.resize(img, (cols, rows))

	img_pyr = []
	img_pyr.append(img_level_0.copy())
	for i in range(levels):
		img_pyr.append(_half_sample(img_pyr[-1]))
	return img_pyr

	