#!/usr/bin/env python
import numpy as np
import cv2
import time

from src import Frame, PinholeCamera, FastDetector, GoodFeaturesDetector, SiftDetector
import src

####### 0.0 params #######
FAST = 0
SIFT = 1
GOOD_FEATURES = 2

Dectors = {
	FAST: 'fast',
	SIFT: 'sift',
	GOOD_FEATURES: 'good features'
}

detector = GOOD_FEATURES
draw_layers = True
use_grid = True

name = Dectors[detector]
print('using: ' + name)

####### 1.0 read image #######
# img = cv2.imread('src/imgs/chessboard.jpg')
img = cv2.imread('imgs/left.JPG')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rows, cols = img_gray.shape
print 'image shape:', img_gray.shape


####### 2.0 create camera, frame and detector #######
cam = PinholeCamera(width=cols, height=rows,
					fx=315.5, fy=315.5, 
					cx=376.0, cy=240.0
					)
frame = Frame(cam, img_gray, 0.0)
if detector == FAST:
	d = FastDetector(width=cols, height=rows, cell_size=25, pyr_levels=3)
elif detector == SIFT:
	d = SiftDetector(width=cols, height=rows, cell_size=25, pyr_levels=3)
else:
	d = GoodFeaturesDetector(width=cols, height=rows, cell_size=25, pyr_levels=3)


####### 3.0 detect features #######
s = time.time()
features = d.detect(frame, frame.img_pyr, 0.01)
e = time.time()
print 'total time:', e - s
print 'total features:', len(features)


####### 4.0 draw grid on images #######
imgs = src.utils.create_img_pyramid(img, 3)
if use_grid:
	imgs = src.utils.draw_grid(imgs, 25)


####### 5.0 draw points on each layer #######
if draw_layers:
	for f in features:
		scale = 2 ** f.level
		cv2.circle(imgs[f.level], tuple(f.xy / scale), 2, (0, 255, 0), 1)
	imgs[0] = cv2.resize(imgs[0], (0, 0), fx=2, fy=2)
	imgs[1] = cv2.resize(imgs[1], (0, 0), fx=2, fy=2)
	imgs[2] = cv2.resize(imgs[2], (0, 0), fx=2, fy=2)
	cv2.imshow(name + ' layer 1', imgs[0])
	cv2.imshow(name + ' layer 2', imgs[1])
	cv2.imshow(name + ' layer 3', imgs[2])


####### 6.0 draw all points in base layer #######
for f in features:
	cv2.circle(img, tuple(f.xy), 2 * (f.level + 1), (0, 255, 0), 1)
img = cv2.resize(img, (0, 0), fx=2, fy=2)
cv2.imshow(name, img)

cv2.waitKey(0)

