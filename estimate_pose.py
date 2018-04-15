#!/usr/bin/env python

import numpy as np
import cv2
import time

from src import Frame, PinholeCamera, Initialization
import src


####### 1.0 read images #######
img0 = cv2.imread('../../dataset/sequences/00/image_0/000000.png')
img1 = cv2.imread('../../dataset/sequences/00/image_0/000001.png')
gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
rows, cols = gray0.shape
print('image shape:', gray0.shape)


####### 2.0 create camera, frames #######
cam = PinholeCamera(width=cols, height=rows,
                    fx=718.856, fy=718.856, 
                    cx=607.1928, cy=185.2157
                    )

first_frame = Frame(cam, gray0, 0.0)
second_frame = Frame(cam, gray1, 0.0)


####### 3.0 initialize pose estimator #######
init = Initialization()
init.add_first_frame(first_frame)
init.add_second_frame(second_frame)
# print(init.dir_ref)