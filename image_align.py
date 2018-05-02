import numpy as np
import cv2
import time

from src import Frame, PinholeCamera, SparseImgAlign
import src

def getName(k):
    name0 = "%06d" % k
    name1 = "%06d" % (k+1)
    name2 = "%06d" % (k+2)
    sufix = '.png'
    return (name0 + sufix, name1 + sufix, name2 + sufix)


path = '../../dataset/sequences/00/image_0/'
init = src.Initialization()

####### 1.0 read images #######
name0, name1, name2 = getName(0)
img0 = cv2.imread(path + name0)
img1 = cv2.imread(path + name1)
img2 = cv2.imread(path + name2)
gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
rows, cols = gray0.shape

####### 2.0 create camera, frames #######
cam = PinholeCamera(width=cols, height=rows,
                    fx=718.856, fy=718.856, 
                    cx=607.1928, cy=185.2157
                    )

first_frame = Frame(cam, gray0, 0.0)
second_frame = Frame(cam, gray1, 0.0)
third_frame = Frame(cam, gray2, 0.0)

init.add_first_frame(first_frame)
init.add_second_frame(second_frame)

src.SparseImgAlign(4, 1, 30, SparseImgAlign.GaussNewton, False, False)
