import numpy as np
import cv2
import time
import sophus as sp

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
                    cx=607.1928, cy=185.2157)

first_frame = Frame(cam, gray0, 0.0)
second_frame = Frame(cam, gray1, 0.0)
third_frame = Frame(cam, gray2, 0.0)

init.add_first_frame(first_frame)
init.add_second_frame(second_frame)



# tmp for debug
# b = np.array([-0.000700019, 0.00482854, -0.00345816, -0.000251848, 0.00178883, -0.042933])
T = np.array([[0.99998236323097067,   0.0034564484332153672,   0.0048297195810387038, -0.00025184768675006765],
               [- 0.0034598284905195077,     0.99999377557329561,  0.00069166572966620858,   0.0017888346447656627],
               [- 0.0048272988118755545, -0.00070836353232518669,     0.99998809763281027,   -0.042933047769165435],
               [0,                       0,                       0,                       1]])
second_frame.T_from_w = sp.SE3(T)




img_align = src.SparseImgAlign(4, 1, 30, SparseImgAlign.GAUSS_NEWTON, False, False)
img_align_n_tracked = img_align.run(second_frame, third_frame)

print("Img Align:\t Tracked = ", img_align_n_tracked)
print("first pose:", first_frame.T_from_w)
print("second pose:", second_frame.T_from_w)
print("third pose:", third_frame.T_from_w)
