#!/usr/bin/env python

import time
import cv2

from src import Frame, PinholeCamera, Initialization


# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

def getName(k):
    name0 = "%06d" % k
    name1 = "%06d" % (k + 1)
    sufix = '.png'
    return name0 + sufix, name1 + sufix


####### 1.0 read images #######
path = '../../dataset/sequences/00/image_0/'
init = Initialization()

for i in range(1):
    print('loop', i)
    name0, name1 = getName(i)
    img0 = cv2.imread(path + name0)
    img1 = cv2.imread(path + name1)
    gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    rows, cols = gray0.shape

    ####### 2.0 create camera, frames #######
    cam = PinholeCamera(width=cols, height=rows,
                        fx=718.856, fy=718.856,
                        cx=607.1928, cy=185.2157
                        )

    first_frame = Frame(cam, gray0, 0.0)
    second_frame = Frame(cam, gray1, 1.0)

    ####### 3.0 initialize pose estimator #######
    s = time.time()
    init.add_first_frame(first_frame)
    e = time.time()
    init.add_second_frame(second_frame)
    print('time elapsed:', e - s)

    # pts = np.array([ft.point.pos for ft in second_frame.features if ft.point is not None])
    # x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    # fig = plt.figure()
    # ax = plt.subplot(111, projection='3d')
    # ax.scatter(x, y, z, c='y')
    # ax.set_zlabel('Z')  # 坐标轴
    # ax.set_ylabel('Y')
    # ax.set_xlabel('X')
    # plt.show()
