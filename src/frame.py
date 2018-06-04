import numpy as np
import cv2
import itertools

import sophus as sp
from .log import LOG_DEBUG, LOG_INFO, LOG_WARN, LOG_ERROR, LOG_CRITICAL

class Frame(object):
    """docstring for Frame"""
    id_generator = itertools.count(0)

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
            LOG_ERROR("""Frame: provided image has not the same size as the camera model
                         or image is not grayscale""")
        # TODO: use cv2.buildOpticalFlowPyramid instead
        self._create_img_pyramid(img, 5)
        LOG_INFO('Image pyramid of', 5, 'created. Frame id:', self.id)

    def _create_img_pyramid(self, img_level_0, levels):
        self.img_pyr.append(img_level_0)
        for i in range(1, levels):
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

    def remove_keypoint(self, feature):
        found = False
        for f in self.features:
            if f == feature:
                f = None
                found = True
        if found:
            self.set_keypoints()
        
    def set_keypoints(self):
        pass

    @staticmethod
    def jacobian_xyz2uv(xyz):
        x, y, z = xyz
        z_inv = 1.0 / z
        z_inv_2 = z_inv ** 2
        
        J00 = -z_inv                # -1/z
        J01 = 0                     # 0
        J02 = x*z_inv_2             # x/z^2
        J03 = y*J02                 # x*y/z^2
        J04 = -(1 + x*J02)          # -(1 + x^2/z^2)
        J05 = y*z_inv               # y/z

        J10 = 0                     # 0
        J11 = -z_inv                # -1/z
        J12 = y*z_inv_2             # y/z^2
        J13 = 1 + y*J12             # 1 + y^2/z^2
        J14 = -J03                  # -x*y/z^2
        J15 = -x*z_inv              # x/z

        return np.array([[J00, J01, J02, J03, J04, J05],
                         [J10, J11, J12, J13, J14, J15]], dtype=np.float64)