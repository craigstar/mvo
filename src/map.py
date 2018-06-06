import numpy as np


class Map(object):
    """docstring for Map"""
    def __init__(self):
        self.keyframes = []
        self.points = []

    def get_close_keyframes(self, frame):
        close_kfs = []
        for kf in self.keyframes:
            for kp in kf.keypoints:
                if kp is not None and frame.isvisible(kp.point.pos):
                    dist = np.norm(frame.T_from_w.inverse().translation() - kf.T_from_w.inverse().translation())
                    close_kfs.append((kf, dist))
                    break
        