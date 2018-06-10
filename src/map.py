import numpy as np
from .frame import Frame


class Map(object):
    """docstring for Map"""
    def __init__(self):
        self.keyframes = []
        self.points = []

    def get_close_keyframes(self, frame):
        # TODO: this includes keyframe itself
        close_kfs = []
        for kf in self.keyframes:
            for kp in kf.keypoints:
                if kp is not None and frame.isvisible(kp.point.pos):
                    dist = np.linalg.norm(frame.T_from_w.inverse().translation()
                         - kf.T_from_w.inverse().translation())
                    close_kfs.append((kf, dist))
                    break
        return close_kfs
        
    def add_keyframe(self, frame):
        if isinstance(frame, Frame):
            self.keyframes.append(frame)
            return 0
        return -1