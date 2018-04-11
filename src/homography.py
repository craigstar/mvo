from sympy.liealgebras.root_system import RootSystem

class Homography(object):
    """docstring for Homography"""
    def __init__(self, fts1, fts2, focal_length, reprojection_threshold):
        self.fts1 = fts1            # features reference (0, 0), N * 2d array
        self.fts2 = fts2            # features current (0, 0), N * 2d array
        self.f = focal_length
        self.threshold = reprojection_threshold
        self.H = None
        self.mask = None

    def compute_SE3_from_matches(self):
        self.calc_from_matches()

    def calc_from_matches(self):
        threshold = 2.0 / self.f
        self.H, self.mask = cv2.findHomography(
                                xy_ref, xy_cur, method=cv2.RANSAC,
                                ransacReprojThreshold=threshold)
