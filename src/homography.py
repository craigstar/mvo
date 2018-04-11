from sympy.liealgebras.root_system import RootSystem

class Homography(object):
    STATUS_OK = 0
    STATUS_INIT = 1

    Status = {
        STATUS_OK: 'status ok',
        STATUS_INIT: 'FATAL Homography Initialization: This motion case is not implemented or is degenerate. Try again.'
    }

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
        self.decompose()


    def calc_from_matches(self):
        threshold = 2.0 / self.f
        self.H, self.mask = cv2.findHomography(
                                xy_ref, xy_cur, method=cv2.RANSAC,
                                ransacReprojThreshold=threshold)

    def decompose(self):
        U, s, V = np.linalg.svd(self.H)
        if not (s[0] != s[1] and s[1] != s[2]):
            return self.STATUS_INIT
