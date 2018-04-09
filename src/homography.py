class Homography(object):
    """docstring for Homography"""
    def __init__(self, fts1, fts2, focal_length, reprojection_threshold):
        self.fts1 = fts1            # features reference
        self.fts2 = fts2            # features current
        self.f = focal_length
        self.threshold = reprojection_threshold

    def computeSE3fromMatches(self):
        pass