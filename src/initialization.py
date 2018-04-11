from .detector import GoodFeaturesDetector
from .homography import Homography

class Initialization(object):
    """docstring for Initialization"""

    SUCCESS = 0
    FAILURE = 1
    NO_KEYFRAME = 2

    InitResult = {
        FAILURE: 'init failed',
        NO_KEYFRAME: 'no key frame',
        SUCCESS: 'init succeed'
    }

    def __init__(self):
        self.kps_ref = []       # position of reference feature points
        self.kps_cur = []       # position of current feature points
        self.dir_ref = []       # directions of feature points
        self.frm_ref = None # reference frame

    def _reset(self):
        self.kps_cur = []
        self.frm_ref = None

    def _detect_features(self, frame):
        rows, cols = frame.img.shape
        d = GoodFeaturesDetector(width=cols, height=rows,
                                 cell_size=25, pyr_levels=3)
        fts = d.detect(frame, frame.img_pyr, 0.01)
        
        positions = []
        directions = []
        for ft in fts:
           positions.append(ft.uv)
           directions.append(ft.direction)
        return (np.array(positions), np.array(directions))

    def _track_klt(self, frm_ref, frm_cur, kps_ref, dir_ref):
        win_sz = 30.0
        max_iter = 30
        eps = 0.001
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
                    max_iter, eps)

        kps_cur, status, error = cv2.calcOpticalFlowPyrLK(
                                frm_ref.img_pyr[0], frm_cur.img_pyr[0],
                                kps_ref, None, winSize=(win_sz, win_sz),
                                maxLevel=4, criteria=criteria)

        mask = status.flatten() == 1
        kps_ref, kps_cur = kps_ref[mask], kps_cur[mask]
        dir_ref = dir_ref[mask]
        dir_cur = np.apply_along_axis(frm_cur.c2f, 1, kps_cur)
        disparities = np.linalg.norm(kps_ref - kps_cur, axis=1)

        return (kps_ref, kps_cur, dir_ref, dir_cur, disparities)

    def _point2d(self, xyz):
        if xyz.ndim == 1:
            return xyz[:2] / xyz[2]
        elif xyz.ndim == 2:
            return xyz[:, :2] / xyz[:, 2, np.newaxis]
        else:
            return np.zeros(2)

    def computeHomography(self, dir_ref, dir_cur, f, reprojection_threshold):
        xy_ref = self._point2d(dir_ref)
        xy_cur = self._point2d(dir_cur)
        homo = Homography(xy_ref, xy_cur, f, reprojection_threshold)
        homo.computeSE3fromMatches()
        # H, mask = cv2.findHomography(xy_ref, xy_cur, method=cv2.RANSAC, 
        #                              ransacReprojThreshold=5.0)


    def add_first_frame(self, frame):
        self._reset()
        self.kps_ref, self.dir_ref = self._detect_features(frame)
        if (len(kps_ref) < 100):
            print('First image has less than 100 features.' \
                  ' Retry in more textured environment.')
            return Initialization.FAILURE

        self.frm_ref = frame
        self.kps_cur = list(self.kps_ref) # make a copy of kps_ref
        return Initialization.SUCCESS

    def add_second_frame(self, frm_cur):
        kps_ref, kps_cur, dir_ref, dir_cur, disparities = self._track_klt(
            self.frm_ref, frm_cur, self.kps_ref, self.dir_ref)
        print('Init: KLT tracked:', len(disparities), 'features')
        
        if (len(disparities) < 50):
            return Initialization.FAILURE
        disparity = np.median(disparities)
        print('Init: KLT', disparity, 'px average disparity.')

        if (disparity < 50):
            return Initialization.NO_KEYFRAME

        self.computeHomography()