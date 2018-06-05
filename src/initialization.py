import numpy as np
import cv2

from .detector import GoodFeaturesDetector
from .feature import Feature
from .point3d import Point3d
import sophus as sp
from .log import LOG_DEBUG, LOG_INFO, LOG_WARN, LOG_ERROR, LOG_CRITICAL
from . import log

log.init_log('mvo', log.DEBUG, mode=log.MODE_CONSOLE)

class Initialization(object):
    """docstring for Initialization"""

    SUCCESS = 0
    FAILURE = 1
    NO_KEYFRAME = 2

    INIT_RESULT = {
        FAILURE: 'init failed',
        NO_KEYFRAME: 'no key frame',
        SUCCESS: 'init succeed'
    }

    def __init__(self):
        self.kps_ref = []               # position of reference feature points
        self.kps_cur = []               # position of current feature points
        self.dir_ref = []               # directions of feature points
        self.frm_ref = None             # reference frame
        self.T_cur_from_ref = sp.SE3()  # translation from reference to current

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
        win_sz = 30
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

    def _compute_RT(self, pts1, pts2, threshold):
        E, mask = cv2.findEssentialMat(pts1, pts2,
                                       focal=718.856,
                                       pp=(607.1928, 185.2157),
                                       method=cv2.RANSAC,
                                       threshold=threshold,
                                       prob=0.999)
        n, R, t, mask = cv2.recoverPose(E, pts1, pts2,
                                        focal=718.856,
                                        pp=(607.1928, 185.2157),
                                        mask=mask)
        return (R, t, mask.ravel() == 1)

    def _compose_projection(self, R, t):
        p_ref = self.frm_ref.cam.compose_P(np.eye(3), np.zeros((3, 1)))
        p_cur = self.frm_cur.cam.compose_P(R, t)
        return (p_ref, p_cur)


    def _triangulate_points(self, P_ref, P_cur, kps_ref, kps_cur):
        """Calculate 3d points"""
        pts3d_homo = cv2.triangulatePoints(P_ref, P_cur, kps_ref.T, kps_cur.T).T
        pts3d = cv2.convertPointsFromHomogeneous(pts3d_homo)
        return pts3d.reshape((-1, 3)).astype(np.float64)

    def _depth_check(self, pts3d):
        """Remove points have depth less than 1"""
        valid_depth = pts3d[:, 2] > 0
        return (pts3d[valid_depth], valid_depth)

    def add_first_frame(self, frame):
        self._reset()
        self.kps_ref, self.dir_ref = self._detect_features(frame)
        LOG_INFO(len(self.kps_ref), 'features detected in 1st frame')

        if (len(self.kps_ref) < 100):
            LOG_ERROR('First image has less than 100 features.' \
                      ' Retry in more textured environment.')
            return Initialization.FAILURE

        self.frm_ref = frame
        self.kps_cur = self.kps_ref.copy() # make a copy of kps_ref
        return Initialization.SUCCESS

    def add_second_frame(self, frm_cur):
        kps_ref, kps_cur, dir_ref, dir_cur, disparities = self._track_klt(
            self.frm_ref, frm_cur, self.kps_ref, self.dir_ref)
        LOG_INFO('Init: KLT tracked:', len(disparities), 'features')

        if (len(disparities) < 50):
            LOG_ERROR('Error, number of features < 50')
            return Initialization.FAILURE
        disparity = np.median(disparities)
        LOG_INFO('Init: KLT', disparity, 'px average disparity.')

        if (disparity < 5):
            LOG_ERROR('Error, disparity < 5')
            return Initialization.NO_KEYFRAME

        # assign to class member
        self.frm_cur = frm_cur

        reprojection_threshold = 2
        R, t, mask = self._compute_RT(
            kps_ref, kps_cur, reprojection_threshold)
        
        # set T
        self.T_cur_from_ref = sp.SE3(R, t.ravel())

        # filter out outliers
        kps_ref, kps_cur = kps_ref[mask], kps_cur[mask]
        dir_ref, dir_cur = dir_ref[mask], dir_cur[mask]
        LOG_INFO("Init: Essential RANSAC", np.sum(mask), "inliers.")

        P_ref, P_cur = self._compose_projection(R, t)
        pts3d = self._triangulate_points(P_ref, P_cur, kps_ref, kps_cur)
        pts3d, mask = self._depth_check(pts3d)
        kps_ref, kps_cur = kps_ref[mask], kps_cur[mask]
        dir_ref, dir_cur = dir_ref[mask], dir_cur[mask]
        LOG_INFO('valid 3d points:', len(pts3d))

        # calculate scale
        scale = 1.0 / np.mean(pts3d[:, 2])
        LOG_INFO('scale is:', scale)

        # calculate current T from world
        frm_cur.T_from_w = self.T_cur_from_ref * self.frm_ref.T_from_w

        # calculate current translation by scale
        ref_pos, cur_pos = self.frm_ref.pos(), frm_cur.pos()
        new_pos = ref_pos + scale * (cur_pos - ref_pos)
        translation = np.matmul(-frm_cur.T_from_w.rotationMatrix(), new_pos)
        frm_cur.T_from_w.setTranslation(translation)
        LOG_INFO('current pose:\n', frm_cur.T_from_w)

        # T to translate camera points to world coordinate
        T_world_cur = self.frm_cur.T_from_w.inverse()
        pts3d_ref = T_world_cur * (pts3d * scale)

        for p3, kr, kc, dr, dc in zip(pts3d_ref, kps_ref, kps_cur, dir_ref, dir_cur):
            if (self.frm_ref.cam.is_in_frame(kr, 10) and
                self.frm_ref.cam.is_in_frame(kc, 10)):
                # create Point3d Feature, and add Feature to frame, to Point3d
                new_point = Point3d(p3)
                feature_cur = Feature(self.frm_cur, kc, pt3d=new_point, direction=dc)
                self.frm_cur.add_feature(feature_cur)
                new_point.add_frame_ref(feature_cur)

                feature_ref = Feature(self.frm_ref, kr, pt3d=new_point, direction=dr)
                self.frm_ref.add_feature(feature_ref)
                new_point.add_frame_ref(feature_ref)