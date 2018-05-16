import numpy as np
from scipy import signal

from .log import LOG_DEBUG, LOG_INFO, LOG_WARN, LOG_ERROR, LOG_CRITICAL
from . import my_sophus as sp
from .nlls_solver import NLLSSolver
from .frame import Frame

class SparseImgAlign(NLLSSolver):
    """docstring for SparseImgAlign"""
    def __init__(self, max_level, min_level, n_iter, method, display, verbose):
        super(SparseImgAlign, self).__init__(n_iter_init=n_iter, n_iter=n_iter,
            method=method, verbose=verbose, eps=1e-6)
        self._max_level = max_level
        self._min_level = min_level
        self._display = display
        self._level = 0

        self._patch_halfsize = 2
        self._patch_size = self._patch_halfsize * 2
        self._patch_area = self._patch_size ** 2

        self._have_ref_patch_cache = False
        self._jacobian_cache = np.zeros((0, self._patch_area, 6), dtype=np.float64)
        self._ref_patch_cache = np.zeros((0, self._patch_area), dtype=np.float32)
        self._visible_fts = np.full(0, False)

        self._frm_ref = None
        self._frm_cur = None

        self._resimg = np.zeros(0, dtype=np.uint8)



    def run(self, frm_ref, frm_cur):
        """Computes T for current frame"""
        self.reset()

        if not len(frm_ref.features):
            LOG_ERROR('SparseImgAlign: no features to track!')

        self._frm_ref = frm_ref
        self._frm_cur = frm_cur

        N = len(frm_ref.features)
        self._ref_patch_cache.resize(N, self._patch_area)
        self._jacobian_cache.resize(N, self._patch_area, 6)
        self._visible_fts = np.full(N, False)

        # identity matrix at initial
        T_cur_from_ref = sp.SE3(frm_cur.T_from_w * frm_ref.T_from_w.inverse())

        for self._level in range(self._max_level, self._min_level - 1, -1):
            LOG_INFO("Pyramid level:", self._level)
            self._mu = 0.1
            self._have_ref_patch_cache = False
            self.optimize(T_cur_from_ref)

        frm_cur.T_from_w = T_cur_from_ref * frm_ref.T_from_w
        return self._n_meas / self._patch_area

    def _precompute_reference_patches(self):
        border = self._patch_halfsize + 1
        img_ref = self._frm_ref.img_pyr[self._level]
        rows, cols = img_ref.shape
        scale = 1.0 / (2 ** self._level)
        pos_ref = self._frm_ref.pos()
        f = self._frm_ref.cam.get_focal_length()

        for i, ft in enumerate(self._frm_ref.features):
            u_ref, v_ref = ft.uv * scale
            u_ref_i, v_ref_i = int(u_ref), int(v_ref)

            if (not ft.point or
                u_ref_i - border < 0 or u_ref_i + border >= cols or
                v_ref_i - border < 0 or v_ref_i + border >= rows):
                # if no 3d point or point is outlier
                continue

            self._visible_fts[i] = True

            # here we use depth instead of 3d coordinates to avoid reprojection error
            depth = np.linalg.norm(ft.point.pos - pos_ref)
            xyz_ref = ft.direction * depth

            J_frm = Frame.jacobian_xyz2uv(xyz_ref)
            
            subpix_u_ref = u_ref - u_ref_i
            subpix_v_ref = v_ref - v_ref_i
            w00 = (1 - subpix_u_ref) * (1 - subpix_v_ref)
            w01 = subpix_u_ref * (1 - subpix_v_ref)
            w10 = (1 - subpix_u_ref) * subpix_v_ref
            w11 = subpix_u_ref * subpix_v_ref

            mask_x = np.array([[-w00, -w01, w00, w01],
                               [-w10, -w11, w10, w11]], dtype=np.float64)
            mask_y = mask_x.T
            mask_intensity = mask_x[:, 2:]

            img_patch_x = img_ref[v_ref_i - self._patch_halfsize     : v_ref_i + self._patch_halfsize + 1,
                                  u_ref_i - self._patch_halfsize - 1 : u_ref_i + self._patch_halfsize + 2]
            img_patch_y = img_ref[v_ref_i - self._patch_halfsize - 1 : v_ref_i + self._patch_halfsize + 2,
                                  u_ref_i - self._patch_halfsize     : u_ref_i + self._patch_halfsize + 1]
            img_patch_intensity = img_ref[v_ref_i - self._patch_halfsize : v_ref_i + self._patch_halfsize + 1,
                                          u_ref_i - self._patch_halfsize : u_ref_i + self._patch_halfsize + 1]

            self._ref_patch_cache[i] = signal.correlate2d(
                img_patch_intensity, mask_intensity, mode='valid').ravel()

            # inverse compositional
            dx = signal.correlate2d(img_patch_x, mask_x, mode='valid')
            dy = signal.correlate2d(img_patch_y, mask_y, mode='valid')

            # cache the jacobian
            self._jacobian_cache[i] = ((dx.reshape(-1, 1).dot(J_frm[None, 0])
                                      + dy.reshape(-1, 1).dot(J_frm[None, 1]))
                                      * f * scale)

        LOG_INFO('Precomputed', i, 'features for reference patch')
        self._have_ref_patch_cache = True


    def _compute_residuals(self, T, linearize_system, compute_weight_scale):
        """
        T is T_cur_from_ref, linearize_system
        ---------------------
        In: (SE3, bool, bool)
        Out: float
        ---------------------
        """
        img_cur = self._frm_cur.img_pyr[self._level]

        if linearize_system and self._display:
            self._resimg = np.full(img_cur.shape, 0, dtype=np.float32)

        if not self._have_ref_patch_cache:
            self._precompute_reference_patches()

        if compute_weight_scale:
            # This is not the case for now
            errors = np.zeros_like(self._visible_fts)
            pass

        border = self._patch_halfsize + 1
        rows, cols = img_cur.shape
        scale = 1.0 / (2 ** self._level)
        pos_ref = self._frm_ref.pos()
        chi2 = 0.0

        for i, ft in enumerate(self._frm_ref.features):
            # check if point is visible in frame
            if not self._visible_fts[i]:
                continue

            # here we use depth instead of 3d coordinates to avoid reprojection error
            depth = np.linalg.norm(ft.point.pos - pos_ref)
            xyz_ref = ft.direction * depth
            xyz_cur = T * xyz_ref
            u_cur, v_cur = self._frm_cur.cam.world2cam(xyz_cur) * scale
            u_cur_i, v_cur_i = int(u_cur), int(v_cur)

            if (u_cur_i - border < 0 or u_cur_i + border >= cols or
                v_cur_i - border < 0 or v_cur_i + border >= rows):
                # point not visible in current frame
                continue

            subpix_u_cur = u_cur - u_cur_i
            subpix_v_cur = v_cur - v_cur_i
            w00 = (1 - subpix_u_cur) * (1 - subpix_v_cur)
            w01 = subpix_u_cur * (1 - subpix_v_cur)
            w10 = (1 - subpix_u_cur) * subpix_v_cur
            w11 = subpix_u_cur * subpix_v_cur

            mask_intensity = np.array([[w00, w01],
                                       [w10, w11]], dtype=np.float64)

            img_patch_intensity = img_cur[v_cur_i - self._patch_halfsize : v_cur_i + self._patch_halfsize + 1,
                                          u_cur_i - self._patch_halfsize : u_cur_i + self._patch_halfsize + 1]
            intensity_cur = signal.correlate2d(img_patch_intensity, mask_intensity, mode='valid').ravel()
            res = intensity_cur - self._ref_patch_cache[i]
            chi2 += (res**2).sum()
            # print(res)
            if linearize_system:
                J = self._jacobian_cache[i]
                self._H += J.T.dot(J)
                self._Jres -= res.ravel().dot(J)

        print(self._H)

        return chi2 / self._visible_fts.sum()

    def _solve(self):
        return True

    def _update(self, model):
        pass

    def _start_iteration(self):
        pass

    def _finish_iteration(self):
        if self._display:
            cv2.imshow("residuals", self._resimg * 10)
            cv2.waitKey(0)
