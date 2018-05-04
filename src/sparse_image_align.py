import numpy as np

from .log import LOG_DEBUG, LOG_INFO, LOG_WARN, LOG_ERROR, LOG_CRITICAL
from . import my_sophus as sp
from .nlls_solver import NLLSSolver

class SparseImgAlign(NLLSSolver):
    """docstring for SparseImgAlign"""
    def __init__(self, max_level, min_level, n_iter, method, display, verbose):
        super(SparseImgAlign, self).__init__(n_iter_init=n_iter, n_iter=n_iter,
            method=method, verbose=verbose, eps=1e-6)
        self._max_level = max_level
        self._min_level = min_level
        self._display = display
        self._level = 0

        self.patch_halfsize = 2
        self.patch_size = self.patch_halfsize * 2
        self.patch_area = self.patch_size ** 2

        self.frm_ref = None
        self.frm_cur = None

        self._resimg = np.zeros(0, dtype=np.uint8)


    def run(self, frm_ref, frm_cur):
        """Computes T for current frame"""
        self.reset()

        if not len(frm_ref.features):
            LOG_ERROR('SparseImgAlign: no features to track!')

        self.frm_ref = frm_ref
        self.frm_cur = frm_cur

        # identity matrix at initial
        T_cur_from_ref = sp.SE3(frm_cur.T_from_w * frm_cur.T_from_w.inverse())

        for level in range(self._max_level, self._min_level - 1, -1):
            if (self._verbose): LOG_INFO("Pyramid level:", level)                
            self._mu = 0.1
            self.optimize(T_cur_from_ref)

        frm_cur.T_from_w = T_cur_from_ref * frm_ref.T_from_w
        return self._n_meas / self.patch_area


    def _compute_residuals(self, T_cr, linearize_system, compute_weight_scale):
        """
        T_cr is T_cur_from_ref, linearize_system
        ------------------
        In: (SE3, bool, bool)
        Out: float
        ------------------
        """
        print('hello')

    def _solve(self):
        return True

    def _update(self):
        pass

    def _start_iteration(self):
        pass

    def _finish_iteration(self):
        if self._display:
            cv2.imshow("residuals", self._resimg * 10)
            cv2.waitKey(0)
