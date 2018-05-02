import numpy as np

from .log import LOG_DEBUG, LOG_INFO, LOG_WARN, LOG_ERROR, LOG_CRITICAL
from . import my_sophus as sp
from .nlls_solver import NLLSSolver

class SparseImgAlign(NLLSSolver):
    """docstring for SparseImgAlign"""
    def __init__(self, max_level, min_level, n_iter, method, display, verbose):
        self.arg = arg

    def run(self, frm_ref, frm_cur):
        """Computes"""
        self._reset()

        if not len(frm_ref.features):
            LOG_ERROR('SparseImgAlign: no features to track!')

        self.frm_ref = frm_ref
        self.frm_cur = frm_cur





    def compute_residuals(self, T_cr, linearize_system, compute_weight_scale):
        """
        T_cr is T_cur_from_ref, linearize_system
        ------------------
        In: (SE3, bool, bool)
        Out: float
        ------------------
        """


