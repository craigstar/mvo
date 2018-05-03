import numpy as np

from .log import LOG_DEBUG, LOG_INFO, LOG_WARN, LOG_ERROR, LOG_CRITICAL
from . import my_sophus as sp


class NLLSSolver(object):
    """docstring for NLLSSolver"""

    LEVENBERG_MARQUARDT = 1
    GAUSS_NEWTON = 2

    METHODS = {
        LEVENBERG_MARQUARDT: 'Levenberg Marquardt',
        GAUSS_NEWTON: 'GaussNewton'
    }

    def __init__(self, n_iter_init=15, n_iter=15, verbose=True, eps=1e-10,
        method=LEVENBERG_MARQUARDT):
        # damp parameter. if mu>0, J is positive define which ensures downward direction.
        # if mu is very big, it means step is very big
        self._mu_init = 0.01
        self._mu = self._mu_init
        self._n_meas = 0
        self._n_trials = 0                  # number of attemption
        self._n_trials_max = 5              # number of max attemption
        self._n_iter_init = n_iter_init
        self._n_iter = n_iter_init
        self._verbose = verbose
        self._eps = eps

        self._rho = 0
        self._method = method

        self._use_weight = False


    def reset(self):
        """Reset parameters and optimize again"""
        self._n_meas = 0
        self._n_iter = self._n_iter_init

    def optimize(self, model):
        """Optimize model by LEVENBERG_MARQUARDT or GAUSS_NEWTON"""
        opts = {
            self.GAUSS_NEWTON: self._optimize_gauss_newton,
            self.LEVENBERG_MARQUARDT: self._ptimize_levenberg_marquardt
        }

        opts[self._method](model)
   

    def _optimize_gauss_newton(self, model):
        """optimize by gauss newton method"""

        # calculate weight scale
        if self._use_weight:
            self._compute_residuals(model, False, False)

        old_model = sp.SE3(a.matrix())

        for i in range(self._n_iter):
            self._rho = 0
            self._start_iteration()

            # attempt to calculate and update, if failed, increase mu
            self._n_trials = 0

            while not (self._rho > 0 or self._stop):

            if self._stop: break

            self._finish_iteration()

    def _ptimize_levenberg_marquardt(self, model):
        pass
