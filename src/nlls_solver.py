import numpy as np

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
        # initialization
        self._verbose = verbose
        self._eps = eps
        self._n_iter_init = n_iter_init
        self._n_iter = n_iter_init
        self._method = method
        self._n_meas = 0

    def reset(self):
        """Reset parameters and optimize again"""
        pass

    def optimize(self, model):
        """Optimize model by LEVENBERG_MARQUARDT or GAUSS_NEWTON"""
        opts = {
            self.GAUSS_NEWTON: self._optimize_gauss_newton,
            self.LEVENBERG_MARQUARDT: self._ptimize_levenberg_marquardt
        }

        opts[self._method](model)
   

    def _optimize_gauss_newton(self, model):
        pass

    def _ptimize_levenberg_marquardt(self, model):
        pass
