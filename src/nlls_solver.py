import numpy as np

class NLLSSolver(object):
    """docstring for NLLSSolver"""

    LEVENBERG_MARQUARDT = 1
    GAUSS_NEWTON = 2

    METHODS = {
        LEVENBERG_MARQUARDT: 'Levenberg Marquardt',
        GAUSS_NEWTON: 'GaussNewton'
    }

    def __init__(self):
        self._verbose = True
        self._eps = 1e-10
        self._n_iter_init = 15
        self._n_iter = self._n_iter_init
        # self._method = 


    def reset(self):
        """Reset parameters and optimize again"""
        pass
