import numpy as np
import sophus as sp

from .log import LOG_DEBUG, LOG_INFO, LOG_WARN, LOG_ERROR, LOG_CRITICAL


class NLLSSolver(object):
    """docstring for NLLSSolver"""

    LEVENBERG_MARQUARDT = 1
    GAUSS_NEWTON = 2

    METHODS = {
        LEVENBERG_MARQUARDT: 'Levenberg Marquardt',
        GAUSS_NEWTON: 'GaussNewton'
    }

    DIM = 6

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
        self._stop = False                  # stop sign

        self._rho = 0
        self._method = method

        self._use_weight = False

        self._H = np.zeros((self.DIM, self.DIM), dtype=np.float64)      # Hessian approximation
        self._Jres = np.zeros(self.DIM, dtype=np.float64)               # Jacobian x Residual
        self._x = np.zeros(self.DIM, dtype=np.float64)                  # update variable
        self._have_prior = False
        self._chi2 = 0

    def _apply_prior(self, model_cur):
        pass

    def reset(self):
        """Reset parameters and optimize again"""
        self._n_meas = 0
        self._n_iter = self._n_iter_init
        self._have_prior = False
        self._chi2 = 1e10
        self._mu = self._mu_init
        self._stop = False

    def optimize(self, model):
        """Optimize model by LEVENBERG_MARQUARDT or GAUSS_NEWTON"""
        opts = {
            self.GAUSS_NEWTON: self._optimize_gauss_newton,
            self.LEVENBERG_MARQUARDT: self._ptimize_levenberg_marquardt
        }

        return opts[self._method](model)
   

    def _optimize_gauss_newton(self, model):
        """optimize by gauss newton method"""

        # calculate weight scale
        if self._use_weight:
            LOG_INFO('Using weight scale')
            self._compute_residuals(model, False, True)

        old_model = sp.SE3(model)

        for i in range(self._n_iter):
            self._rho = 0
            self._start_iteration()

            self._H.fill(0)
            self._Jres.fill(0)

            # calculate initial residuals
            self._n_meas = 0
            new_chi2 = self._compute_residuals(model, True, False)

            # add prior estimate
            if self._have_prior:
                self._apply_prior(model)

            # calculate linear problem
            if not self._solve():
                LOG_WARN('Matrix is close to singular! Stop Optimizing.')
                LOG_INFO('H =', self._H)
                LOG_INFO('Jres =', self._Jres)
                self._stop = True

            # check if error has increased, roll model back when yes
            if (i > 0 and new_chi2 > self._chi2) or self._stop:
                LOG_ERROR('Iteration.', i, 'Failure. new_chi2 =', new_chi2, 'Error increased. Stop optimizing.')
                model = old_model
                break

            new_model = self._update(model)
            old_model = model
            model = new_model 
            
            self._chi2 = new_chi2 

            LOG_INFO('Iteration.', i, 'Success. new_chi2 =', new_chi2,
                'n_meas=', self._n_meas, 'x norm=', max(abs(self._x)))

            self._finish_iteration()

            # stop when converge, step is too small
            if max(abs(self._x)) < self._eps:
                break
        
        return model

    def _ptimize_levenberg_marquardt(self, model):
        pass