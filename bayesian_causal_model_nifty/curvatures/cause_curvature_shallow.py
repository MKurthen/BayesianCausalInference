import numpy as np
import scipy.linalg

import nifty5
from nifty5.sugar import exp
from ..utilities import probe_operator


class BetaCurvature(nifty5.EndomorphicOperator):
    """
    curvature operator for bayesian causal inference.
    Refers to the curvature of the gamma functional

    Parameters
    ----------
    domain: domain on which the curvature is defined

    beta: nifty field instance

    k: np array with same dimension as grid, defines counts per bin point

    a : scalar
        amplitude of the power spectrum, P(q) = a*f(q)

    q_0 : scalar
        scaling factor of the power spectrum, P(q) = f(q/q_0)

    rho : scalar
        density of the response
    """
    def __init__(
            self,
            domain,
            beta,
            B,
            rho=1,
            verbosity=0,
            ):
        super().__init__()
        self.beta = beta
        self._domain = (domain, )
        self.fft = nifty5.FFTOperator(self._domain)
        self.h_space = self.fft.target[0]
        self.rho = rho
        self.B = B
        #B_inv = scipy.linalg.inv(probe_operator(B))
        #self.matrix = B_inv + np.diag(np.exp(beta.to_global_data()))

    @property
    def capability(self):
        return self.TIMES #| self.INVERSE_TIMES

    def apply(self, x, mode):

        self._check_mode(mode)
        if mode == self.TIMES:
            # B^(-1)
            term1 = self.B.inverse_times(x)

            # rho * hat{e^beta}
            term2 = self.rho * exp(self.beta) * x

            return term1 + term2
        #if mode == self.INVERSE_TIMES:
        #    inverse_matrix = scipy.linalg.inv(self.matrix)
        #    x_arr = x.to_global_data()
        #    res_arr = inverse_matrix@x_arr
        #    res = nifty5.Field.from_global_data(domain=x.domain, arr=res_arr)
        #    return res


    @property
    def self_adjoint(self):
        return True

    @property
    def target(self):
        return self._domain

    @property
    def domain(self):
        return self._domain

    @property
    def unitary(self):
        return False
