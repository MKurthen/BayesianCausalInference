import numpy as np

import nifty5

from .tau_f_curvature import TauFCurvature


class EffectCurvatureShallow(nifty5.EndomorphicOperator):
    """
    curvature operator for shallow effect energy (i.e. without inference of
        noise inference
    """
    def __init__(
            self,
            p_space,
            smoothness_operator_f,
            ):

        super().__init__()
        self._domain = p_space
        self.len_p_space = p_space.shape[0]
        self.smoothness_operator_f = smoothness_operator_f
        self.tau_f_curvature = TauFCurvature(
                p_space=p_space,
                smoothness_operator_f=smoothness_operator_f)

    @property
    def capability(self):
        return self.TIMES

    def apply(self, x, mode):
        """
        second derivative wrt tau_f
        """
        self._check_mode(mode=mode)
        if mode == self.TIMES:
            result = self.tau_f_curvature(x)
        else:
            x0 = nifty5.Field.from_zeros(domain=x.domain)
            energy = nifty5.QuadraticEnergy(A=self.times, b=x, position=x0)
            r = self.inverter(energy, preconditioner=self.preconditioner)[0]
            result = r.position
        return result

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
