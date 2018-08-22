import numpy as np

import nifty5


class TauFCurvature(nifty5.EndomorphicOperator):
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
        self.p_space = p_space
        self.len_p_space = p_space.shape[0]
        self.smoothness_operator_f = smoothness_operator_f

    @property
    def capability(self):
        return self.TIMES

    def apply(self, x, mode):
        """
        second derivative wrt tau_f
        """
        self._check_mode(mode=mode)
        if mode == self.TIMES:

            # tr(-G.Lambda_z.G.Lambda_z')
            term1 = nifty5.Field.from_global_data(
                    domain=x.domain,
                    arr=np.array([nifty5.Field.from_global_data(
                        domain=x.domain,
                        arr=np.array([np.trace(
                            -self.G@self.Lambdas[i]@self.G@self.Lambdas[j]) for
                                i in range(self.len_p_space)])).vdot(x)
                        for j in range(self.len_p_space)]))

            # tr(G.Lambda_z.delta_zz')
            term2 = nifty5.Field.from_global_data(
                domain=x.domain,
                arr=np.array([np.trace(self.G@self.Lamdas[i]) for i in range(
                    self.len_p_space)]))*x

            # y.(G.Lambda_z.G.Lambda_z'.G ).y
            term3 = nifty5.Field.from_global_data(
                    domain=x.domain,
                    arr=np.array([nifty5.Field.from_global_data(
                        domain=x.domain,
                        arr=np.array([self.y@(
                            self.G@self.Lambdas[i]@self.G@self.Lambdas[j])@
                            self.y for i in range(self.len_p_space)])).vdot(x)
                        for j in range(self.len_p_space)]))

            # y.(- G.Lambda.delta_zz').y
            term4 = nifty5.Field.from_global_data(
                    domain=x.domain,
                    arr=np.array([
                        -self.y@(self.G@self.Lamdas[i])@self.y
                        for i in range(self.len_p_space)]))*x

            term5 = self.smoothness_operator_f(x)

            result = term1 + term2 + term3 + term4 + term5

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
