import numpy as np

import nifty5
from nifty5.sugar import exp

from ..curvatures.cause_curvature_shallow import BetaCurvature


class CauseEnergyShallow(nifty5.Energy):
    """
    energy class for the bayesian causal inference.
    refers to the gamma functional

    Parameters:
    ----------
    position : nifty field
        Referred to as beta

    k : np array with same dimension as grid
        defines counts per bin point

    s_space : nifty domain
        the signal space, the domain on which beta (the position) is defined

    a : scalar
        amplitude of the power spectrum

    q_0 : scalar
        scaling factor of the power spectrum

    rho : scalar,
        density of the response
    """
    def __init__(
            self,
            position,
            k,
            s_space,
            power_spectrum_beta,
            rho,
            ):
        super().__init__(position=position)
        self.k = k
        self.rho = rho
        self.power_spectrum_beta = power_spectrum_beta
        self.N = k.sum()
        self.beta = position
        self.beta_arr = position.to_global_data()
        self.s_space = s_space
        self.fft = nifty5.FFTOperator(self.s_space)
        self.h_space = self.fft.target[0]

        # B in Fourier space:
        self.B_h = nifty5.create_power_operator(
                domain=self.h_space,
                power_spectrum=power_spectrum_beta)
        self.B = nifty5.SandwichOperator.make(self.fft, self.B_h)

    @property
    def value(self):
        value_terms = self.get_value_terms()
        value = np.sum(np.array(value_terms))
        return value

    def get_value_terms(self):
        """
        get the terms for the value

        returns:
        --------
        (term1, term2, term3) : tuple of scalars (float)
        """
        # -k^dagger.beta_vec
        term1 = -self.k.vdot(self.beta)
        # rho^dagger.e^beta_vec
        term2 = self.rho * exp(self.beta_arr).sum()

        # 1/2 beta_0^dagger.B^-1.beta_0
        term3 = 0.5*self.beta.vdot(self.B.inverse_times(self.beta))

        return (term1, term2, term3)

    @property
    def gradient(self):
        gradient_terms = self.get_gradient_terms()
        gradient = np.sum(np.array(gradient_terms), axis=0)
        return nifty5.Field(domain=self.s_space, val=gradient)

    def get_gradient_terms(self):
        """
        get the terms of the gradient

        returns:
        --------
        (term1, term2, term3) : tuple of nifty fields
        """
        # -k
        term1 = -self.k.copy()

        # rho(e^beta_vec)
        term2 = self.rho * exp(self.beta)

        # beta.B^(-1)
        term3 = self.B.adjoint_inverse_times(self.beta)

        return (term1, term2, term3)

    @property
    def curvature(self):
        iteration_controller = nifty5.GradientNormController(
                iteration_limit=300,
                tol_abs_gradnorm=1e-3,
                name=None)
        return nifty5.InversionEnabler(
            BetaCurvature(
                    domain=self.s_space,
                    beta=self.beta,
                    B=self.B,
                    rho=self.rho),
                iteration_controller=iteration_controller)

    def at(self, position):
        return self.__class__(
                position=position,
                k=self.k,
                s_space=self.s_space,
                power_spectrum_beta=self.power_spectrum_beta,
                rho=self.rho)
