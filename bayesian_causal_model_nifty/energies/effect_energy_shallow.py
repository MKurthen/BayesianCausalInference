import numpy as np

import nifty5

from ..utilities import probe_operator
from ..curvatures.effect_curvature_shallow import EffectCurvatureShallow


class EffectEnergyShallow(nifty5.Energy):
    def __init__(
            self,
            position,
            k,
            x,
            y,
            x_indices,
            s_space,
            h_space,
            p_space,
            grid,
            sigma_f=1,
            noise_variance=0.01,
            Lambda_modes_list=None,
            term_factors=[1]*11,
            ):

        super().__init__(position=position)
        self.domain = position.domain
        self.k, self.x, self.y = k, x, y
        self.x_indices = x_indices
        self.N = len(self.x)
        self.sigma_f = sigma_f
        self.noise_variance = noise_variance
        self.tau_f = position
        self.s_space = s_space
        self.h_space = h_space
        self.p_space = p_space
        self.len_p_space = self.p_space.shape[0]
        self.grid = grid

        self.fft = nifty5.FFTOperator(domain=self.s_space, target=self.h_space)
        self.F_h = nifty5.create_power_operator(
            domain=self.h_space,
            power_spectrum=nifty5.exp(self.tau_f))
        self.F = nifty5.SandwichOperator.make(self.fft, self.F_h)
        self.F_matrix = probe_operator(self.F)
        self.F_tilde = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(i+1):
                self.F_tilde[i, j] = self.F_matrix[
                        x_indices[i], x_indices[j]]
                if i != j:
                    self.F_tilde[j, i] = self.F_matrix[
                            x_indices[i], x_indices[j]]

        self.noise_covariance = np.diag([
            self.noise_variance for i in range(self.N)])
        self.G = np.linalg.inv(self.F_tilde + self.noise_covariance)

        self.smoothness_operator_f = nifty5.SmoothnessOperator(
                domain=self.p_space,
                logarithmic=True,
                strength=1/self.sigma_f)

        if Lambda_modes_list is None:
            self.get_Lambda_modes(speedup=True)
        else:
            self.Lambda_modes_list = Lambda_modes_list
        self.term_factors = term_factors
        self.get_Lambdas()

    @property
    def value(self):
        value_terms = self.get_value_terms()
        value = sum(value_terms)
        return value

    def get_value_terms(self):

        # log( det( F_tilde[tau_f] + hat(e^eta(x))))
        sign, log_det = np.linalg.slogdet(self.F_tilde + self.noise_covariance)
        if sign <= 0:
            print('warning, computed sign {} in ln_det_DB'.format(sign))
        term1 = 0.5 * log_det

        # y^dagger.(F_tilde + hat(e^eta(x)))^-1.y
        term2 = 0.5*self.y@self.G@self.y

        # tau_beta laplace laplace tau_beta
        term3 = 0.5*self.smoothness_operator_f.times(self.tau_f).vdot(
                self.tau_f)

        terms = (
            self.term_factors[0]*term1,
            self.term_factors[1]*term2,
            self.term_factors[2]*term3,
            )

        return terms

    def get_Lambda_modes(self, speedup=False):
        # this is an array of matrices:
        #   Lambda_mode(z)_ij = (1/2pi) (cos(z(x_i-x_j)) + sin(z(x_i + x_j)) )
        self.Lambda_modes_list = list()
        if not False:
            # conventional way
            for k in range(self.len_p_space):
                p_spec = np.zeros(self.len_p_space)
                p_spec[k] = np.exp(self.tau_f.val[k])
                p_spec[k] = 1
                Lambda_h = nifty5.create_power_operator(
                        domain=self.h_space,
                        power_spectrum=nifty5.Field(
                            domain=self.p_space, val=p_spec))
                Lambda_kernel = nifty5.SandwichOperator.make(self.fft, Lambda_h)
                Lambda_kernel_matrix = probe_operator(Lambda_kernel)
                Lambda_modes = np.zeros((self.N, self.N))
                for i in range(self.N):
                    for j in range(i+1):
                        Lambda_modes[i, j] = Lambda_kernel_matrix[
                                self.x_indices[i], self.x_indices[j]]
                        if i != j:
                            Lambda_modes[j, i] = Lambda_kernel_matrix[
                                    self.x_indices[i], self.x_indices[j]]
                self.Lambda_modes_list.append(Lambda_modes)

        else:
            # based on the thought that we are just dealing with single
            #   fourier modes, we can calculate these directly
            Lambda_modes = np.zeros((self.N, self.N))
            for i in range(self.N):
                for j in range(i+1):
                    x_i_index = self.x_indices[i]
                    x_j_index = self.x_indices[j]
                    a = (
                            self.grid_coordinates[x_i_index] -
                            self.grid_coordinates[x_j_index])
                    # dirty hack, if we take cos((x-y)*2*pi) we get the
                    #   desired result
                    Lambda_modes[i, j] = np.cos(a*2*np.pi)
                    if i != j:
                        Lambda_modes[j, i] = np.cos(a*2*np.pi)
            # use the relation cos(nx) = T_n(cos(x)) where T_n is the nth
            #   chebyhsev polynomial
            for i, z in enumerate(self.p_space[0].k_lengths):
                factor = 2*self.len_s_space**(-2)
                if i == 0:
                    factor = factor/2
                if i == (self.len_p_space-1):
                    factor = factor/2
                z = int(z)
                self.Lambda_modes_list.append(
                        factor*np.polynomial.Chebyshev(coef=[0]*z + [1])(
                            Lambda_modes))

    def get_Lambdas(self):
        """
        we only need to multiply with a prefactor
        """
        self.Lambdas = list()
        for i in range(self.len_p_space):
            Lambda = self.Lambda_modes_list[i] * np.exp(self.tau_f.val[i])
            self.Lambdas.append(Lambda)

    def get_gradient_terms(self):
        """
        calculate the terms for the gradient wrt. tau_f and return them as a
            tuple
        """
        # tr(G Lambda)
        term1 = 0.5*nifty5.Field(
                domain=self.p_space,
                val=np.array([np.trace(self.G@self.Lambdas[i]) for i in range(
                        self.len_p_space)]))
        # y.G.Lambda.G.y
        term2 = -0.5*nifty5.Field.from_global_data(
                domain=self.p_space,
                arr=np.array([
                    self.y@self.G@self.Lambdas[i]@self.G@self.y
                    for i in range(self.len_p_space)]))

        term3 = self.smoothness_operator_f(self.position)

        return (
                self.term_factors[0]*term1,
                self.term_factors[1]*term2,
                self.term_factors[2]*term3)

    @property
    def gradient(self):

        gradient = sum(self.get_gradient_terms())

        return gradient

    @property
    def curvature(self):
        curvature = EffectCurvatureShallow(
            p_space=self.p_space,
            smoothness_operator_f=self.smoothness_operator_f)
        return curvature

    def at(self, position):
        return self.__class__(
                position=position,
                k=self.k,
                x=self.x,
                y=self.y,
                x_indices=self.x_indices,
                s_space=self.s_space,
                h_space=self.h_space,
                p_space=self.p_space,
                grid=self.grid,
                sigma_f=self.sigma_f,
                noise_variance=self.noise_variance,
                Lambda_modes_list=self.Lambda_modes_list,
                term_factors=self.term_factors,
                )

    def get_curvature_log_determinant(self, return_terms=0):

        # tr(-G.Lambda_z.G.Lambda_z' + G.Lambda_z.delta_zz')
        term1 = np.array([[np.trace(
                -self.G@self.Lambdas[i]@self.G@self.Lambdas[j] +
                (self.G@self.Lambdas[i] if i == j else 0))
            for i in range(self.len_p_space)]
            for j in range(self.len_p_space)])

        # y.(G.Lambda_z.G.Lambda_z'.G - G.Lambda.delta_zz').y
        term2 = np.array([[
            self.y@(
                2*self.G@self.Lambdas[i]@self.G@self.Lambdas[j]@self.G
                + (self.G@self.Lambdas[i]@self.G if i == j else 0))@self.y
            for i in range(self.len_p_space)]
                for j in range(self.len_p_space)])
        term3 = probe_operator(self.smoothness_operator_f)

        A = term1 + term2 + term3

        sign, ln_det = np.linalg.slogdet(A)

        return ln_det
