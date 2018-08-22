import sys
sys.path.append('../GlobalNewton')

import numpy as np

import nifty5
import global_newton

from ..curvatures.cause_curvature_deep import CausalCurvatureCauseFull
from ..utilities import probe_operator


class EffectEnergyDeep(nifty5.Energy):
    def __init__(
            self,
            position,
            k,
            grid,
            sigma_f=1,
            mode='multifield',
            single_fields=None,
            term_factors=[1]*5,
            ):

        super().__init__(position=position)
        self.domain = position.domain
        self.k = k
        self.sigma_f = sigma_f
        self.mode = mode
        self.fields = single_fields
        self.term_factors = term_factors
        if mode == 'multifield':
            self.beta = position.val['beta']
            self.tau_beta = position.val['tau_beta']
        else:
            self.fields[mode] = position
            self.beta = self.fields['beta']
            self.tau_beta = self.fields['tau_beta']
        self.s_space = self.beta.domain[0]
        self.h_space = self.s_space.get_default_codomain()
        self.p_space = self.tau_beta.domain
        self.len_p_space = self.p_space.shape[0]
        self.len_s_space = self.s_space.shape[0]
        self.grid = grid
        self.grid_coordinates = [i*self.s_space.distances[0] for i in range(
            self.s_space.shape[0])]
        # beta_vector is the R^N_bins vector with beta field values at the grid
        #   positions
        self.beta_vector = np.array([self.beta.val[i] for i in self.grid])

        self.fft = nifty5.FFTOperator(domain=self.s_space, target=self.h_space)
        self.B_inv_h = nifty5.create_power_operator(
            domain=self.h_space,
            power_spectrum=nifty5.exp(-self.tau_beta))
        self.B_inv = nifty5.SandwichOperator.make(self.fft, self.B_inv_h)

        self.smoothness_operator_beta = nifty5.SmoothnessOperator(
                domain=self.p_space,
                strength=1/self.sigma_beta)

    def get_Lambdas(self):
        """
        we only need to multiply with a prefactor
        """
        self.Lambdas = list()
        for i in range(self.len_p_space):
            Lambda = self.Lambda_modes_list[i] * np.exp(self.tau_f.val[i])
            self.Lambdas.append(Lambda)

    def get_Lambdas_old(self):
        """
        convenience function to get an array of Lambda(z) matrices for each
        z \in p_space.
        by using a seperate method for this we can achieve better profiling
        """
        self.Lambdas = list()
        for k in range(self.len_p_space):
            p_spec = np.zeros(self.len_p_space)
            p_spec[k] = np.exp(self.tau_f.val[k])
            Lambda_h = nifty5.create_power_operator(
                    domain=self.h_space,
                    power_spectrum=nifty5.Field(
                        domain=self.p_space, val=p_spec))
            Lambda_kernel = nifty5.SandwichOperator.make(self.fft, Lambda_h)
            Lambda_kernel_matrix = probe_operator(Lambda_kernel)
            Lambda = np.zeros((self.N, self.N))
            for i in range(self.N):
                for j in range(i+1):
                    Lambda[i, j] = Lambda_kernel_matrix[
                            self.x_indices[i], self.x_indices[j]]
                    if i != j:
                        Lambda[j, i] = Lambda_kernel_matrix[
                                self.x_indices[i], self.x_indices[j]]
            self.Lambdas.append(Lambda)

    def get_Lambda_modes(self, speedup=False):
        # this is an array of matrices:
        #   Lambda_mode(z)_ij = (1/2pi) (cos(z(x_i-x_j)) + sin(z(x_i + x_j)) )
        self.Lambda_modes_list = list()
        if not speedup:
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
                    # we need to take cos((x-y)*2*pi) we get the desired result
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

    def get_del_exp_eta_x(self):
        self.del_exp_eta_x_list = list()
        for i in range(self.len_s_space):
            # first get the all indices of x samples, where x==z by comparing
            #   the indices
            relevant_x_indices = np.argwhere(self.x_indices == i)
            # construct the derivative of the covariance matrix hat(exp(eta(x)))
            #   which is again a diagonal matrix, only having non-zero entries
            #   at the indices of x==z
            diag = np.zeros(self.N)
            for idx in relevant_x_indices:
                diag[idx] = np.exp(self.eta.val[i])
            del_exp_eta_x = np.diag(diag)
            self.del_exp_eta_x_list.append(del_exp_eta_x)

    @property
    def value(self):
        value_terms = self.get_value_terms()
        value = sum(value_terms)
        return value

    def get_value_terms(self):

        # log( det( F_tilde[tau_f] + hat(e^eta(x))))
        sign, log_det = np.linalg.slogdet(self.F_tilde + self.exp_hat_eta_x)
        if sign <= 0:
            print('warning, computed sign {} in ln_det_DB'.format(sign))
        term1 = 0.5 * log_det

        # y^dagger.(F_tilde + hat(e^eta(x)))^-1.y
        term2 = 0.5*self.y@self.G@self.y

        # one^dagger.eta(x)
        eta_x = [self.eta.val[self.x_indices[i]] for i in range(self.N)]
        term3 = -0.5*np.sum(eta_x)

        # tau_f.laplace.laplace.tau_f
        term4 = 0.5*self.smoothness_operator_f(self.tau_f).vdot(self.tau_f)

        # eta^dagger.nabla^dagger.nabla.eta / sigma_eta
        term5 = 0.5*self.nabla_nabla_eta_field.vdot(self.eta)/self.sigma_eta

        terms = (
            self.term_factors[0]*term1,
            self.term_factors[1]*term2,
            self.term_factors[2]*term3,
            self.term_factors[3]*term4,
            self.term_factors[4]*term5,
            )

        return terms

    @property
    def gradient(self):

        if self.mode == 'multifield':
            gradient_beta = sum(self.get_gradient_beta_terms())
            gradient_tau_beta = sum(self.get_gradient_tau_beta_terms())

            gradients = {
                    'beta': gradient_beta,
                    'tau_beta': gradient_tau_beta,
                    }
            gradient = global_newton.MultiField(
                    self.position.domain, val=gradients)
        elif self.mode == 'beta':
            gradient = sum(self.get_gradient_beta_terms())
        elif self.mode == 'tau_beta':
            gradient = sum(self.get_gradient_tau_beta_terms())
        else:
            print('invalid mode')

        return gradient

    def get_gradient_tau_f_terms(self):
        # gradient wrt tau_f
        # tr(G Lambda)
        term1 = 0.5*nifty5.Field(
                domain=self.p_space,
                val=np.array([np.trace(self.G@self.Lambdas[i]) for i in range(
                        self.len_p_space)]))
        # y.G.Lambda.G.y
        term2 = -0.5*nifty5.Field(
                domain=self.p_space,
                val=np.array([
                    self.y@self.G@self.Lambdas[i]@self.G@self.y
                    for i in range(self.len_p_space)]))

        term3 = self.smoothness_operator_f.times(self.tau_f)

        return (
                self.term_factors[0]*term1,
                self.term_factors[6]*term2,
                self.term_factors[9]*term3)

    def get_gradient_eta_terms(self):
        # gradient wrt eta
        # tr( G e^eta(x) delta_x)

        term1 = 0.5*nifty5.Field(
                domain=self.s_space,
                val=np.array([
                    np.trace(self.G@self.del_exp_eta_x_list[i])
                    for i in range(self.len_s_space)]))

        term2 = -0.5*nifty5.Field(
                domain=self.s_space,
                val=np.array([
                    self.y@self.G@self.del_exp_eta_x_list[i]@self.G@self.y
                    for i in range(self.len_s_space)]))
        term3_val = np.zeros(self.len_s_space)
        for i, pos in enumerate(self.x_indices):
            term3_val[pos] -= 0.5
        term3 = nifty5.Field(domain=self.s_space, val=term3_val)
        term4 = self.nabla_nabla_eta_field / self.sigma_eta
        return (
                self.term_factors[0]*term1,
                self.term_factors[6]*term2,
                self.term_factors[7]*term3,
                self.term_factors[10]*term4)

    @property
    def curvature(self):
        curvature = CausalCurvatureCauseFull(
            domain=self.domain,
            grid=self.grid,
            B_inv=self.B_inv,
            del_B_inv_list=self.del_B_inv_list,
            smoothness_operator_beta=self.smoothness_operator_beta,
            rho=self.rho,
            position=self.position)
        return curvature

    def at(self, position):
        return self.__class__(
                position=position,
                k=self.k,
                grid=self.grid,
                sigma_beta=self.sigma_beta,
                mode=self.mode,
                single_fields=self.fields,
                term_factors=self.term_factors,
                )

    def get_curvature_log_determinant(self, return_terms=0):
        # we can use det((A B)(B^T D)) = det(A)det(D - B^T A^-1 B)
        # (hat(rho e^beta_vec) + B_inv)
        B_inv_matrix = probe_operator(self.B_inv)

        rho_exp_beta = np.zeros(self.len_s_space)
        for i, pos in enumerate(self.grid):
            rho_exp_beta[pos] = (
                    self.rho*np.exp(self.beta.val[pos]))
        hat_rho_exp_beta_matrix = np.diag(rho_exp_beta)
        A = B_inv_matrix + hat_rho_exp_beta_matrix

        # (delta_{zz'} beta.del_B_inv_z.beta + smoothing_beta)
        delta_beta_del_B_inv_beta = np.diag([self.beta.vdot(
            del_B_inv.times(self.beta)) for del_B_inv in self.del_B_inv_list])

        smoothing_beta_matrix = probe_operator(
                self.smoothness_operator_beta)
        D = delta_beta_del_B_inv_beta + smoothing_beta_matrix

        # -beta.del_B_inv
        B = -np.array(
                [del_B_inv.adjoint_times(self.beta).val for del_B_inv in
                    self.del_B_inv_list])
        sign, ln_det1 = np.linalg.slogdet(A)
        sign, ln_det2 = np.linalg.slogdet(D - B@np.linalg.inv(A)@B.T)

        ln_det = ln_det1 + ln_det2

        if return_terms:
            return ln_det, (ln_det1, ln_det2)

        else:
            return ln_det
