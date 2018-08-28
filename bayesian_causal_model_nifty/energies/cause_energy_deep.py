import sys
sys.path.append('../GlobalNewton')

import numpy as np

import nifty5
import global_newton

from ..curvatures.cause_curvature_deep import CausalCurvatureCauseFull
from ..utilities import probe_operator


class CauseEnergyDeep(nifty5.Energy):
    def __init__(
            self,
            position,
            k,
            grid,
            sigma_beta=1,
            rho=1,
            mode='multifield',
            single_fields=None,
            term_factors=[1]*5,
            ):

        super().__init__(position=position)
        self.domain = position.domain
        self.k = k
        self.sigma_beta = sigma_beta
        self.rho = rho
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

        self.get_del_B()

    @property
    def value(self):
        value_terms = self.get_value_terms()
        value = sum(value_terms)
        return value

    def get_value_terms(self):

        # log( det(2pi hat(e^tau_beta)))
        # using log det = tr log
        term1 = 0.5 * (np.log(2*np.pi) + self.tau_beta).sum()

        # -k^daggercasjbeta_vec
        term2 = -self.k@self.beta_vector

        # rho^dagger.e^beta_vec
        term3 = self.rho * np.sum(np.exp(self.beta_vector))

        # beta^dagger.B_inv[tau_beta].beta
        term4 = 0.5*self.B_inv.times(self.beta).vdot(self.beta)

        # tau_beta laplace laplace tau_beta
        term5 = 0.5*self.smoothness_operator_beta.times(self.tau_beta).vdot(
                self.tau_beta)

        terms = (
            self.term_factors[0]*term1,
            self.term_factors[1]*term2,
            self.term_factors[2]*term3,
            self.term_factors[3]*term4,
            self.term_factors[4]*term5,
            )

        return terms

    def get_del_B(self):
        self.del_B_inv_list = list()
        for i in range(self.len_p_space):
            p_spec = np.zeros(self.len_p_space)
            p_spec[i] = np.exp(-self.tau_beta.val[i])

            del_B_inv_h = nifty5.create_power_operator(
                    domain=self.h_space,
                    power_spectrum=nifty5.Field(
                        domain=self.p_space,
                        val=p_spec))

            del_B_inv = nifty5.SandwichOperator.make(
                    self.fft, del_B_inv_h)

            self.del_B_inv_list.append(del_B_inv)

    def get_gradient_beta_terms(self):
        # gradient wrt beta
        # -k
        term1_val = np.zeros(self.len_s_space)
        # rho(e^beta_vec)
        term2_val = np.zeros(self.len_s_space)
        for i, pos in enumerate(self.grid):
            term1_val[pos] = -self.k[i]
            term2_val[pos] = self.rho*np.exp(self.beta_vector[i])

        term1 = nifty5.Field(domain=self.s_space, val=term1_val)
        term2 = nifty5.Field(domain=self.s_space, val=term2_val)

        # beta.B^(-1)
        term3 = self.B_inv.adjoint_times(self.beta)
        return (
                self.term_factors[1]*term1,
                self.term_factors[2]*term2,
                self.term_factors[3]*term3)

    def get_gradient_tau_beta_terms(self):
        # gradient wrt tau_beta
        # 1/2
        term1 = nifty5.Field.from_global_data(domain=self.p_space, arr=0.5)

        # beta.del_B_inv.beta
        term2 = -0.5*nifty5.Field.from_global_data(
                domain=self.p_space, arr=np.array([self.beta.vdot(
                    self.del_B_inv_list[i].times(self.beta)) for i in
                    range(self.len_p_space)]))
        term3 = self.smoothness_operator_beta.times(self.tau_beta)
        return (
                self.term_factors[0]*term1,
                self.term_factors[3]*term2,
                self.term_factors[4]*term3)

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

        """
        super().__init__()
        self._domain = domain
        self.s_space = s_space
        self.p_space = p_space
        self.grid = grid
        self.beta = position.val['beta']
        self.tau_beta = position.val['tau_beta']
        self.tau_f = position.val['tau_f']
        self.eta = position.val['eta']
        self.cause_part = CausalCurvatureCausePart(
            s_space=self.s_space,
            p_space=self.p_space,
            grid=self.grid,
            B_inv=B_inv,
            del_B_inv_list=del_B_inv_list,
            smoothness_operator_beta=smoothness_operator_beta,
            rho=rho,
            position=position)
        IC = nifty5.GradientNormController(iteration_limit=100)
        self.inverter = global_newton.MultiConjugateGradient(controller=IC)
        self.preconditioner = preconditioner
        """
