import numpy as np
import scipy.ndimage

import nifty5

import sys
sys.path.append('../GlobalNewton')
from .bayesian_causal_model import probe_operator, BayesianCausalModel
import global_newton


class FullCausalModel(BayesianCausalModel):
    ...


class FullCausalEnergy(nifty5.Energy):
    def __init__(
            self,
            position,
            k,
            x,
            y,
            grid,
            sigma_f=1,
            sigma_beta=1,
            sigma_eta=1,
            rho=1,
            mode='multifield',
            single_fields=None,
            Lambda_modes_list=None,
            term_factors=[1]*11,
            ):

        super().__init__(position=position)
        self.domain = position.domain
        self.k, self.x, self.y = k, x, y
        self.N = len(self.x)
        self.sigma_beta = sigma_beta
        self.sigma_f = sigma_f
        self.sigma_eta = sigma_eta
        self.rho = rho
        self.mode = mode
        self.fields = single_fields
        if mode == 'multifield':
            self.beta = position.val['beta']
            self.tau_beta = position.val['tau_beta']
            self.tau_f = position.val['tau_f']
            self.eta = position.val['eta']
        else:
            self.fields[mode] = position
            self.beta = self.fields['beta']
            self.tau_beta = self.fields['tau_beta']
            self.tau_f = self.fields['tau_f']
            self.eta = self.fields['eta']
        self.s_space = self.beta.domain[0]
        self.h_space = self.s_space.get_default_codomain()
        self.p_space = self.tau_f.domain
        self.len_p_space = self.p_space.shape[0]
        self.len_s_space = self.s_space.shape[0]
        self.grid = grid
        self.grid_coordinates = [i*self.s_space.distances[0] for i in range(
            self.s_space.shape[0])]
        # beta_vector is the R^N_bins vector with beta field values at the grid
        #   positions
        self.beta_vector = np.array([self.beta.val[i] for i in self.grid])

        self.fft = nifty5.FFTOperator(domain=self.s_space, target=self.h_space)
        self.F_h = nifty5.create_power_operator(
            domain=self.h_space,
            power_spectrum=nifty5.exp(self.tau_f))
        self.B_inv_h = nifty5.create_power_operator(
            domain=self.h_space,
            power_spectrum=nifty5.exp(-self.tau_beta))
        self.B_inv = nifty5.SandwichOperator.make(self.fft, self.B_inv_h)
        self.F = nifty5.SandwichOperator.make(self.fft, self.F_h)
        self.F_matrix = probe_operator(self.F)
        self.F_tilde = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(i+1):
                self.F_tilde[i, j] = self.F_matrix[
                        self.x_indices[i], self.x_indices[j]]
                if i != j:
                    self.F_tilde[j, i] = self.F_matrix[
                            self.x_indices[i], self.x_indices[j]]

        self.exp_hat_eta_x = np.diag([np.exp(
            self.eta.val[self.x_indices[i]]) for i in range(self.N)])
        self.G = np.linalg.inv(self.F_tilde + self.exp_hat_eta_x)

        self.smoothness_operator_f = nifty5.SmoothnessOperator(
                domain=self.p_space,
                strength=1/self.sigma_f)
        self.smoothness_operator_beta = nifty5.SmoothnessOperator(
                domain=self.p_space,
                strength=1/self.sigma_beta)

        # second derivative of eta
        nabla_nabla_eta = np.zeros(self.eta.val.shape)
        scipy.ndimage.laplace(
                input=self.eta.val,
                output=nabla_nabla_eta)
        self.nabla_nabla_eta_field = nifty5.Field(
                domain=self.s_space,
                val=nabla_nabla_eta)
        if Lambda_modes_list is None:
            self.get_Lambda_modes(speedup=True)
        else:
            self.Lambda_modes_list = Lambda_modes_list
        self.term_factors = term_factors

        self.get_Lambdas()
        self.get_del_B()
        self.get_del_exp_eta_x()


"""
        # tr(-G.Lambda_z.G.Lambda_z' + G.Lambda_z.delta_zz')
        term1 = np.array([[
            np.trace(-self.G@self.Lambdas[i]@self.G@self.Lambdas[j]
            + (self.G@self.Lambdas[i] if i == j else 0))
            for i in range(self.len_p_space)]
            for j in range(self.len_p_space)])

        # y.(G.Lambda_z.G.Lambda_z'.G - G.Lambda.delta_zz').y
        term2 = np.array([[
            self.y@(
                2*self.G@self.Lambdas[i]@self.G@self.Lambdas[j]@self.G
                + (self.G@self.Lambdas[i]@self.G if i==j else 0))@self.y
                for i in range(self.len_p_space)]
                for j in range(self.len_p_space)])
        term3 = probe_operator(self.smoothness_operator_f)

        A = term1 + term2 + term3


        # tr(-G.del_exp_eta_x_z.G.Lambda_z' + G.del_exp_eta_x_z.delta_zz')
        term1 = np.array([[
            np.trace(-self.G@self.del_exp_eta_x_list[i]
                @self.G@self.del_exp_eta_x_list[j]
            + (self.G@self.del_exp_eta_x_list[i] if i == j else 0))
            for i in range(self.len_s_space)]
            for j in range(self.len_s_space)])

        # y.(G.del_exp_eta_x_z.G.del_exp_eta_x_z'.G - 
        G.del_exp_eta_x.delta_zz.G').y
        term2 = np.array([[
            self.y@(
                2*self.G@self.del_exp_eta_x_list[i]
                @self.G@self.del_exp_eta_x_list[j]@self.G
                + (self.G@self.del_exp_eta_x_list[i]@self.G
                if i==j else 0))@self.y
                for i in range(self.len_s_space)]
                for j in range(self.len_s_space)])

        # 1/sigma_eta nabla nabla
        term3 = np.zeros((self.len_s_space, self.len_s_space))

        for j in range(self.len_s_space):
            vector = np.zeros(self.len_s_space)
            vector[j] = 1

            scipy.ndimage.laplace(input=vector, output=term3[:, j])
        term3 = term3/self.sigma_eta

        D = term1 + term2 + term3

        term1 = np.array([[
            np.trace(-self.G@self.del_exp_eta_x_list[i]
                @self.G@self.Lambdas[j])
            for i in range(self.len_s_space)]
            for j in range(self.len_p_space)])

        # y.(G.del_exp_eta_x_z.G.del_exp_eta_x_z'.G
        - G.del_exp_eta_x.delta_zz.G').y
        term2 = np.array([[
            self.y@(
                self.G@self.del_exp_eta_x_list[i]
                @self.G@self.Lambdas[j]@self.G)@self.y
                for i in range(self.len_s_space)]
                for j in range(self.len_p_space)])
        B = term1 + term2

        sign, ln_det3 = np.linalg.slogdet(A)
        sign, ln_det4 = np.linalg.slogdet(D - B.T@np.linalg.inv(A)@B)

        ln_det = ln_det1 + ln_det2 + ln_det3 + ln_det4

        if return_terms:
            return ln_det, (ln_det1, ln_det2, ln_det3, ln_det4)

        else:
            return ln_det


class FullCausalCurvature(global_newton.MultiEndomorphicOperator):
    def __init__(
            self,
            domain,
            B_inv,
            s_space,
            p_space,
            grid,
            del_B_inv_list,
            smoothness_operator_beta,
            position,
            rho,
            preconditioner=None,
            ):

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

    @property
    def domain(self):
        return self._domain

    @property
    def capability(self):
        return self.TIMES | self.INVERSE_TIMES

    def 




"""
