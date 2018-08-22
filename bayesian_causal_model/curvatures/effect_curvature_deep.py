import sys
sys.path.append('../GlobalNewton')

import numpy as np

import nifty5



class CausalCurvatureEffectPart(global_newton.MultiEndomorphicOperator):
    def __init__(
            self,
            s_space,
            p_space,
            grid,
            smoothness_operator_f,
            position,
            rho,
            ):
        super().__init__()
        self.s_space = s_space
        self.len_s_space = s_space.shape[0]
        self.p_space = p_space
        self.grid = grid
        self.beta = position.val['beta']
        self.tau_beta = position.val['tau_beta']
        self.tau_f = position.val['tau_f']
        self.eta = position.val['eta']
        self.rho=rho

    @property
    def capability(self):
        return self.TIMES

    def apply(self, x, mode):
        self._check_mode(mode=mode)
        if mode == self.TIMES:
            x_tau_f = x.val['tau_f']
            x_eta = x.val['eta']

            # tr(-G.del_exp_eta_x_z.G.Lambda_z' + G.del_exp_eta_x_z.delta_zz')
            term1 = np.array([[
                np.trace(-self.G@self.del_exp_eta_x_list[i]
                    @self.G@self.del_exp_eta_x_list[j]
                + (self.G@self.del_exp_eta_x_list[i] if i == j else 0))
                for i in range(self.len_s_space)] 
                for j in range(self.len_s_space)])

            # y.(G.del_exp_eta_x_z.G.del_exp_eta_x_z'.G - G.del_exp_eta_x.delta_zz.G').y
            term2 = np.array([[
                self.y@(
                    2*self.G@self.del_exp_eta_x_list[i]
                    @self.G@self.del_exp_eta_x_list[j]@self.G
                    + (self.G@self.del_exp_eta_x_list[i]@self.G if i==j else 0))@self.y
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

            # y.(G.del_exp_eta_x_z.G.del_exp_eta_x_z'.G - G.del_exp_eta_x.delta_zz.G').y
            term2 = np.array([[
                self.y@(
                    self.G@self.del_exp_eta_x_list[i]
                    @self.G@self.Lambdas[j]@self.G)@self.y
                    for i in range(self.len_s_space)]
                    for j in range(self.len_p_space)])
            B = term1 + term2

            result_fields= {
                'beta': term_beta_1 + term_beta_2 + term_beta_3,
                'tau_beta': term_tau_beta_1 + term_tau_beta_2 + term_tau_beta_3,
                'tau_f': nifty5.Field.zeros(self.p_space),
                'eta': nifty5.Field.zeros(self.s_space),
            }

            result = global_newton.MultiField(domain=x.domain, val=result_fields)
            return result
