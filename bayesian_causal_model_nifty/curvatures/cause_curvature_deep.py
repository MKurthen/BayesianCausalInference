import sys
sys.path.append('../GlobalNewton')

import numpy as np

import nifty5
import global_newton


class CausalCurvatureCauseFull(global_newton.MultiEndomorphicOperator):
    def __init__(
            self,
            domain,
            grid,
            B_inv,
            del_B_inv_list,
            smoothness_operator_beta,
            position,
            rho,
            ):
        super().__init__()
        self._domain = domain
        self.s_space = domain.domains['beta']
        self.len_s_space = self.s_space.shape[0]
        self.p_space = domain.domains['tau_beta']
        self.grid = grid
        self.B_inv = B_inv
        self.del_B_inv_list = del_B_inv_list
        self.smoothness_operator_beta = smoothness_operator_beta
        self.beta = position.val['beta']
        self.tau_beta = position.val['tau_beta']
        self.rho = rho
        self.beta_curvature = BetaCurvature(
                s_space=self.s_space,
                p_space=self.p_space,
                grid=self.grid,
                B_inv=self.B_inv,
                position=position,
                rho=rho)
        self.tau_beta_curvature = TauBetaCurvature(
                s_space=self.s_space,
                p_space=self.p_space,
                grid=self.grid,
                del_B_inv_list=self.del_B_inv_list,
                smoothness_operator_beta=self.smoothness_operator_beta,
                position=position,
                )
        self.beta_tau_beta_curvature = BetaTauBetaMixedCurvature(
                s_space=self.s_space,
                p_space=self.p_space,
                grid=self.grid,
                del_B_inv_list=self.del_B_inv_list,
                position=position,
                )
        self.tau_beta_beta_curvature = TauBetaBetaMixedCurvature(
                s_space=self.s_space,
                p_space=self.p_space,
                grid=self.grid,
                del_B_inv_list=self.del_B_inv_list,
                position=position,
                )

    @property
    def domain(self):
        return self._domain

    @property
    def capability(self):
        return self.TIMES | self.INVERSE_TIMES

    def apply(self, x, mode):
        self._check_mode(mode=mode)
        if mode == self.TIMES:
            result = (
                    self.beta_curvature.times(x) +
                    self.beta_tau_beta_curvature.times(x) +
                    self.tau_beta_beta_curvature.times(x) +
                    self.tau_beta_curvature.times(x)
                    )

        else:
            x0 = global_newton.MultiField.zeros(self.domain)
            energy = nifty5.QuadraticEnergy(A=self.times, b=x, position=x0)
            r = self.inverter(energy, preconditioner=self.preconditioner)[0]
            result = r.position
        return result


class BetaCurvature(global_newton.MultiEndomorphicOperator):
    def __init__(
            self,
            s_space,
            p_space,
            grid,
            B_inv,
            position,
            rho,
            ):
        super().__init__()
        self.s_space = s_space
        self.len_s_space = s_space.shape[0]
        self.p_space = p_space
        self.grid = grid
        self.B_inv = B_inv
        self.beta = position.val['beta']
        self.tau_beta = position.val['tau_beta']
        self.rho = rho

    @property
    def capability(self):
        return self.TIMES

    @property
    def domain(self):
        return self._domain

    def apply(self, x, mode):
        self._check_mode(mode=mode)
        if mode == self.TIMES:
            x_beta = x.val['beta']

            # application of operator to beta field
            term_beta_1 = self.B_inv.times(x_beta)

            rho_exp_beta = np.zeros(self.len_s_space)
            for i, pos in enumerate(self.grid):
                rho_exp_beta[pos] = (
                        self.rho*np.exp(self.beta.val[pos]))
            rho_exp_beta_field = nifty5.Field.from_global_data(
                    domain=self.s_space, arr=rho_exp_beta)
            term_beta_2 = rho_exp_beta_field * x_beta

            result_fields = {
                'beta': term_beta_1 + term_beta_2,
                'tau_beta': nifty5.Field.zeros(domain=self.p_space),
            }

            result = global_newton.MultiField(
                    domain=x.domain, val=result_fields)
            return result


class TauBetaCurvature(global_newton.MultiEndomorphicOperator):
    def __init__(
            self,
            s_space,
            p_space,
            grid,
            del_B_inv_list,
            smoothness_operator_beta,
            position,
            ):
        super().__init__()
        self.s_space = s_space
        self.len_s_space = s_space.shape[0]
        self.p_space = p_space
        self.grid = grid
        self.del_B_inv_list = del_B_inv_list
        self.smoothness_operator_beta = smoothness_operator_beta
        self.beta = position.val['beta']
        self.tau_beta = position.val['tau_beta']

    @property
    def capability(self):
        return self.TIMES

    @property
    def domain(self):
        return self._domain

    def apply(self, x, mode):
        self._check_mode(mode=mode)
        if mode == self.TIMES:
            x_tau_beta = x.val['tau_beta']

            # (delta_{zz'} beta.del_B_inv_z.beta + smoothing_beta)
            field_for_diagonal = nifty5.Field.from_local_data(
                    domain=self.p_space, arr=np.array([self.beta.vdot(
                        del_B_inv.times(self.beta))
                        for del_B_inv in self.del_B_inv_list]))
            delta_beta_del_B_inv_beta = nifty5.DiagonalOperator(
                    domain=self.p_space, diagonal=field_for_diagonal)
            term_tau_beta_1 = delta_beta_del_B_inv_beta.times(x_tau_beta)
            term_tau_beta_2 = self.smoothness_operator_beta.times(x_tau_beta)

            result_fields = {
                'beta': nifty5.Field.zeros(domain=self.s_space),
                'tau_beta': term_tau_beta_1 + term_tau_beta_2,
            }

            result = global_newton.MultiField(
                    domain=x.domain, val=result_fields)
            return result


class BetaTauBetaMixedCurvature(global_newton.MultiEndomorphicOperator):
    def __init__(
            self,
            s_space,
            p_space,
            grid,
            del_B_inv_list,
            position,
            ):
        super().__init__()
        self.s_space = s_space
        self.len_s_space = s_space.shape[0]
        self.p_space = p_space
        self.grid = grid
        self.del_B_inv_list = del_B_inv_list
        self.beta = position.val['beta']
        self.tau_beta = position.val['tau_beta']

    @property
    def capability(self):
        return self.TIMES

    @property
    def domain(self):
        return self._domain

    def apply(self, x, mode):
        self._check_mode(mode=mode)
        if mode == self.TIMES:
            x_tau_beta = x.val['tau_beta']

            # -beta.del_B_inv applied to tau_beta
            term_beta = nifty5.Field.from_global_data(
                    domain=self.s_space, arr=np.array([
                        nifty5.Field.from_global_data(
                            domain=self.p_space, arr=np.array([
                                B_inv.adjoint_times(self.beta).val[i]
                                for B_inv in self.del_B_inv_list])).vdot(
                                    x_tau_beta)
                        for i in range(self.s_space.shape[0])]))

            result_fields = {
                'beta': term_beta,
                'tau_beta': nifty5.Field.zeros(domain=self.p_space)
            }

            result = global_newton.MultiField(
                    domain=x.domain, val=result_fields)
            return result


class TauBetaBetaMixedCurvature(global_newton.MultiEndomorphicOperator):
    def __init__(
            self,
            s_space,
            p_space,
            grid,
            del_B_inv_list,
            position,
            ):
        super().__init__()
        self.s_space = s_space
        self.len_s_space = s_space.shape[0]
        self.p_space = p_space
        self.grid = grid
        self.del_B_inv_list = del_B_inv_list
        self.beta = position.val['beta']
        self.tau_beta = position.val['tau_beta']

    @property
    def capability(self):
        return self.TIMES

    @property
    def domain(self):
        return self._domain

    def apply(self, x, mode):
        self._check_mode(mode=mode)
        if mode == self.TIMES:
            x_beta = x.val['beta']

            # -beta.del_B_inv applied to beta
            term_tau_beta = nifty5.Field.from_global_data(
                    domain=self.p_space,
                    arr=np.array(
                        [-del_B_inv.adjoint_times(self.beta).vdot(x_beta)
                         for del_B_inv in self.del_B_inv_list]))

            result_fields = {
                'beta': nifty5.Field.zeros(domain=self.p_space),
                'tau_beta': term_tau_beta,
            }

            result = global_newton.MultiField(
                    domain=x.domain, val=result_fields)
            return result
