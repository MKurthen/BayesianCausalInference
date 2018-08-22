import numpy as np

import nifty5

from .bayesian_causal_model import BayesianCausalModel
from .cause_model_shallow import DBOperator
from .energies.cause_energy_shallow import CauseEnergyShallow
from .energies.effect_energy_shallow import EffectEnergyShallow
from .utilities import probe_operator


class CausalModelEffectShallow(BayesianCausalModel):
    """
    model to infer causal direction via bayesian hierarchic approach, where
        the covariance for the function is marginalized out but the noise
        variance and the beta powerspectrum are given
    """
    def __init__(
            self,
            N_pix=1024,
            power_spectrum_beta=lambda k: 2/(k**4+1),
            noise_variance=0.01,
            rho=1,
            sigma_f=1,
            minimizer=None,
            controller=None,
            beta_init=None,
            tau_f_init=None,
            ):

        super().__init__(
                N_pix=N_pix)
        self.power_spectrum_beta = power_spectrum_beta
        self.noise_variance = noise_variance
        self.rho = rho
        self.sigma_f = sigma_f
        if controller is None:
            controller = nifty5.GradientNormController(
                    tol_rel_gradnorm=1e-2,
                    iteration_limit=500)
        if minimizer is None:
            minimizer = nifty5.VL_BFGS(controller=controller)
        self.minimizer = minimizer
        self.beta_init = beta_init
        self.tau_f_init = tau_f_init

    def get_evidence(
            self,
            direction=1,
            beta_init=None,
            infer_power_spectrum_beta=False,
            infer_power_spectrum_f=False,
            infer_noise_variance=False,
            grid_size=None,
            verbosity=0,
            return_terms=False,
            ):
        """
        computes the Hamiltonian of the evidence (d|X->Y), where d = (x,y)

        Parameters:
        ----------
        x : np.array
            samples of the cause variable
        y: np.array
            samples of the effect variable
        grid_size: int
            number of grid points z1, ..., z_r
        verbosity : int
            controls verbosity of output

        Returns:
        --------
        evidence : float
        """

        if direction == 1:
            k = self.k_x
            cause_samples = self.x
            cause_sample_indices = self.x_indices
            effect_samples = self.y
            log_prod_k_fact = self.log_prod_k_x_fact

        elif direction == -1:
            k = self.k_y
            cause_samples = self.y
            cause_sample_indices = self.y_indices
            effect_samples = self.x
            log_prod_k_fact = self.log_prod_k_y_fact

        else:
            raise Exception('invalid direction')

        if beta_init is None:
            beta_init = nifty5.Field.from_global_data(
                domain=self.s_space,
                arr=(k/self.power_spectrum_beta(0)**4))

        if self.tau_f_init is None:
            self.tau_f_init = nifty5.Field.from_global_data(
                domain=self.p_space,
                arr=np.array(
                    [1/(k**4+1) for k in range(self.p_space.shape[0])]))

        energy_cause = CauseEnergyShallow(
                beta_init,
                k,
                grid=self.grid,
                s_space=self.s_space,
                power_spectrum_beta=self.power_spectrum_beta,
                rho=self.rho,
                )

        energy_effect = EffectEnergyShallow(
                self.tau_f_init,
                k=k,
                x=cause_samples,
                y=effect_samples,
                x_indices=cause_sample_indices,
                s_space=self.s_space,
                h_space=self.h_space,
                p_space=self.p_space,
                grid=self.grid,
                noise_variance=self.noise_variance,
                sigma_f=self.sigma_f,
                )

        self.energy_cause_min = self.minimizer(energy_cause)[0]
        self.energy_effect_min = self.minimizer(energy_effect)[0]

        energy_cause_terms = self.energy_cause_min.get_value_terms()
        energy_effect_terms = self.energy_effect_min.get_value_terms()

        # ln det (D)
        ln_det_curvature_effect = (
                self.energy_effect_min.get_curvature_log_determinant())

        # instead of the cause curvature determinant, log det D, get log det DB
        DB = DBOperator(
                domain=self.s_space,
                beta=self.energy_cause_min.position,
                k=k,
                grid=self.grid,
                power_spectrum=self.power_spectrum_beta,
                rho=self.rho
                )

        if verbosity > 0:
            print('probing the curvature')
        DB_matrix = probe_operator(DB)

        # (1/2)*ln(det(D))
        sign, ln_det_DB = np.linalg.slogdet(DB_matrix)
        if sign <= 0:
            print('warning, computed sign {} in ln_det_DB'.format(sign))

        terms = {
                'ln_det_DB': ln_det_DB,
                'log_prod_k_fact': log_prod_k_fact,
                'energy_cause_term_1': energy_cause_terms[0],
                'energy_cause_term_2': energy_cause_terms[1],
                'energy_cause_term_3': energy_cause_terms[2],
                'energy_effect_term_1': energy_effect_terms[0],
                'energy_effect_term_2': energy_effect_terms[1],
                'energy_effect_term_3': energy_effect_terms[2],
                }
        evidence = sum([
                sum(energy_cause_terms),
                sum(energy_effect_terms),
                0.5*ln_det_DB,
                0.5*ln_det_curvature_effect,
                2*log_prod_k_fact])

        if verbosity > 0:
            print(
                    'numerical values of terms: \n'
                    '   1/2 ln(|D_cause.B|):          {:.2e}\n'
                    '   1/2 ln(|D_effect|):           {:.2e}\n'
                    '   -k^dagger.beta_vec:           {:.2e}\n'
                    '   rho(One.e^beta_0_vec):        {:.2e}\n'
                    '   1/2 beta_0.B^-1.beta_0:       {:.2e}\n'
                    '   ln(prod(k_j!)):               {:.2e}\n'
                    '   1/2 y^T.((F_tilde + N)^-1).y: {:.2e}\n'
                    '   1/2 ln(|F_tilde + N|):        {:.2e}\n'
                    '   1/2 laplace.laplace.tau_f     {:.2e}'.format(
                        0.5*ln_det_DB,
                        0.5*ln_det_curvature_effect,
                        energy_cause_terms[0],
                        energy_cause_terms[1],
                        energy_cause_terms[2],
                        log_prod_k_fact,
                        energy_effect_terms[0],
                        energy_effect_terms[1],
                        energy_effect_terms[2],
                        ))
        if return_terms:
            return evidence, terms
        else:
            return evidence

    def minimize_beta(
            self,
            energy,
            k,
            grid=None,
            beta_init=None,
            verbosity=0,
            minimization=None,
            callback=None,
            ):
        """
        get beta, s.th. the energy functional is minimized
        """

        return evidence
