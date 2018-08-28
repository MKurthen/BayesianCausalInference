import numpy as np

import nifty5

import sys
sys.path.append('/afs/mpa/home/maxk/causality/GlobalNewton')
import global_newton
from .bayesian_causal_model import BayesianCausalModel
from .energies.cause_energy_deep import CauseEnergyDeep
from .utilities import probe_operator


class CausalModelCauseDeep(BayesianCausalModel):
    def __init__(
            self,
            N_pix=1024,
            power_spectrum_f=lambda k: 2/(k**4+1),
            noise_var=0.1,
            rho=1,
            sigma_beta=1,
            minimizer=None,
            controller=None,
            beta_init=None,
            tau_beta_init=None,
            ):

        super().__init__(
                N_pix=N_pix)
        self.power_spectrum_f = power_spectrum_f
        self.noise_var = noise_var
        self.rho = rho
        self.sigma_beta = sigma_beta
        if controller is None:
            controller = nifty5.GradientNormController(
                    tol_rel_gradnorm=1e-2,
                    iteration_limit=500)
        if minimizer is None:
            minimizer = nifty5.VL_BFGS(controller=controller)
        self.minimizer = minimizer

    def get_evidence(
            self,
            direction=1,
            infer_power_spectrum_beta=False,
            infer_power_spectrum_f=False,
            infer_noise_variance=False,
            grid_size=None,
            callback=None,
            verbosity=0,
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
            log_prod_k_fact = self.log_prod_k_x_fact

        elif direction == -1:
            k = self.k_y
            log_prod_k_fact = self.log_prod_k_y_fact

        else:
            raise Exception('invalid direction')

        beta_init = nifty5.Field.from_global_data(
            domain=self.s_space, arr=(k/np.max(k)))

        tau_beta_init = nifty5.Field.from_global_data(
                domain=self.p_space,
                arr=np.array(
                    [1/(k**4+1) for k in range(self.p_space.shape[0])]))

        multi_domain = global_newton.MultiDomain({
            'beta': self.s_space,
            'tau_beta': self.p_space,
        })

        fields = {
                'beta': beta_init,
                'tau_beta': tau_beta_init,
        }

        init_field = global_newton.MultiField(multi_domain, val=fields)

        energy = CauseEnergyDeep(
                init_field,
                k,
                sigma_beta=self.sigma_beta,
                grid=self.grid,
                rho=self.rho,
                )

        self.energy_min = self.minimizer(energy)[0]

        energy_terms = self.energy_min.get_value_terms()
        effect_terms = self.get_effect_terms(direction=direction)

        # (1/2) ln det (D)
        ln_det_D = self.energy_min.get_curvature_log_determinant()

        evidence = (sum(energy_terms) + sum(effect_terms) + ln_det_D +
                    log_prod_k_fact)
        return evidence
