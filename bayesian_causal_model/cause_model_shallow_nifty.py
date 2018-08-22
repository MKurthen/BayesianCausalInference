import datetime

import numpy as np
import nifty5
from nifty5.sugar import exp

from .bayesian_causal_model import BayesianCausalModel
from .energies.cause_energy_shallow import CauseEnergyShallow
from .utilities import probe_operator


class CausalModelCauseShallow(BayesianCausalModel):
    """
    bayesian causal model where the causal direction is inferred. Here with
        given power spectra and noise variance. A Laplace approximation via
        minimization is being used for the the cause distribution field beta.
    """
    def __init__(
            self,
            N_pix=1024,
            power_spectrum_f=lambda k: 1/(k**4+1),
            power_spectrum_beta=lambda k: 1/(k**4+1),
            noise_var=0.1,
            rho=1,
            minimization=None,
            minimizer=None,
            ):
        super().__init__(
                N_pix=N_pix)
        # because of how nifty implements the FFT, we have to multiply the 
        #   amplitude by N_pix
        self.power_spectrum_f = lambda q: power_spectrum_f(q)*N_pix
        self.power_spectrum_beta = lambda q: power_spectrum_beta(q)*N_pix
        self.noise_var = noise_var
        self.rho = rho
        if minimizer is None:
            minimizer = nifty5.VL_BFGS(
                    controller=nifty5.GradientNormController(
                        tol_abs_gradnorm=1,
                        iteration_limit=100))
        self.minimizer = minimizer

    def get_evidence(
            self,
            direction=1,
            grid_size=None,
            verbosity=0,
            return_terms=False,
            beta_init=None,
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
            k = nifty5.Field.from_global_data(
                    domain=self.s_space, arr=self.k_x)
            log_prod_k_fact = self.log_prod_k_x_fact
        elif direction == -1:
            k = nifty5.Field.from_global_data(
                    domain=self.s_space, arr=self.k_y)
            log_prod_k_fact = self.log_prod_k_y_fact
        else:
            raise Exception('invalid direction')
        # TEST
        # log_prod_k_fact = 0

        # initial guess for beta is proportional to k
        if beta_init is None:
            beta_init = nifty5.Field.from_global_data(
                domain=self.s_space,
                arr=(np.log(k.val/self.rho + 1e-3)),
                )

        energy = CauseEnergyShallow(
                beta_init,
                k,
                s_space=self.s_space,
                power_spectrum_beta=self.power_spectrum_beta,
                rho=self.rho,
                )

        self.energy_cause_min = self.minimizer(energy)[0]

        self.DB = DBOperator(
                domain=self.s_space,
                beta=self.energy_cause_min.position,
                k=k,
                grid=self.grid,
                power_spectrum=self.power_spectrum_beta,
                rho=self.rho
                )

        # get the explicit matrix for DB, the gamma curvature operator * B
        if verbosity > 0:
            print('probing the curvature')
        self.DB_matrix = probe_operator(self.DB)

        # (1/2)*ln(det(D))
        sign, ln_det_DB = np.linalg.slogdet(self.DB_matrix)
        if sign <= 0:
            print('warning, computed sign {} in ln_det_DB'.format(sign))

        energy_terms = self.energy_cause_min.get_value_terms()
        effect_terms = self.get_effect_terms(direction=direction)

        # compute the evidence (x,y | X->Y)
        evidence = (
                0.5*ln_det_DB +
                log_prod_k_fact +
                sum(energy_terms) +
                sum(effect_terms)
                )

        terms = [
                ('1/2 ln(det(DB))', 0.5*ln_det_DB),
                ('ln(prod(k_j!))', log_prod_k_fact),
                ('-k^dagger.beta_0', energy_terms[0]),
                ('rho(One.e^beta_0)', energy_terms[1]),
                ('1/2 beta_0.(B^-1).beta_0', energy_terms[2]),
                ('1/2 y.((F_tilde + N)^-1).y', effect_terms[0]),
                ('1/2 ln(det(F_tilde + N))', effect_terms[1]),
                ]

        if verbosity > 0:
            """
            print(
                    'numerical values of terms: \n'
                    '   1/2 ln(|DB|):                  {:.2e}\n'
                    '   -k^dagger.beta_vec:           {:.2e}\n'
                    '   rho(One.e^beta_0_vec):        {:.2e}\n'
                    '   1/2 beta_0.B^-1.beta_0:       {:.2e}\n'
                    '   ln(prod(k_j!)):               {:.2e}\n'
                    '   1/2 y^T.((F_tilde + N)^-1).y: {:.2e}\n'
                    '   1/2 ln(|F_tilde + N|):        {:.2e}'.format(
                        0.5*ln_det_DB,
                        energy_terms[0], energy_terms[1], energy_terms[2],
                        log_prod_k_fact,
                        effect_terms[0], effect_terms[1],))

            """
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            details = {
                    'timestamp': timestamp,
                    'direction': direction,
                    'terms': terms
                    }

            with open('./details.txt', 'a') as f:
                f.write(str(details) + '\n')

        if return_terms:
            return evidence, terms
        else:
            return evidence


class DBOperator(
        nifty5.LinearOperator):
    def __init__(
            self,
            domain,
            beta,
            k,
            grid,
            power_spectrum=lambda q: 2/(q**4 + 1),
            rho=1,
            verbosity=0):
        super().__init__()
        self.beta = beta
        self._domain = (domain, )
        self.fft = nifty5.FFTOperator(self._domain)
        self.h_space = self.fft.target[0]
        self.grid = grid
        self.k = k
        self.rho = rho
        B_h = nifty5.create_power_operator(
                domain=self.h_space, power_spectrum=power_spectrum)
        self.B = nifty5.SandwichOperator.make(self.fft, B_h)
        # the diagonal operator rho*e^beta

        rho_e_beta = np.zeros(domain.shape[0])
        for i, pos in enumerate(self.grid):
            rho_e_beta[pos] = (
                    self.rho*np.exp(self.beta.val[pos]))
        rho_e_beta_field = nifty5.Field(domain=domain, val=rho_e_beta)
        self.rho_e_beta_diag = nifty5.DiagonalOperator(
                domain=self._domain,
                diagonal=rho_e_beta_field)

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

    @property
    def capability(self):
        # TODO: use InversionEnabler here
        return 1

    def apply(self, x, mode):
        if mode == 1:
            term1 = self.rho * self.B(exp(self.beta)*x)
            return term1 + x
