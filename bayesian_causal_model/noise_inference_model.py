import numpy as np

from .cause_model_shallow_numpy import CausalModelShallow

class CausalModelNoiseInference(CausalModelShallow):
    def __init__(
            self,
            N_pix=1024,
            power_spectrum_f=lambda k: 1/(k**4+1),
            power_spectrum_beta=lambda k: 1/(k**4+1),
            sigma_eta=1,
            rho=1,
            ):
        super().__init__(
                N_pix=N_pix,
                power_spectrum_f=power_spectrum_f,
                power_spectrum_beta=power_spectrum_beta,
                rho=rho)

        self.power_spectrum_f = power_spectrum_f
        self.power_spectrum_beta = power_spectrum_beta
        self.sigma_eta = sigma_eta
        self.rho = rho
        # calculate Discrete Hartley Transform Matrix
        DFT = scipy.linalg.dft(N_pix, scale='sqrtn')
        DHT = np.real(DFT) + np.imag(DFT)
        self.B_h = np.diag([
            (power_spectrum_beta(q)/N_pix)
            for q in list(range(N_pix//2 + 1)) + list(range(N_pix//2 - 1, 0, -1))])
        self.B = DHT.T@self.B_h@DHT
        self.B_inv = scipy.linalg.inv(self.B)
        self.F_h = np.diag([
            (power_spectrum_f(q)/N_pix)
            for q in list(range(N_pix//2 + 1)) + list(range(N_pix//2 - 1, 0, -1))])
        self.F = DHT.T@self.F_h@DHT

   def get_evidence(
            self,
            direction=1,
            verbosity=0,
            return_terms=False,
            beta_init=None,
            method='Newton-CG',
            ):
        """
        computes the Hamiltonian of the evidence (d|X->Y), where d = (x,y)

        ftParameters:
        ----------
        x : np.array
            samples of the cause variable
        y: np.array
            samples of the effect variable
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

        cause_terms = self.get_cause_terms(k, 

        effect_terms = self.get_effect_terms(
                direction=direction, F=self.F, noise_var=self.noise_var)

        # compute the evidence (x,y | X->Y)
        evidence = (
                log_prod_k_fact +
                sum(cause_terms) +
                sum(effect_terms)
                )

        terms = [
                ('ln(prod(k_j!))', log_prod_k_fact),
                ('1/2 ln(det(curvature_gamma_beta))', cause_terms[0]),
                ('-k^dagger.beta_0', cause_terms[1]),
                ('rho(One.e^beta_0)', cause_terms[2]),
                ('1/2 beta_0.(B^-1).beta_0', cause_terms[3]),
                ('1/2 y.((F_tilde + N)^-1).y', effect_terms[0]),
                ('1/2 ln(det(F_tilde + N))', effect_terms[1]),
                ]

        if verbosity > 0:
            print(
                    'numerical values of terms: \n'
                    '   1/2 ln(|curvature_gamma_beta|):                  {:.2e}\n'
                    '   -k^dagger.beta_vec:           {:.2e}\n'
                    '   rho(One.e^beta_0_vec):        {:.2e}\n'
                    '   1/2 beta_0.B^-1.beta_0:       {:.2e}\n'
                    '   ln(prod(k_j!)):               {:.2e}\n'
                    '   1/2 y^T.((F_tilde + N)^-1).y: {:.2e}\n'
                    '   1/2 ln(|F_tilde + N|):        {:.2e}'.format(
                        cause_terms[0],
                        cause_terms[1], cause_terms[2], cause_terms[3],
                        log_prod_k_fact,
                        effect_terms[0], effect_terms[1],))


        if return_terms:
            return evidence, terms
        else:
            return evidence


def energy_eta(eta, F_tilde, x_indices,):
    # term1 (1/2 ln det (F_tilde + diag(exp(eta(x)))))
    noise_covariance = np.diag(np.exp(eta[x_indices]))
    sign, log_det = np.linalg.slogdet(F_tilde + noise_covariance)
    term1 = 0.5 * log_det

    # term2 

