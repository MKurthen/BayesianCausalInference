import numpy as np
import scipy.linalg
import scipy.optimize

from .cause_model_shallow import CausalModelShallow
from .utilities import get_diff_matrix


class CausalModelNoiseInference(CausalModelShallow):
    def __init__(
            self,
            N_bins=512,
            power_spectrum_f=lambda k: 1/(k**4+1),
            power_spectrum_beta=lambda k: 1/(k**4+1),
            sigma_eta=1,
            rho=1,
            ):
        super().__init__(
                N_bins=N_bins,
                power_spectrum_f=power_spectrum_f,
                power_spectrum_beta=power_spectrum_beta,
                rho=rho)

        self.power_spectrum_f = power_spectrum_f
        self.power_spectrum_beta = power_spectrum_beta
        self.sigma_eta = sigma_eta
        self.rho = rho

    def get_cause_terms(
            self):
            diff_matrix = get_diff_matrix(self.N_bins)
            args = (
                    self.x_indices,
                    self.F_tilde,
                    self.y,
                    self.sigma_eta,
                    diff_matrix)

            minimization_result_eta = scipy.optimize.minimize(
                fun=energy_eta,
                x0=np.log(0.05)*np.ones(self.N_bins),
                args=args,
                method='Newton-CG',
                jac=gradient_eta,
                hess=curvature_eta,
                )
            eta_0 = minimization_result_eta.x
            curvature_eta_0 = curvature_eta(eta_0, *args)
            sign, log_det_curvature_eta_0 = np.linalg.slogdet(
                    (1/(2*np.pi))*curvature_eta_0)
            term1 = 0.5*log_det_curvature_eta_0
            term2, term3, term4 = energy_eta(eta_0, *args, return_terms=True)

            return (term1, term2, term3, term4)

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

        Parameters:
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
            log_prod_k_fact = self.log_prod_k_x_fact
        elif direction == -1:
            log_prod_k_fact = self.log_prod_k_y_fact
        else:
            raise Exception('invalid direction')

        cause_terms = self.get_cause_terms()

        effect_terms = self.get_effect_terms(
                direction=direction, F=self.F, noise_var=self.noise_var)

        # compute the evidence (x,y | X->Y)
        evidence = (
                log_prod_k_fact +
                sum(cause_terms) +
                sum(effect_terms)
                )

        terms = [
                ('log(prod(k_j!))', log_prod_k_fact),
                ('1/2 log(det(curvature_gamma_beta))', cause_terms[0]),
                ('-k^dagger.beta_0', cause_terms[1]),
                ('rho(One.e^beta_0)', cause_terms[2]),
                ('1/2 beta_0.(B^-1).beta_0', cause_terms[3]),
                ('1/2 log(|curvature_gamma_eta_0/ 2pi|)', effect_terms[0]),
                ('1/2 log(|F_tilde + hat(exp(eta_0(x)))|)', effect_terms[1]),
                ('1/2 y.[(F_tilde + hat(exp(eta_0(x))))^-1].y',
                    effect_terms[2]),
                ('(1/2sigma_eta)* eta_0.nabla.nabla.eta_0', effect_terms[3]),
                ]

        if return_terms:
            return evidence, terms
        else:
            return evidence


def energy_eta(
        eta,
        x_indices,
        F_tilde,
        y,
        sigma_eta,
        diff_matrix,
        return_terms=False):
    """
    implementation of gamma_eta from the paper, i.e. the energy for the Laplace
        approximation of the noise variance field eta
    """
    noise_covariance = np.diag(np.exp(eta[x_indices]))
    sign, log_det = np.linalg.slogdet((F_tilde + noise_covariance))
    # 0.5 * log(| F_tilde + hat(exp(eta(x)))|)
    term1 = 0.5 * log_det
    G = np.linalg.inv(F_tilde + noise_covariance)
    # 0.5 * y.[(F_tilde + hat(exp(eta(x))))^-1].y
    term2 = 0.5*y@G@y
    # (1/ 2sigma_eta)* eta.nabla.nabla.eta
    term3 = (1/(2*sigma_eta))*np.gradient(eta)@np.gradient(eta)
    if return_terms:
        return (term1, term2, term3)
    else:
        return term1 + term2 + term3


def gradient_eta(eta, x_indices, F_tilde, y, sigma_eta, diff_matrix):
    exp_eta_x = np.exp(eta[x_indices])
    noise_covariance = np.diag(exp_eta_x)
    G = np.linalg.inv(F_tilde + noise_covariance)

    # tr(G.diag(exp(eta(x))delta_x))
    term1 = 0.5*np.array([
        np.trace(G@np.diag(np.where(
            np.equal(x_indices, i), exp_eta_x, np.zeros(len(y)))))
        for i in range(len(eta))])
    term2 = -0.5*np.array([y@G@np.diag(np.where(
            np.equal(x_indices, i), exp_eta_x, np.zeros(len(y)))
                                      )@G@y
                           for i in range(len(eta))])
    term3 = (1/sigma_eta)*diff_matrix@eta
    return term1 + term2 + term3


def curvature_eta(eta, x_indices, F_tilde, y, sigma_eta, diff_matrix):
    exp_eta_x = np.exp(eta[x_indices])
    del_exp_eta_list = [np.diag(np.where(
            np.equal(x_indices, i), exp_eta_x, np.zeros(len(y))))
                        for i in range(len(eta))]
    noise_covariance = np.diag(exp_eta_x)
    G = np.linalg.inv(F_tilde + noise_covariance)

    # tr(G.diag(exp(eta(x))delta_x))
    term1 = np.array([[np.trace(-G@del_exp_eta_list[i]@G@del_exp_eta_list[j] +
                      (G@del_exp_eta_list[i] if i == j else 0))
                       for i in range(len(eta))]
                      for j in range(len(eta))])
    term2 = np.array([[
                y@(2*G@del_exp_eta_list[i]@G@del_exp_eta_list[j]@G
                    - (G@del_exp_eta_list[i]@G if i == j else 0))@y
                for i in range(len(eta))]
                for j in range(len(eta))])
    term3 = (1/sigma_eta)*diff_matrix
    return term1 + term2 + term3
