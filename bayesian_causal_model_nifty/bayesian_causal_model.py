import numpy as np

import nifty5.operators.chain_operator
import scipy.stats


from .utilities import probe_operator, get_count_vector


class BayesianCausalModel(object):
    """
    model to infer a bivariate causal direction from observations. i.e.
        given X and Y the goal is to infer wether X->Y or Y->X

    Parameters:
    ----------
    power_spectrum_beta: function
        the power spectrum of the beta field distribution covariance
    power_spectrum_f: function
        the power spectrum of the covrariance of the distribution for the
        field representing the relating function
    N_pix : int
        the number of grid points for the field discretization
    noise_var : scalar
        the variance for the sampling of noise variables
    rho : scalar
        density of the response
    minimization : string
        currently supported: 'nifty5.vl_bfgs', 'nifty5.relaxed_newton',
        'nifty5.steepest_descent', 'scipy'
    """
    def __init__(
            self,
            N_pix=1024,
            ):
        self.N_pix = N_pix
        self.s_space = nifty5.RGSpace(
                [N_pix], distances=1/N_pix)

        self.fft = nifty5.FFTOperator(self.s_space)
        self.h_space = self.s_space.get_default_codomain()
        self.p_space = nifty5.PowerSpace(harmonic_partner=self.h_space)
        self.grid_coordinates = [i*self.s_space.distances[0] for i in range(
            self.N_pix)]
        # as measurement grid we take the underlying grid of the nifty
        #   signal space
        self.grid = np.arange(self.N_pix)

    def get_effect_terms(self, direction=1):
        """
        convenience function to collect terms for the evidence hamiltonian
            which belong to the effect, i.e. H(y|x, F, N, X->Y)
        """
        if direction == 1:
            cause_sample_indices = [
                np.abs(self.grid_coordinates - self.x[i]).argmin()
                for i in range(self.N)]
            effect_samples = self.y
        else:
            cause_sample_indices = [
                np.abs(self.grid_coordinates - self.y[i]).argmin()
                for i in range(self.N)]
            effect_samples = self.x

        F_h = nifty5.create_power_operator(
                domain=self.h_space, power_spectrum=self.power_spectrum_f)
        # get the F(x_i, x_j) matrix
        F = nifty5.operators.chain_operator.ChainOperator.make(
                (self.fft.adjoint, F_h, self.fft))

        self.F_matrix = probe_operator(F)
        F_tilde = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(i+1):
                F_tilde[i, j] = self.F_matrix[
                        cause_sample_indices[i], cause_sample_indices[j]]
                if i != j:
                    F_tilde[j, i] = self.F_matrix[
                            cause_sample_indices[i], cause_sample_indices[j]]

        # the noise covariance matrix N
        noise_covariance = np.diag(
                [self.noise_var for _ in range(self.N)])

        # get the inverse (G_x)_ij = (F(x_i, x_j) + N)^-1
        self.Gx = np.linalg.inv(F_tilde + noise_covariance)

        # 1/2 y^T.(F_tilde + N)^-1).y
        term1 = 0.5*effect_samples@self.Gx@effect_samples

        # we further need 1/2*ln(det(x_F_x + N))
        sign, ln_det_F_tilde = np.linalg.slogdet(F_tilde + noise_covariance)
        if sign <= 0:
            print('warning, computed sign {} in ln(det(x_F_x + N))'.format(
                sign))
        term2 = 0.5*ln_det_F_tilde
        return (term1, term2)

    def set_data(self, x, y):
        assert (len(x) == len(y))
        self.x = x
        self.y = y
        self.k_x, self.x_indices = get_count_vector(
                x, self.grid_coordinates, return_indices=True)
        self.k_y, self.y_indices = get_count_vector(
                y, self.grid_coordinates, return_indices=True)
        self.N = len(x)

        # ln(prod(k_j!)) = sum(ln(k_j!))
        # the gammaln functions is numerically stable for large numbers
        self.log_prod_k_x_fact = np.sum(
            scipy.special.gammaln(self.k_x[self.k_x > 1]))
        self.log_prod_k_y_fact = np.sum(
            scipy.special.gammaln(self.k_y[self.k_y > 1]))

    def get_causal_direction(
            self,
            x,
            y,
            verbosity=0,
            ):
        """
        calculate the evidence Hamiltonian H(d|X->Y) and H(d|Y->X), omitting
            parts not dependent on (x,y). Infer the causal direction by
            comparing and taking the smaller Hamiltonian

        Parameters:
        -----------
        x : np.array
            samples of the cause variable
        y : np.array
            samples of the effect variable

        Returns:
        direction: int
            1 for X->Y, -1 for Y->X
        """
        hamiltonian1 = self.get_evidence(x, y, verbosity=verbosity-1)
        if verbosity > 0:
            print('H(d|X->Y) - H_0 = {:.2e}'.format(hamiltonian1))
        hamiltonian2 = self.get_evidence(y, x, verbosity=verbosity-1)
        if verbosity > 0:
            print('H(d|Y->X) - H_0 = {:.2e}'.format(hamiltonian2))
        # smaller Hamiltonian means higher evidence
        if hamiltonian1 < hamiltonian2:
            direction = 1
        else:
            direction = -1
        return direction
