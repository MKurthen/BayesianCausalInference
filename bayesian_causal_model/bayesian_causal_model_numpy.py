import numpy as np
import scipy
import scipy.linalg

from .utilities import get_count_vector, remove_duplicates

class BayesianCausalModel(object):
    """
    model to infer a bivariate causal direction from observations. i.e.
        given X and Y the goal is to infer wether X->Y or Y->X

    Parameters:
    ----------
    N_pix : int
        the number of grid points for the field discretization
    """
    def __init__(
            self,
            N_pix=1024,
            ):
        self.N_pix = N_pix
        self.grid_coordinates = np.arange(0, 1, 1/N_pix)
        self.grid = np.arange(self.N_pix)

    def get_effect_terms(self, direction, F, noise_var):
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

        self.F_tilde = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(i+1):
                self.F_tilde[i, j] = F[
                        cause_sample_indices[i], cause_sample_indices[j]]
                if i != j:
                    self.F_tilde[j, i] = F[
                            cause_sample_indices[i], cause_sample_indices[j]]

        # the noise covariance matrix N
        noise_covariance = np.diag(
                [noise_var for _ in range(self.N)])

        # get the inverse (G_x)_ij = (F(x_i, x_j) + N)^-1
        self.Gx = np.linalg.inv(self.F_tilde + noise_covariance)

        # 1/2 y^T.(F_tilde + N)^-1).y
        term1 = 0.5*effect_samples@self.Gx@effect_samples

        # we further need 1/2*ln(det(F_tilde + N))
        sign, ln_det = np.linalg.slogdet(self.F_tilde + noise_covariance)
        if sign <= 0:
            print('warning, computed sign {} in ln(det(F_tilde + N))'.format(
                sign))
        term2 = 0.5*ln_det

        return (term1, term2)

    def get_cause_terms(self, k, method='Newton-CG', beta_init=None):
        # initial guess for beta is proportional to k
        if beta_init is None:
            beta_init = np.log(k/self.rho + 1e-3)

        self.minimization_result = scipy.optimize.minimize(
                fun=energy_cause_shallow,
                args=(k, self.B_inv, self.rho),
                x0=beta_init,
                method=method,
                jac=gradient_cause_shallow,
                hess=curvature_cause_shallow)

        beta_0 = self.minimization_result.x

        curvature = curvature_cause_shallow(beta_0, k, self.B_inv, self.rho)

        # (1/2)*ln(det(D))
        sign, ln_det_curvature = np.linalg.slogdet(curvature)
        if sign <= 0:
            print('warning, computed sign {} in ln_det_DB'.format(sign))

        term1 = 0.5 * ln_det_curvature
        term2 = -k@beta_0
        term3 = self.rho*np.sum(np.exp(beta_0))
        term4 = 0.5 * beta_0@self.B_inv@beta_0
        cause_terms = (term1, term2, term3, term4)

        return cause_terms


    def set_data(self, x, y, no_duplicates=False):
        assert (len(x) == len(y))
        self.k_x, self.x_indices = get_count_vector(
                x, self.grid_coordinates, return_indices=True)
        self.x = self.grid_coordinates[self.x_indices]
        self.k_y, self.y_indices = get_count_vector(
                y, self.grid_coordinates, return_indices=True)
        self.y = self.grid_coordinates[self.y_indices]

        if no_duplicates:
            self.x, self.y = remove_duplicates(
                    self.x, self.y)
            self.k_x, self.x_indices = get_count_vector(
                self.x, self.grid_coordinates, return_indices=True)
            self.k_y, self.y_indices = get_count_vector(
                self.y, self.grid_coordinates, return_indices=True)
        self.N = len(self.x)

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


# Energy
def energy_cause_shallow(beta, k, B_inv, rho):
    term1 = -k@beta
    term2 = rho*np.sum(np.exp(beta))
    term3 = 0.5 * beta@B_inv@beta
    return term1 + term2 + term3

# Gradient / Jacobian
def gradient_cause_shallow(beta, k, B_inv, rho):
    term1 = -k
    term2 = rho*np.exp(beta)
    term3 = beta@B_inv
    return term1 + term2 + term3

# Curvature / Hessian
def curvature_cause_shallow(beta, k, B_inv, rho):
    term1 = rho*np.diag(np.exp(beta))
    term2 = B_inv
    return term1 + term2
