import numpy as np
import scipy.stats
from sklearn.preprocessing import MinMaxScaler


class BayesianCausalSampler(object):
    """
    class to sample a a bivariate distribution (X, Y) where Y is
    deterministically related to X via a f: X->Y
    """

    def __init__(
            self,
            N_bins=512,
            power_spectrum_beta=lambda q: 512/(q**4 + 1),
            power_spectrum_f=lambda q: 512/(q**4 + 1),
            noise_var=0.1,
            ):
        """
        N_bins : int
            number of bins for the sample spaces
        p_spec_beta : function
            power spectrum for beta
        p_spec_f : function
            power spectrum for f
        noise_var : scalar
            the variance of the noise variable
        """
        self.N_bins = N_bins

        # calculate Discrete Hartley Transform Matrix
        DFT = scipy.linalg.dft(N_bins, scale='sqrtn')
        DHT = np.real(DFT) + np.imag(DFT)

        fourier_modes = (
                list(range(N_bins//2 + 1)) + list(range(N_bins//2 - 1, 0, -1)))

        self.power_spectrum_beta = power_spectrum_beta
        self.B_h = np.diag([
            power_spectrum_beta(q)
            for q in fourier_modes])
        self.B = DHT.T@self.B_h@DHT
        self.B_inv = scipy.linalg.inv(self.B)

        self.F_h = np.diag([
            power_spectrum_f(q)
            for q in fourier_modes])
        self.F = DHT.T@self.F_h@DHT

        # set the noise variance
        self.noise_var = noise_var

    def draw_sample_fields(self, invertible_mechanism=False):
        """
        draws the actual sample fields beta and f

        Parameters:
        ----------
        invertible_mechanism : Bool, experimental approach to draw an
            invertible causal mechanism f, by first applying exp (to ensure
            positivity) and afterwards accumulating the values
        """
        self.beta = np.random.multivariate_normal(
                mean=np.zeros(self.N_bins),
                cov=self.B)

        # numerical values for p(x|beta) = exp(beta(x)) / sum_z exp(beta(z))
        self.p_x = np.exp(self.beta)
        self.p_x = (1/np.sum(self.p_x))*self.p_x

        self.f = np.random.multivariate_normal(
                mean=np.zeros(self.N_bins),
                cov=self.F)

        if invertible_mechanism:
            self.f = np.cumsum(np.exp(self.f), axis=0)

    def get_samples(self, sample_size, poisson=True, discretize=True):
        """
        return samples for X, Y with a sample size
        if poisson=True, the sample_size will only be approximately

        poisson : bool,
            use the Poissonian formulation of the thesis for the forward model

        discretize : bool,
            perform the discretization also for the y-variable (the sampled
            y-data will be rounded to neares grid points)
        """
        if poisson:
            # draw with poisson dist, where each lambda[i] propto exp(beta)
            k_sample = np.random.poisson(
                    lam=self.p_x*sample_size, size=self.N_bins)
            x_sample_indices = []
            for i in range(self.N_bins):
                x_sample_indices += [i]*k_sample[i]
            x_sample_indices = np.array(x_sample_indices)

        if not poisson:
            # Draw directly with p_x propto exp(beta)
            # we use the scipy.stats.rv_discrete class to represent the
            #    the sampled X distribution
            # x_k do not refer to the x values, but to the indices of the
            #   underlying pixelization
            x_k = np.arange(0, self.N_bins)
            p_x_dist = scipy.stats.rv_discrete(values=(x_k, self.p_x))
            # get the pixel indices for the x sample
            x_sample_indices = p_x_dist.rvs(size=sample_size)

        # go back to the values on the axis [0, 1]
        x_sample = x_sample_indices / self.N_bins

        # sample the noise
        n_sample = np.random.normal(
                loc=0,
                scale=np.sqrt(self.noise_var),
                size=len(x_sample))

        # sample y by applying the function f
        Scaler = MinMaxScaler(feature_range=(0, 1))
        f_scaled = Scaler.fit_transform(self.f.reshape(-1, 1)).reshape(-1)

        y_sample = f_scaled[x_sample_indices] + n_sample

        if discretize:
            y_sample = Scaler.fit_transform(
                    y_sample.reshape(-1, 1)).reshape(-1)
            grid_coordinates = np.linspace(0, 1, self.N_bins)
            y_sample = grid_coordinates[
                    ([np.abs(grid_coordinates - y_sample[i]).argmin()
                        for i in range(len(y_sample))])]

        return (x_sample, y_sample)
