import numpy as np
import scipy.stats
from sklearn.preprocessing import MinMaxScaler

import nifty5


class BayesianCausalSampler(object):
    """
    class to sample a a bivariate distribution (X, Y) where Y is
    deterministically related to X via a f: X->Y
    """

    def __init__(
            self,
            N_bins=1024,
            power_spectrum_beta=lambda q: 1/(q**4 + 1),
            power_spectrum_f=lambda q: 1/(q**4 + 1),
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
        self.s_space = nifty5.RGSpace([N_bins], )
        self.h_space = self.s_space.get_default_codomain()
        self.p_space = nifty5.PowerSpace(self.h_space)

        # covariance operator for the beta distribution
        self.power_spectrum_beta = power_spectrum_beta
        B_h = nifty5.create_power_operator(
                self.h_space,
                power_spectrum=self.power_spectrum_beta)

        fft = nifty5.FFTOperator(self.s_space)
        self.B = nifty5.SandwichOperator.make(fft, B_h)
        self.beta = self.B.draw_sample()

        # numerical values for p(x|beta) = exp(beta(x)) / sum_z exp(beta(z))
        self.p_x_val = np.exp(np.array(self.beta.to_global_data()))
        self.p_x_val = (1/np.sum(self.p_x_val))*self.p_x_val

        # get the covariance operator for the f distribution
        self.power_spectrum_f = power_spectrum_f
        F_h = nifty5.create_power_operator(
                self.h_space,
                power_spectrum=self.power_spectrum_f)
        self.F = nifty5.SandwichOperator.make(fft, F_h)

        # sample the transformation function f
        self.f = self.F.draw_sample()
        self.f_val = np.array(self.f.to_global_data())

        # set the noise variance
        self.noise_var = noise_var

    def get_samples(self, sample_size, poisson=True):
        """
        return samples for X, Y with a sample size
        if poisson=True, the sample_size will only be approximately
        """
        if poisson:
            # draw with poisson dist, where each lambda[i] propto exp(beta)
            k_sample = np.random.poisson(
                    lam=self.p_x_val*sample_size, size=self.N_bins)
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
            p_x_dist = scipy.stats.rv_discrete(values=(x_k, self.p_x_val))
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
        f_scaled = Scaler.fit_transform(self.f_val.reshape(-1, 1)).reshape(-1)

        y_sample = f_scaled[x_sample_indices] + n_sample

        return (x_sample, y_sample)
