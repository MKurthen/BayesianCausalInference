import numpy as np

from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.append('..')
import bayesian_causal_model.bayesian_causal_sampling_numpy

BENCHMARK_FOLDER = (
    '/afs/mpa/home/maxk/bayesian_causal_inference/benchmarks/bcs_30samples/')

power_spectrum_beta = lambda q: 512/(q**4 + 1)
power_spectrum_f = lambda q: 512/(q**4 + 1)

for i in range(100):
    np.random.seed(i)
    bcs = bayesian_causal_model.bayesian_causal_sampling_numpy.BayesianCausalSampler(
        N_bins=512,
        power_spectrum_beta=power_spectrum_beta,
        power_spectrum_f=power_spectrum_f,
        noise_var=5e-2)

    bcs.draw_sample_fields()

    x, y = bcs.get_samples(30)
    scaler = MinMaxScaler(feature_range=(0, 1))
    x, y = scaler.fit_transform(np.array((x, y)).T).T
    # flip x and y with probability 1/2
    flip = bool(np.random.binomial(1, 0.5))
    if flip:
        np.savetxt(
                (BENCHMARK_FOLDER +
                'pair0{:03d}.txt'.format(i+1)),
                np.array([y, x]).T, delimiter=' ')

        with open(BENCHMARK_FOLDER + 'pairmeta.txt', 'a') as f:
            f.write('0{:03d} 2 2 1 1 1\n'.format(i+1))
    else:
        np.savetxt(
                (BENCHMARK_FOLDER +
                'pair0{:03d}.txt'.format(i+1)),
                np.array([x, y]).T, delimiter=' ')
        with open(BENCHMARK_FOLDER + 'pairmeta.txt', 'a') as f:
            f.write('0{:03d} 1 1 2 2 1\n'.format(i+1))
