import numpy as np

np.random.seed(1)

TCEP_FOLDER = (
    '/afs/mpa/home/maxk/bayesian_causal_inference/benchmarks/tcep/')
BENCHMARK_FOLDER = (
    '/afs/mpa/home/maxk/bayesian_causal_inference/benchmarks/tcep_30samples/')

for i in range(108):
    dataset = np.genfromtxt(
        (TCEP_FOLDER + 'pair0{:03d}.txt'.format(i+1)))
    subsample_indices = np.random.choice(
            np.arange(dataset.shape[0]), 50, replace=False)
    dataset = dataset[subsample_indices]
    np.savetxt(
               (BENCHMARK_FOLDER + 'pair0{:03d}.txt'.format(i+1)),
               dataset, delimiter=' ')

# manually copy pairmeta.txt
