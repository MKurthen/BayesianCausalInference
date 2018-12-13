import numpy as np

# script to subsample the TCEP benchmark set with 30 samples per dataset. This
#   is repeated 20 times, so a somewhat representative average can be 
#   calculated


TCEP_FOLDER = (
    '/afs/mpa/home/maxk/bayesian_causal_inference/benchmarks/tcep/')
BENCHMARK_FOLDER = (
    '/afs/mpa/home/maxk/bayesian_causal_inference/benchmarks/tcep_30samples_x20/')

meta = np.genfromtxt(TCEP_FOLDER + 'pairmeta.txt')

for x in range(20):
    np.random.seed(x)
    for i in range(108):
        dataset = np.genfromtxt(
            (TCEP_FOLDER + 'pair0{:03d}.txt'.format(i+1)))
        subsample_indices = np.random.choice(
                np.arange(dataset.shape[0]), 30, replace=False)
        dataset = dataset[subsample_indices]
        np.savetxt(
                   (BENCHMARK_FOLDER + 'pair0{:03d}.txt'.format((x*108)+i+1)),
                   dataset, delimiter=' ')

        with open(BENCHMARK_FOLDER + 'pairmeta.txt', 'a') as f:
            f.write('0{:03d} {} {} {} {} {}\n'.format(
                (x*108)+i+1,
                int(meta[i, 1]),
                int(meta[i, 2]),
                int(meta[i, 3]),
                int(meta[i, 4]),
                meta[i, 5]))
