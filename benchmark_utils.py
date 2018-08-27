import argparse
import os
import re

import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))


DEFAULT_BENCHMARK_LENGTH = {
    'CE-Cha': 300,
    'tcep': 108,
    'bcs_power6_nvar5e-2': 100,
    'bcs_power6_nvar1e-1': 100,
    'bcs_power6_nvar5e-1': 100,
    'bcs_power4_nvar1': 100,
    'bcs_default': 100,
    'bcs_nvar1e-1': 100,
    'bcs_nvar5e-1': 100,
    'bcs_8bins': 100,
    'bcs_16bins': 100,
    'bcs_10samples': 100,
    'bcs_30samples': 100,
    'SIM': 100,
    }

def get_benchmark_default_length(benchmark):
    return DEFAULT_BENCHMARK_LENGTH[benchmark]

def get_pair(i, benchmark):
    if benchmark == 'CE-Cha':
        with open(os.path.join(
                dir_path, './benchmark_data_pairwise/CE-Cha_pairs.tab')) as f:
            lines = f.readlines()
        ser_1 = [l.split('"')[3].split(' ')[1:] for l in lines[1:]]
        ser_2 = [l.split('"')[5].split(' ')[1:] for l in lines[1:]]
        pair = np.array(
                ser_1[i], dtype='float'), np.array(ser_2[i], dtype='float')

        with open('./benchmark_data_pairwise/CE-Cha_targets.tab') as f:
            lines = f.readlines()[1:]
        targets = [
                re.findall('1\.0|-1\.0', l)[0].replace('-1', '0')
                for l in lines]
        targets = np.array([int(float(t)) for t in targets])
        true_direction = targets[i]
        if true_direction == 0:
            true_direction = -1
        weight = 1

    elif benchmark in [ 
            'SIM',
            'tcep',
            'bcs_power6_nvar5e-2',
            'bcs_power6_nvar1e-1',
            'bcs_power6_nvar5e-1',
            'bcs_power4_nvar1',
            'bcs_default',
            'bcs_nvar1e-1',
            'bcs_nvar5e-1',
            'bcs_8bins',
            'bcs_16bins',
            'bcs_10samples',
            'bcs_30samples',
            ]:
        # these benchmark datasets use the formatting from Mooij16,
        #   in the folder with the name of the benchmark there is a number of
        #   txt-files, named pair0003.txt e.g., with ' '-delimited data
        #   pairmeta.txt columns 1,2,3,4 contain first column index of cause, 
        #       last column
        #   index of cause, first column index of effect, last column index of
        #   effect. We exclude datasets with more than 1 effect or cause
        #   variables
        dataset = np.genfromtxt(os.path.join(
            dir_path,
            'benchmarks/{}/pair0{:03d}.txt'.format(benchmark, i+1)))
        pair = (dataset[:, 0], dataset[:, 1])
        meta = np.genfromtxt(os.path.join(
            dir_path,
            './benchmarks/{}/pairmeta.txt'.format(benchmark)))
        if (
                meta[i, 1] == 1 and meta[i, 2] == 1
                and meta[i, 3] == 2 and meta[i, 4] == 2):
            true_direction = 1
        elif (
                meta[i, 1] == 2 and meta[i, 2] == 2
                and meta[i, 3] == 1 and meta[i, 4] == 1):
            true_direction = -1
        else:
            true_direction = 0

        if benchmark == 'tcep':
            weight = meta[i, 5]
        else:
            weight = 1

    return pair, true_direction, weight


class BCMParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(description='perform inference tests')

        self.add_argument(
                '--name', type=str, help='name for the benchmark results')
        self.add_argument(
                '--nbins', type=int, default=256,
                help='number of bins for inference model')
        self.add_argument(
                '--num_samples', type=int, default=256,
                help='number of samples to draw')
        self.add_argument(
                '--noise_var', type=float, default=0.1,
                help='value for noise_var')
        self.add_argument(
                '--config', type=int, default=None,
                help='id referring to a parameter configuration in '
                '"model_configurations.txt"')
        self.add_argument(
                '--first_id', type=int, default=1,
                help='first id to process, 1 <= id <= 300')
        self.add_argument(
                '--last_id', type=int, default=None,
                help='last id to process, 1 <= id <= 300')
        self.add_argument(
                '--power_spectrum_beta', type=str, default='1/(q**4 + 1)',
                help='assumed power spectrum beta')
        self.add_argument(
                '--power_spectrum_f', type=str, default='1/(q**4 + 1)',
                help='assumed power spectrum f')
        self.add_argument(
                '--rho', type=float, default=1., help='rho value')
        self.add_argument(
                '--iteration_limit', type=int, default=500,
                help='iteration limit for minimization')
        self.add_argument(
                '--tol_rel_gradnorm', type=float, default=1e-3,
                help='tol rel gradnorm parameter for minimization')
        self.add_argument(
                '--verbosity', type=int, default=0,
                help='verbosity of output')
        self.add_argument(
                '--benchmark', type=str, default='bcs_default',
                help='which benchmark case, see DEFAULT_BENCHMARK_LENGTH dict')
        self.add_argument(
                '--model', type=int, default=1,
                help='which model to use for inference')
        self.add_argument(
                '--scale_max', type=float, default=1,
                help='scale the data to the interval [0, scale_max]')
