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
    'bcs_nvar5e-2': 100,
    'bcs_nvar1e-1': 100,
    'bcs_nvar5e-1': 100,
    'bcs_8bins': 100,
    'bcs_16bins': 100,
    'SIM': 100,
    }

def get_benchmark_default_length(benchmark):
    return DEFAULT_BENCHMARK_LENGTH[benchmark]

def get_pair(i, benchmark, return_weight=False):
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

    elif benchmark in [ 
            'SIM',
            'tcep',
            'bcs_power6_nvar5e-2',
            'bcs_power6_nvar1e-1',
            'bcs_power6_nvar5e-1',
            'bcs_power4_nvar1',
            'bcs_nvar5e-2',
            'bcs_nvar1e-1',
            'bcs_nvar5e-1',
            'bcs_8bins',
            'bcs_16bins',
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
            true_direction = 0
        else:
            true_direction = -1

        if benchmark == 'tcep':
            weight = meta[i, 5]

            if return_weight:
                return pair, true_direction, weight
    return pair, true_direction
