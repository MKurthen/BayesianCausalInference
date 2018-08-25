import os
import re
import sys

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matlab.engine

CAUSALITY_ROOT = '/afs/mpa/home/maxk/bayesian_causal_inference/'
sys.path.append(CAUSALITY_ROOT)
from benchmark_utils import get_benchmark_default_length, get_pair, BCMParser

parser = BCMParser()
args = parser.parse_args()
NAME = args.name
BENCHMARK = args.benchmark
FIRST_ID = args.first_id
LAST_ID = args.last_id

if LAST_ID is None:
    LAST_ID = get_benchmark_default_length(BENCHMARK)

eng = matlab.engine.start_matlab()
eng.addpath(CAUSALITY_ROOT + 'comparison_methods/Mooij16/cep')
eng.startup(nargout=0)
eng.local_config(nargout=0)

methodpars = eng.struct()
methodpars['FITC'] = 0
methodpars['minimize'] = 'minimize_lbfgsb'
methodpars['evaluation'] = 'HSIC'

accuracy = 0
correct_decisions = 0
undecided = 0

for i in range(FIRST_ID-1, LAST_ID):
    (x, y), true_direction = get_pair(i, BENCHMARK)
    if true_direction == 0:
        continue

    x_ = matlab.double(x.tolist())
    x_T = eng.transpose(x_)
    y_ = matlab.double(y.tolist())
    y_T = eng.transpose(y_)

    result = eng.cep_anm(x_T, y_T, methodpars)
    predicted_direction = result['decision'] 

    if predicted_direction == true_direction:
        correct_decisions += 1
    if predicted_direction == 0:
        undecided += 1
    
    accuracy = correct_decisions / (i + 1)

    print('dataset {}, true direction: {}, predicted direction {}\n'
            'accuracy so far: {:.2f}'.format(
        i, true_direction, predicted_direction, accuracy))


print('accuracy: {:.2f}, undecided: {}'.format(accuracy, undecided))

