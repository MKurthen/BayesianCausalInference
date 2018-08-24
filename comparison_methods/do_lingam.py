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
eng.addpath(CAUSALITY_ROOT + 'comparison_methods/LiNGAM/lingam-1.4.2/code/')
eng.addpath(CAUSALITY_ROOT + 'comparison_methods/LiNGAM/lingam-1.4.2/FastICA_23/')

# prediction_file = './benchmark_predictions/{}_{}.txt'.format(BENCHMARK, NAME)
# if os.path.isfile(prediction_file):
#     c = 0
#     while os.path.isfile(prediction_file):
#         c += 1
#         prediction_file = './benchmark_predictions/{}_{}_{}.txt'.format(
#                 BENCHMARK, NAME, c)

accuracy = 0
tp, tn, fp, fn = 0, 0, 0, 0

for i in range(FIRST_ID-1, LAST_ID):
    (x, y), true_direction = get_pair(i, BENCHMARK)
    if true_direction == 0:
        continue

    x_ = matlab.double(y.tolist())
    y_ = matlab.double(x.tolist())
    d = eng.vertcat(x_, y_)

    
    [B, stde, ci, k] = eng.estimate(d, nargout=4)
    k = np.array(k).reshape(-1)
    print(k)
    if k[0] < k[1]:
        predicted_direction = 1
    elif k[0] > k[1]:
        predicted_direction = -1
    else:
        raise Exception('no causal ordering returned')


    if predicted_direction == 1 and true_direction == 1:
        tp += 1
    if predicted_direction == 1 and true_direction == 0:
        fp += 1
    if predicted_direction == 0 and true_direction == 1:
        fn += 1
    if predicted_direction == 0 and true_direction == 0:
        tn += 1

    accuracy = (tp+tn)/(tp+tn+fp+fn)

    print('dataset {}, true direction: {}, predicted direction {}\n'
            'accuracy so far: {:.2f}'.format(
        i, true_direction, predicted_direction, accuracy))

#    with open(prediction_file, 'a') as f:
#        f.write('{} {} {} {}\n'.format(i+1, predicted_direction, H1, H2))

accuracy = (tp+tn)/(tp+tn+fp+fn)
print('accuracy: {:.2f}'.format(accuracy))

