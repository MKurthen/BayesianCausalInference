import sys

import numpy as np
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
eng.addpath(
        CAUSALITY_ROOT + 'comparison_methods/LiNGAM/lingam-1.4.2/FastICA_23/')

# prediction_file = './benchmark_predictions/{}_{}.txt'.format(BENCHMARK, NAME)
# if os.path.isfile(prediction_file):
#     c = 0
#     while os.path.isfile(prediction_file):
#         c += 1
#         prediction_file = './benchmark_predictions/{}_{}_{}.txt'.format(
#                 BENCHMARK, NAME, c)

accuracy = 0
sum_of_weights = 0
weighted_correct = 0
undecided = 0

for i in range(FIRST_ID-1, LAST_ID):
    (x, y), true_direction, weight = get_pair(i, BENCHMARK)
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
        predicted_direction = 0
        undecided += 1

    weighted_correct += weight*(predicted_direction == true_direction)
    sum_of_weights += weight
    accuracy = weighted_correct / sum_of_weights

    print(
            'dataset {}, true direction: {}, predicted direction {}\n'
            'accuracy so far: {:.2f}'.format(
                i, true_direction, predicted_direction, accuracy))

print('accuracy: {:.2f}'.format(accuracy))
