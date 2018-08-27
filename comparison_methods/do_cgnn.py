import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

CAUSALITY_ROOT = '/afs/mpa/home/maxk/bayesian_causal_inference/'
sys.path.append(CAUSALITY_ROOT)
sys.path.append(CAUSALITY_ROOT + 'comparison_methods/CGNN/Code')
from benchmark_utils import get_benchmark_default_length, get_pair, BCMParser
import cgnn
parser = BCMParser()
args = parser.parse_args()
NAME = args.name
BENCHMARK = args.benchmark
FIRST_ID = args.first_id
LAST_ID = args.last_id

if LAST_ID is None:
    LAST_ID = get_benchmark_default_length(BENCHMARK)

# Params
cgnn.SETTINGS.GPU = False
cgnn.SETTINGS.NB_GPU = 1
cgnn.SETTINGS.NB_JOBS = 1
cgnn.SETTINGS.h_layer_dim = 30

#Setting for CGNN-MMD
cgnn.SETTINGS.use_Fast_MMD = True
cgnn.SETTINGS.NB_RUNS =8



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

for i in range(FIRST_ID-1, LAST_ID):
    (x, y), true_direction, weight = get_pair(i, BENCHMARK)
    if true_direction == 0:
        continue

    df = pd.DataFrame(index=[0], columns=['SampleID', 'A', 'B'], dtype='object')
    df.SampleID.iloc[0] = 'pair1'
    df.A.iloc[0] = x
    df.B.iloc[0] = y
    model = cgnn.GNN(backend="TensorFlow")
    predictions = model.predict_dataset(df, printout='_printout.csv')
    predictions = pd.DataFrame(predictions, columns=["Predictions"])
    if predictions.iloc[0, 0] > 0:
        predicted_direction = 1
    else:
        predicted_direction = -1
    if predicted_direction == true_direction:
        weighted_correct += weight
    sum_of_weights += weight
    accuracy = weighted_correct / sum_of_weights 


    print('dataset {}, true direction: {}, predicted direction {}\n'
            'accuracy so far: {:.2f}'.format(
        i, true_direction, predicted_direction, accuracy))


print('accuracy: {:.2f}'.format(accuracy))
