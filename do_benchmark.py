import logging
logging.basicConfig(level=logging.ERROR)
import os
import re

import colorama
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import nifty5

import bayesian_causal_model.cause_model_shallow_nifty
import bayesian_causal_model.cause_model_shallow_numpy
#from bayesian_causal_model.cause_model_deep import CausalModelCauseDeep
#from bayesian_causal_model.effect_model_shallow import CausalModelEffectShallow
#from bayesian_causal_model.bayesian_causal_sampling import BayesianCausalSampler
from parser import BCMParser
from benchmark_utils import get_benchmark_default_length, get_pair
logging.basicConfig(level=logging.ERROR)


parser = BCMParser()
args = parser.parse_args()
NAME = args.name
FIRST_ID = args.first_id
LAST_ID = args.last_id

MODEL = args.model
N_BINS = args.nbins
NOISE_VAR = args.noise_var
ITERATION_LIMIT = args.iteration_limit
TOL_REL_GRADNORM = args.tol_rel_gradnorm
BENCHMARK = args.benchmark
VERBOSITY = args.verbosity

POWER_SPECTRUM_BETA_STR = args.power_spectrum_beta
POWER_SPECTRUM_F_STR = args.power_spectrum_f
RHO = args.rho
SCALE_MAX = args.scale_max

CONFIG = args.config
if CONFIG is not None:
    with open('./model_configurations.txt') as f:
        configs = eval(f.read())
    parameters = configs[CONFIG]
    MODEL = parameters.get('model', MODEL)
    N_BINS = parameters.get('nbins', N_BINS)
    NOISE_VAR = parameters.get('noise_var', NOISE_VAR)
    POWER_SPECTRUM_BETA_STR = parameters.get(
            'power_spectrum_beta', POWER_SPECTRUM_BETA_STR)
    POWER_SPECTRUM_F_STR = parameters.get(
            'power_spectrum_f', POWER_SPECTRUM_F_STR)
    ITERATION_LIMIT = parameters.get('iteration_limit', ITERATION_LIMIT)
    TOL_REL_GRADNORM = parameters.get('tol_rel_gradnorm', TOL_REL_GRADNORM)

if LAST_ID is None:
    LAST_ID = get_benchmark_default_length(BENCHMARK)

print(
        'performing {} benchmark for ids {} to {},\n'
        'with N_bins: {},\n'
        'noise variance: {}\n'
        'power spectrum beta: {}\n'
        'power spectrum f: {}\n'
        'rho: {}\n'
        'storing results with suffix {}'.format(
            BENCHMARK, FIRST_ID, LAST_ID, N_BINS,
            NOISE_VAR,
            POWER_SPECTRUM_BETA_STR,
            POWER_SPECTRUM_F_STR,
            RHO,
            NAME))

POWER_SPECTRUM_BETA = lambda q: eval(POWER_SPECTRUM_BETA_STR)
POWER_SPECTRUM_F = lambda q: eval(POWER_SPECTRUM_F_STR)
scale = (0, SCALE_MAX)

prediction_file = './benchmark_predictions/{}_{}.txt'.format(BENCHMARK, NAME)
if os.path.isfile(prediction_file):
    c = 0
    while os.path.isfile(prediction_file):
        c += 1
        prediction_file = './benchmark_predictions/{}_{}_{}.txt'.format(
                BENCHMARK, NAME, c)

accuracy = 0
sum_of_weights = 0
weighted_correct = 0
tp, tn, fp, fn = 0, 0, 0, 0

for i in range(FIRST_ID-1, LAST_ID):
    if BENCHMARK == 'tcep':
        (x, y), true_direction, weight = get_pair(i, BENCHMARK, return_weight=True)
    else:
        (x, y), true_direction = get_pair(i, BENCHMARK)
        weight = 1
    if true_direction == -1:
        continue

    scaler = MinMaxScaler(scale)
    x, y = scaler.fit_transform(np.array((x, y)).T).T

    minimizer = nifty5.RelaxedNewton(controller=nifty5.GradientNormController(
            tol_rel_gradnorm=TOL_REL_GRADNORM,
            iteration_limit=ITERATION_LIMIT,
            convergence_level=5,
            ))

    if MODEL == 1:
        bcm = bayesian_causal_model.cause_model_shallow_nifty.CausalModelShallow(
            N_pix=N_BINS,
            noise_var=NOISE_VAR,
            rho=RHO,
            power_spectrum_beta=POWER_SPECTRUM_BETA,
            power_spectrum_f=POWER_SPECTRUM_F,
            minimizer=minimizer,
            )
    elif MODEL == 2: 
        bcm = bayesian_causal_model.cause_model_shallow_numpy.CausalModelShallow(
            N_pix=N_BINS,
            noise_var=NOISE_VAR,
            rho=RHO,
            power_spectrum_beta=POWER_SPECTRUM_BETA,
            power_spectrum_f=POWER_SPECTRUM_F,
            )

    elif MODEL == 3:
        bcm = CausalModelCauseDeep(
            noise_var=NOISE_VAR,
            N_pix=N_BINS,
            rho=RHO,
            power_spectrum_f=POWER_SPECTRUM_F,
            minimizer=minimizer,
            )
    elif MODEL == 4:
        bcm = CausalModelEffectShallow(
            N_pix=N_BINS,
            sigma_f=1e4,
            power_spectrum_beta=POWER_SPECTRUM_BETA,
            minimizer=minimizer)

    bcm.set_data(x, y)

    H1 = bcm.get_evidence(direction=1, verbosity=1)
    H2 = bcm.get_evidence(direction=-1, verbosity=1)
    predicted_direction = int(H1 < H2)
    if predicted_direction == 1 and true_direction == 1:
        tp += 1
    if predicted_direction == 1 and true_direction == 0:
        fp += 1
    if predicted_direction == 0 and true_direction == 1:
        fn += 1
    if predicted_direction == 0 and true_direction == 0:
        tn += 1


    if predicted_direction == true_direction:
        fore = colorama.Fore.GREEN
        weighted_correct += weight
    else:
        fore = colorama.Fore.RED
    sum_of_weights += weight
    accuracy = weighted_correct / sum_of_weights 

    print('dataset {}, {} true direction: {}, predicted direction {}\n'
            'H1: {:.2e},\n H2: {:.2e},\n{}'
            'accuracy so far: {:.2f}'.format(
        i, fore, true_direction, predicted_direction, H1, H2, colorama.Style.RESET_ALL, accuracy))

    with open(prediction_file, 'a') as f:
        f.write('{} {} {} {}\n'.format(i+1, predicted_direction, H1, H2))

#accuracy = (tp+tn)/(tp+tn+fp+fn)
print('accuracy: {:.2f}'.format(accuracy))


benchmark_information = {
        'model': MODEL,
        'n_bins': N_BINS,
        'noise_var': NOISE_VAR,
        'rho': RHO,
        'power_spectrum_beta': POWER_SPECTRUM_BETA_STR,
        'power_spectrum_f': POWER_SPECTRUM_F_STR,
        'iteration_limit': ITERATION_LIMIT,
        'tol_rel_gradnorm': TOL_REL_GRADNORM,
        'minimizer': 'VL_BFGS',
        'accuracy': accuracy,
        'prediction_file': prediction_file,
        }

with open('cha_benchmarks.txt', 'a') as f:
    f.write(str(benchmark_information) + '\n')
