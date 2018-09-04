import os

import colorama
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import nifty5

import bayesian_causal_model.cause_model_shallow
import bayesian_causal_model_nifty.cause_model_shallow
from benchmark_utils import BCMParser, get_benchmark_default_length, get_pair

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
SUBSAMPLE = args.subsample

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

for i in range(FIRST_ID-1, LAST_ID):
    (x, y), true_direction, weight = get_pair(
                i, BENCHMARK, subsample_size=SUBSAMPLE)
    if true_direction == 0:
        continue

    scaler = MinMaxScaler(scale)
    x, y = scaler.fit_transform(np.array((x, y)).T).T

    minimizer = nifty5.RelaxedNewton(controller=nifty5.GradientNormController(
            tol_rel_gradnorm=TOL_REL_GRADNORM,
            iteration_limit=ITERATION_LIMIT,
            convergence_level=5,
            ))

    if MODEL == 1:
        bcm = bayesian_causal_model.cause_model_shallow.CausalModelShallow(
            N_bins=N_BINS,
            noise_var=NOISE_VAR,
            rho=RHO,
            power_spectrum_beta=POWER_SPECTRUM_BETA,
            power_spectrum_f=POWER_SPECTRUM_F,
            )
    elif MODEL == 2:
        bcm = bayesian_causal_model_nifty.cause_model_shallow.CausalModelShallow(
            N_bins=N_BINS,
            noise_var=NOISE_VAR,
            rho=RHO,
            power_spectrum_beta=POWER_SPECTRUM_BETA,
            power_spectrum_f=POWER_SPECTRUM_F,
            minimizer=minimizer,
            )

    bcm.set_data(x, y)

    H1 = bcm.get_evidence(direction=1, verbosity=1)
    H2 = bcm.get_evidence(direction=-1, verbosity=1)
    predicted_direction = 1 if int(H1 < H2) else 0

    if predicted_direction == true_direction:
        fore = colorama.Fore.GREEN
        weighted_correct += weight
    else:
        fore = colorama.Fore.RED
    sum_of_weights += weight
    accuracy = weighted_correct / sum_of_weights

    print(
            'dataset {}, {} true direction: {}, predicted direction {}\n'
            'H1: {:.2e},\n H2: {:.2e},\n{}'
            'accuracy so far: {:.2f}'.format(
                i,
                fore,
                true_direction,
                predicted_direction,
                H1,
                H2,
                colorama.Style.RESET_ALL,
                accuracy))

    with open(prediction_file, 'a') as f:
        f.write('{} {} {} {}\n'.format(i+1, predicted_direction, H1, H2))

print('accuracy: {:.2f}'.format(accuracy))


benchmark_information = {
        'benchmark': BENCHMARK,
        'model': MODEL,
        'n_bins': N_BINS,
        'noise_var': NOISE_VAR,
        'rho': RHO,
        'power_spectrum_beta': POWER_SPECTRUM_BETA_STR,
        'power_spectrum_f': POWER_SPECTRUM_F_STR,
        'accuracy': accuracy,
        'prediction_file': prediction_file,
        }

with open('benchmark_predictions/benchmarks_meta.txt', 'a') as f:
    f.write(str(benchmark_information) + '\n')
