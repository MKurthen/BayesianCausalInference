import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

CAUSALITY_ROOT = '/afs/mpa/home/maxk/bayesian_causal_inference/'
sys.path.append(CAUSALITY_ROOT)
sys.path.append(CAUSALITY_ROOT + 'comparison_methods/CGNN/Code')

import bayesian_causal_model.cause_model_shallow
from benchmark_utils import get_benchmark_default_length, get_pair, BCMParser
import cgnn
import matlab.engine

parser = BCMParser()
args = parser.parse_args()
BENCHMARK = args.benchmark
FIRST_ID = args.first_id
LAST_ID = args.last_id

if LAST_ID is None:
    LAST_ID = get_benchmark_default_length(BENCHMARK)

eng = matlab.engine.start_matlab('-nodesktop -noFigureWindows -nodisplay')
eng.addpath(CAUSALITY_ROOT + 'comparison_methods/Mooij16/cep')
# add some global variables to workspace
eng.startup(nargout=0)
eng.local_config(nargout=0)
# disable all output
eng.system('dir 1>NUL 2>NUL')

methodpars_anm_hsic = eng.struct()
methodpars_anm_hsic['FITC'] = 0
methodpars_anm_hsic['minimize'] = 'minimize_lbfgsb'
methodpars_anm_hsic['evaluation'] = 'HSIC'

# CGNN Settings
cgnn.SETTINGS.GPU = False
cgnn.SETTINGS.NB_GPU = 1
cgnn.SETTINGS.NB_JOBS = 1
cgnn.SETTINGS.h_layer_dim = 30
cgnn.SETTINGS.use_Fast_MMD = True
cgnn.SETTINGS.NB_RUNS = 8

sum_of_weights = 0
accuracy_bcm = 0
accuracy_anm_hsic = 0
accuracy_igci = 0
accuracy_cgnn = 0
weighted_correct_bcm = 0
weighted_correct_anm_hsic = 0
weighted_correct_igci = 0
weighted_correct_cgnn = 0

for i in range(FIRST_ID-1, LAST_ID):
    (x, y), true_direction, weight = get_pair(
                i, BENCHMARK)
    if true_direction == 0:
        continue

    scaler = MinMaxScaler(feature_range=(0, 1))
    x_scaled, y_scaled = scaler.fit_transform(np.array((x, y)).T).T

    # test bcm
    bcm = bayesian_causal_model.cause_model_shallow.CausalModelShallow(
        N_pix=512,
        noise_var=1e-2,
        rho=1,
        power_spectrum_beta=lambda q: 2048/(q**4 + 1),
        power_spectrum_f=lambda q: 2048/(q**4 + 1),
        )

    bcm.set_data(x_scaled, y_scaled)
    H1 = bcm.get_evidence(direction=1,)
    H2 = bcm.get_evidence(direction=-1,)
    predicted_direction_bcm = 1 if int(H1 < H2) else -1

    # test ANM-HSIC

    x_ = matlab.double(x.tolist())
    x_T = eng.transpose(x_)
    y_ = matlab.double(y.tolist())
    y_T = eng.transpose(y_)

    result = eng.cep_anm(x_T, y_T, methodpars_anm_hsic)
    predicted_direction_anm_hsic = result['decision']

    if predicted_direction_bcm == true_direction:
        weighted_correct_bcm += weight

    if predicted_direction_anm_hsic == true_direction:
        weighted_correct_anm_hsic += weight

    # test LiNGAM
    # test ANM-MML
    # test IGCI
    # test CGNN
    df = pd.DataFrame(
            index=[0], columns=['SampleID', 'A', 'B'], dtype='object')
    df.SampleID.iloc[0] = 'pair1'
    df.A.iloc[0] = x
    df.B.iloc[0] = y
    model = cgnn.GNN(backend="TensorFlow")
    predictions = model.predict_dataset(df)
    predictions = pd.DataFrame(predictions, columns=["Predictions"])
    if predictions.iloc[0, 0] > 0:
        predicted_direction_cgnn = 1
    else:
        predicted_direction_cgnn = -1

    if predicted_direction_cgnn == true_direction:
        weighted_correct_cgnn += weight

    sum_of_weights += weight
    accuracy_bcm = weighted_correct_bcm / sum_of_weights
    accuracy_anm_hsic = weighted_correct_anm_hsic / sum_of_weights
    accuracy_cgnn = weighted_correct_cgnn / sum_of_weights

print(
        'accuracy: \n'
        '   BCM      {:.2f}\n'
        '   ANM-HSIC {:.2f}\n'
        # '   ANM-MML {:.2f}\n'
        # '   IGCI    {:.2f}\n'
        '   CGNN     {:.2f}\n'
        ''.format(
            accuracy_bcm,
            accuracy_anm_hsic,
            #    anm_mml_accuracy,
            #    igci_accuracy,
            accuracy_cgnn
            ))
