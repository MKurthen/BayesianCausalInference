import io
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
VERBOSITY = args.verbosity

if LAST_ID is None:
    LAST_ID = get_benchmark_default_length(BENCHMARK)

out = io.StringIO()
err = io.StringIO()

eng = matlab.engine.start_matlab('-nodesktop -noFigureWindows -nodisplay')
# setup for ANM and IGCI
eng.addpath(CAUSALITY_ROOT + 'comparison_methods/Mooij16/cep')
# add some global variables to workspace
eng.startup(nargout=0)
eng.local_config(nargout=0)

# setup for LiNGAM
eng.addpath(CAUSALITY_ROOT + 'comparison_methods/LiNGAM/lingam-1.4.2/code/')
eng.addpath(
        CAUSALITY_ROOT + 'comparison_methods/LiNGAM/lingam-1.4.2/FastICA_23/')

# parameters for ANM-HSIC
methodpars_anm_hsic = eng.struct()
methodpars_anm_hsic['FITC'] = 0
methodpars_anm_hsic['gaussianize'] = 0
methodpars_anm_hsic['meanf'] = 'meanConst'
methodpars_anm_hsic['minimize'] = 'minimize_lbfgsb'
methodpars_anm_hsic['evaluation'] = 'HSIC'

# parameters for ANM-MML
methodpars_anm_mml = eng.struct()
methodpars_anm_mml['FITC'] = 0
methodpars_anm_mml['gaussianize'] = 0
methodpars_anm_mml['meanf'] = 'meanConst'
methodpars_anm_mml['minimize'] = 'minimize_lbfgsb'
methodpars_anm_mml['evaluation'] = 'MML'

# parameters for IGCI
methodpars_igci = eng.struct()
methodpars_igci['ref_measure'] = 2
methodpars_igci['estimator'] = 'org_entropy'
methodpars_igci['entest'] = ''

# CGNN Settings
cgnn.SETTINGS.GPU = False
cgnn.SETTINGS.NB_GPU = 1
cgnn.SETTINGS.NB_JOBS = 1
cgnn.SETTINGS.h_layer_dim = 30
cgnn.SETTINGS.use_Fast_MMD = True
cgnn.SETTINGS.NB_RUNS = 8

sum_of_weights = 0
weighted_correct_bcm = 0
weighted_correct_lingam = 0
weighted_correct_anm_hsic = 0
weighted_correct_anm_mml = 0
weighted_correct_igci = 0
weighted_correct_cgnn = 0

save_stdout = sys.stdout
if VERBOSITY == 0:
    # silence output
    sys.stdout = open('trash_stdout', 'w')
    sys.stderr = open('trash_stderr', 'w')

for i in range(FIRST_ID-1, LAST_ID):

    (x, y), true_direction, weight = get_pair(
                i, BENCHMARK)
    if true_direction == 0:
        continue

    scaler = MinMaxScaler(feature_range=(0, 1))
    x_scaled, y_scaled = scaler.fit_transform(np.array((x, y)).T).T

    # test bcm
    bcm = bayesian_causal_model.cause_model_shallow.CausalModelShallow(
        N_bins=512,
        noise_var=1e-2,
        rho=1,
        power_spectrum_beta=lambda q: 2048/(q**4 + 1),
        power_spectrum_f=lambda q: 2048/(q**4 + 1),
        )

    bcm.set_data(x_scaled, y_scaled)
    H1 = bcm.get_evidence(direction=1,)
    H2 = bcm.get_evidence(direction=-1,)
    predicted_direction_bcm = 1 if int(H1 < H2) else -1
    weighted_correct_bcm += weight*(predicted_direction_bcm == true_direction)

    # prepare matlab arrays for LiNGAM, ANM, IGCI
    x_ = matlab.double(x.tolist())
    x_T = eng.transpose(x_)
    y_ = matlab.double(y.tolist())
    y_T = eng.transpose(y_)

    # test LiNGAM
    d = eng.vertcat(x_, y_)
    [B, stde, ci, k] = eng.estimate(d, nargout=4, stdout=out, stderr=err)
    k = np.array(k).reshape(-1)
    if k[0] < k[1]:
        predicted_direction_lingam = 1
    elif k[0] > k[1]:
        predicted_direction_lingam = -1
    else:
        predicted_direction_lingam = 0
    weighted_correct_lingam += weight*(
                predicted_direction_lingam == true_direction)

    # test ANM-HSIC
    result = eng.cep_anm(
            x_T, y_T, methodpars_anm_hsic,
            stdout=io.StringIO(), stderr=io.StringIO())
    predicted_direction_anm_hsic = result['decision']
    weighted_correct_anm_hsic += weight*(
                predicted_direction_anm_hsic == true_direction)

    # test ANM-MML

    # the mml code assigns a default parameter "maxclusters = 50" if not
    #   set otherwise. This leads to an error if the number of samples is < 50
    if len(x) < 50:
        methodpars_anm_mml['MML'] = eng.struct()
        methodpars_anm_mml['MML']['maxclusters'] = matlab.double([len(x)])
    result = eng.cep_anm(
            x_T, y_T, methodpars_anm_mml,
            stdout=io.StringIO(), stderr=io.StringIO())
    predicted_direction_anm_mml = result['decision']
    weighted_correct_anm_mml += weight*(
                predicted_direction_anm_mml == true_direction)

    # test IGCI
    result = eng.cep_igci(x_T, y_T, methodpars_igci, stdout=out, stderr=err)
    predicted_direction_igci = result['decision']
    weighted_correct_igci += weight*(
                predicted_direction_igci == true_direction)
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

    weighted_correct_cgnn += weight*(
            predicted_direction_cgnn == true_direction)

    sum_of_weights += weight
    accuracy_bcm = weighted_correct_bcm / sum_of_weights
    accuracy_lingam = weighted_correct_lingam / sum_of_weights
    accuracy_anm_hsic = weighted_correct_anm_hsic / sum_of_weights
    accuracy_anm_mml = weighted_correct_anm_mml / sum_of_weights
    accuracy_igci = weighted_correct_igci / sum_of_weights
    accuracy_cgnn = weighted_correct_cgnn / sum_of_weights

    # reactivate output
    sys.stdout = save_stdout
    print(
            'pair {}, accuracy so far: \n'
            '    BCM      {:.2f}\n'
            '    LiNGAM   {:.2f}\n'
            '    ANM-HSIC {:.2f}\n'
            '    ANM-MML  {:.2f}\n'
            '    IGCI     {:.2f}\n'
            '    CGNN     {:.2f}\n'
            ''.format(
                i+1,
                accuracy_bcm,
                accuracy_lingam,
                accuracy_anm_hsic,
                accuracy_anm_mml,
                accuracy_igci,
                accuracy_cgnn
                ))
    # re-silence output
    if VERBOSITY == 0:
        sys.stdout = open('trash_stdout', 'w')

# reactivate output
sys.stdout = save_stdout
print(
        'accuracy: \n'
        '    BCM      {:.2f}\n'
        '    LiNGAM   {:.2f}\n'
        '    ANM-HSIC {:.2f}\n'
        '    ANM-MML  {:.2f}\n'
        '    IGCI     {:.2f}\n'
        '    CGNN     {:.2f}\n'
        ''.format(
            accuracy_bcm,
            accuracy_lingam,
            accuracy_anm_hsic,
            accuracy_anm_mml,
            accuracy_igci,
            accuracy_cgnn
            ))
