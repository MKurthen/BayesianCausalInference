import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

import bayesian_causal_model.bayesian_causal_model
import bayesian_causal_model.bayesian_causal_sampling
from benchmark_utils import get_pair

color1 = sns.color_palette()[0] # seaborn blue
color2 = sns.color_palette()[3] # seaborn red
for benchmark in [
        'bcs_default',
        'bcs_nvar2e-1',
        'bcs_nvar1',
        'bcs_16bins',
        'bcs_8bins',
        'bcs_30samples',
        'bcs_10samples'
        ]:

    fig = plt.figure(figsize=(16, 16))
    subplots = fig.subplots(10, 10)
    for i in range(100):
        ax = subplots[i//10, i % 10]
        (x, y), true_direction, w = get_pair(i, benchmark)
        if true_direction == -1:
            c = color2
        elif true_direction == +1:
            c = color1
        else:
            continue
        ax.scatter(x, y, s=2, c=c)
        ax.tick_params(
            axis='both',
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            # ticks along the top edge are off
            labelbottom=False,
            labelleft=False)
    fig.savefig('thesis_plots/dataset_{}.png'.format(benchmark))
