import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

plt.rc("text", usetex=True)
plt.rc(
    "text.latex", preamble=r"\usepackage{amsmath} \usepackage{bm} \usepackage{dsfont}"
)

import bayesian_causal_model.bayesian_causal_model
import bayesian_causal_model.cause_model_shallow
import bayesian_causal_model.bayesian_causal_sampling
from benchmark_utils import get_pair

DRAW_SCATTER_PLOTS = False

color1 = sns.color_palette()[0]  # seaborn blue
color2 = sns.color_palette()[3]  # seaborn red
if DRAW_SCATTER_PLOTS:
    for benchmark in [
        "bcs_default",
        "bcs_nvar2e-1",
        "bcs_nvar1",
        "bcs_16bins",
        "bcs_8bins",
        "bcs_30samples",
        "bcs_10samples",
        "tcep",
        "tcep_20samples",
    ]:

        fig = plt.figure(figsize=(16, 16))
        if benchmark.startswith("bcs"):
            subplots = fig.subplots(10, 10)
            num_plots = 100
        elif benchmark.startswith("tcep"):
            subplots = fig.subplots(11, 10)
            num_plots = 108

        idx = -1
        for i in range(num_plots):
            (x, y), true_direction, w = get_pair(i, benchmark)
            if true_direction == -1:
                c = color2
            elif true_direction == +1:
                c = color1
            else:
                continue
            idx += 1
            ax = subplots[idx // 10, idx % 10]
            ax.scatter(x, y, s=2, c=c)
            ax.tick_params(
                axis="both",
                which="both",
                bottom=False,
                top=False,
                left=False,
                right=False,
                # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
        if benchmark.startswith("tcep"):
            for i in range(110 - idx - 1):
                idx += 1
                fig.delaxes(subplots[idx // 10, idx % 10])

        fig.savefig("paper_plots/dataset_{}.pdf".format(benchmark))

###############################################################################
# plot samples for fields drawn from different power spectra

bcs_power2 = bayesian_causal_model.bayesian_causal_sampling.BayesianCausalSampler(
    power_spectrum_beta=lambda q: 1e3 / ((q) ** 2 + 1),
    power_spectrum_f=lambda q: 1e3 / ((q) ** 2 + 1),
    noise_var=5e-2,
)
bcs_power4 = bayesian_causal_model.bayesian_causal_sampling.BayesianCausalSampler(
    power_spectrum_beta=lambda q: 1e3 / ((q) ** 4 + 1),
    power_spectrum_f=lambda q: 1e3 / ((q) ** 4 + 1),
    noise_var=5e-2,
)
bcs_power6 = bayesian_causal_model.bayesian_causal_sampling.BayesianCausalSampler(
    power_spectrum_beta=lambda q: 1e3 / ((q) ** 6 + 1),
    power_spectrum_f=lambda q: 1e3 / ((q) ** 6 + 1),
    noise_var=5e-2,
)

x = np.linspace(0, 1, bcs_power2.N_bins)

linestyles = [":", "-.", "--", "-"]

fig, subplots = plt.subplots(figsize=(14, 16), nrows=3, ncols=2)
# fig2, ax2 = plt.subplots(figsize=(6, 3))
ax1, ax2 = subplots[0]
for i in range(4):
    bcs_power2.draw_sample_fields()
    ax1.plot(x, bcs_power2.beta, linestyle=linestyles[i])
    ax2.plot(x, bcs_power2.p_x, linestyle=linestyles[i])
# fig1.savefig('paper_plots/probability_samples_power2.pdf')
# fig2.savefig('paper_plots/field_samples_power2.pdf')

# fig1, ax1 = plt.subplots(figsize=(6, 3))
# fig2, ax2 = plt.subplots(figsize=(6, 3))
ax1, ax2 = subplots[1]
for i in range(4):
    bcs_power4.draw_sample_fields()
    ax1.plot(x, bcs_power4.beta, linestyle=linestyles[i])
    ax2.plot(x, bcs_power4.p_x, linestyle=linestyles[i])
# fig1.savefig('paper_plots/probability_samples_power4.pdf')
# fig2.savefig('paper_plots/field_samples_power4.pdf')

# fig1, ax1 = plt.subplots(figsize=(6, 3))
# fig2, ax2 = plt.subplots(figsize=(6, 3))
ax1, ax2 = subplots[2]
for i in range(4):
    bcs_power6.draw_sample_fields()
    ax1.plot(x, bcs_power6.beta, linestyle=linestyles[i])
    ax2.plot(x, bcs_power6.p_x, linestyle=linestyles[i])
# fig1.savefig('paper_plots/probability_samples_power6.pdf')
# fig2.savefig('paper_plots/field_samples_power6.pdf')
# fig2.savefig('paper_plots/field_samples_power6.pdf')
fig.savefig("paper_plots/field_samples.pdf")

###############################################################################
# Perform a inference on some dataset and plot the histograms and beta_min
#   fields

figsize = (4, 4)
power_spectrum_beta = lambda q: 2048 / (q ** 4 + 1)
power_spectrum_f = lambda q: 2048 / (q ** 4 + 1)

scale = (0, 1)
scaler = MinMaxScaler(scale)
(x, y), true_direction, w = get_pair(31, "bcs_default")
x, y = scaler.fit_transform(np.array((x, y)).T).T

N_bins = 512
bcm = bayesian_causal_model.cause_model_shallow.CausalModelShallow(
    N_bins=N_bins,
    power_spectrum_beta=power_spectrum_beta,
    power_spectrum_f=power_spectrum_f,
    rho=1,
    noise_var=1e-2,
)

fig = plt.figure(figsize=figsize)
ax = fig.subplots(1, 1)
ax.scatter(x, y, c="black", s=2)
plt.rc("text", usetex=True)
plt.rc(
    "text.latex", preamble=r"\usepackage{amsmath} \usepackage{bm} \usepackage{dsfont}"
)
ax.set_xlabel("$X$")
ax.set_ylabel("$Y$")
fig.savefig("paper_plots/scatter_for_beta_mins.pdf")

bcm.set_data(x, y)
evidence_x_to_y, terms_x_to_y = bcm.get_evidence(direction=1, return_terms=1)
beta0_x_to_y = bcm.minimization_result.x

evidence_y_to_x, terms_y_to_x = bcm.get_evidence(direction=-1, return_terms=1)
beta0_y_to_x = bcm.minimization_result.x

fig = plt.figure(figsize=figsize)
ax = fig.subplots(1, 1)
ax_ = ax.twinx()
ax.ticklabel_format(axis="both", style="sci")
ax.plot(np.arange(N_bins), beta0_x_to_y, label=r"$\beta_{min}$", c="black")
ax.set_xlabel("$X$")
ax.set_ylabel(r"$\beta_0$")
ax_.bar(np.arange(N_bins), bcm.k_x, label=r"$k_x$", color="black")
ax_.set_xticks([0, (N_bins) // 2, N_bins - 1])
ax_.set_xticklabels([0, 0.5, 1])
ax_.set_ylabel(r"$\bm{k}_x$")
ax.legend(loc=0)
ax_.legend(loc=1)
plt.tight_layout()
fig.savefig("paper_plots/histogram_and_beta_min_x_to_y.pdf")

fig = plt.figure(figsize=figsize)
ax = fig.subplots(1, 1)
ax.plot(beta0_y_to_x, np.arange(N_bins, 0, -1), color="black", label=r"$\beta_0$")
ax_ = ax.twiny()
ax_.barh(np.arange(N_bins, 0, -1), bcm.k_y, label=r"$k_y$", color="black")
ax_.set_yticks([0, (N_bins) // 2, N_bins - 1])
ax_.set_yticklabels([0, 0.5, 1])
ax.set_ylabel("$Y$")
ax.set_xlabel(r"$\beta_0$")
ax_.set_xlabel(r"$\bm{k}_y$")
ax.legend(loc=4)
ax_.legend(loc=0)
plt.tight_layout()
fig.savefig("paper_plots/histogram_and_beta_min_y_to_x.pdf")

fig, ax = plt.subplots(figsize=figsize)
index = np.arange(len(terms_x_to_y) + 1)

term_values_x_to_y = [t[1] for t in terms_x_to_y]
term_values_y_to_x = [t[1] for t in terms_y_to_x]
bar_width = 0.35
rects1 = ax.bar(
    index,
    term_values_x_to_y + [evidence_x_to_y],
    bar_width,
    fill=False,
    hatch="///",
    label=r"$X \rightarrow Y$",
)
rects2 = ax.bar(
    index + bar_width,
    term_values_y_to_x + [evidence_y_to_x],
    bar_width,
    fill=False,
    hatch="///\\\\\\",
    label=r"$Y \rightarrow X$",
)
ax.set_xticks(index + bar_width / 2)
plt.rc("text", usetex=True)
plt.rc(
    "text.latex", preamble=r"\usepackage{amsmath} \usepackage{bm} \usepackage{dsfont}"
)
labels = [
    r"$\log(\prod_j k_j!) $",
    r"$\frac{1}{2} \log \left | \Gamma_\beta (\beta_0) \right| $",
    r"$ -\bm{k}^\dagger \beta_0 (\bm{x}) $",
    r"$ \bm{\rho}^\dagger e^{\beta_0(\bm{x})} $",
    r"$ \frac{1}{2}\beta_0^\dagger B^{-1} \beta_0 $",
    r"$ \frac{1}{2} \bm{y}^\dagger(\tilde{F} + \mathcal{E})^{-1}\bm{y}$",
    r"$ \frac{1}{2} \left| \tilde{F} + \mathcal{E}\right|$",
    r"total $\mathcal{H}(\bm{d}|\cdot)$",
]
ax.set_xticklabels(labels, size="small"),
ax.tick_params(axis="x", labelrotation=45)
ax.legend()
plt.tight_layout()
fig.savefig("paper_plots/demonstration_term_bars.pdf")

#############################################################################
