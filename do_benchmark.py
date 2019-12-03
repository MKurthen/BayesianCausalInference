import os

import colorama
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# import nifty5

import bayesian_causal_model.cause_model_shallow

# import bayesian_causal_model_nifty.cause_model_shallow
from benchmark_utils import BCMParser, get_benchmark_default_length, get_pair


def get_args(parser):
    args = parser.parse_args()

    args.power_spectrum_beta_str = args.power_spectrum_beta
    args.power_spectrum_f_str = args.power_spectrum_f
    if args.config is not None:
        with open("./model_configurations.txt") as f:
            configs = eval(f.read())
        cparameters = configs[config]
        args.n_bins = parameters.get("n_bins", args.n_bins)
        args.noise_variance = parameters.get("noise_variance", args.noise_variance)
        args.power_spectrum_beta_str = parameters.get(
            "power_spectrum_beta", power_spectrum_beta_str
        )
        args.power_spectrum_f_str = parameters.get(
            "power_spectrum_f", power_spectrum_f_str
        )
    if args.last_id is None:
        args.last_id = get_benchmark_default_length(args.benchmark)

    return args


def main():
    parser = BCMParser()
    args = get_args(parser)
    verbosity = args.verbosity
    benchmark = args.benchmark

    print(
        f"performing {args.benchmark} benchmark for ids {args.first_id} to {args.last_id},\n"
        f"with N_bins: {args.n_bins},\n"
        f"noise variance: {args.noise_variance}\n"
        f"power spectrum beta: {args.power_spectrum_beta_str}\n"
        f"power spectrum f: {args.power_spectrum_f_str}\n"
        f"rho: {args.rho}\n"
        f"scale_max: {args.scale_max}\n"
        f"storing results with suffix {args.name}"
    )

    power_spectrum_beta = lambda q: eval(args.power_spectrum_beta_str)
    power_spectrum_f = lambda q: eval(args.power_spectrum_f_str)
    scale = (0, args.scale_max)

    prediction_file = "./benchmark_predictions/{}_{}.txt".format(benchmark, args.name)
    if os.path.isfile(prediction_file):
        c = 0
        while os.path.isfile(prediction_file):
            c += 1
            prediction_file = "./benchmark_predictions/{}_{}_{}.txt".format(
                benchmark, args.name, c
            )

    sum_of_weights = 0
    weighted_correct = 0

    for rep in range(args.repetitions):
        np.random.seed(rep)

        for i in range(args.first_id - 1, args.last_id):
            (x, y), true_direction, weight = get_pair(
                i, benchmark, subsample_size=args.subsample
            )
            if true_direction == 0:
                continue

            scaler = MinMaxScaler(scale)
            x, y = scaler.fit_transform(np.array((x, y)).T).T

            bcm = bayesian_causal_model.cause_model_shallow.CausalModelShallow(
                N_bins=args.n_bins,
                noise_var=args.noise_variance,
                rho=args.rho,
                power_spectrum_beta=power_spectrum_beta,
                power_spectrum_f=power_spectrum_f,
            )

            bcm.set_data(x, y)

            H1 = bcm.get_evidence(direction=1, verbosity=verbosity - 1)
            H2 = bcm.get_evidence(direction=-1, verbosity=verbosity - 1)
            predicted_direction = 1 if int(H1 < H2) else -1

            if predicted_direction == true_direction:
                fore = colorama.Fore.GREEN
                weighted_correct += weight
            else:
                fore = colorama.Fore.RED
            sum_of_weights += weight
            accuracy = weighted_correct / sum_of_weights

            if verbosity > 0:
                print(
                    "dataset {}, {} true direction: {}, predicted direction {}\n"
                    "H1: {:.2e},\n H2: {:.2e},\n{}"
                    "accuracy so far: {:.2f}".format(
                        i,
                        fore,
                        true_direction,
                        predicted_direction,
                        H1,
                        H2,
                        colorama.Style.RESET_ALL,
                        accuracy,
                    )
                )

            with open(prediction_file, "a") as f:
                f.write("{} {} {} {}\n".format(i + 1, predicted_direction, H1, H2))

    print("accuracy: {:.2f}".format(accuracy))

    benchmark_information = {
        "benchmark": benchmark,
        "n_bins": args.n_bins,
        "noise_var": args.noise_var,
        "rho": args.rho,
        "power_spectrum_beta": args.power_spectrum_beta_str,
        "power_spectrum_f": args.power_spectrum_f_str,
        "accuracy": accuracy,
        "prediction_file": prediction_file,
    }

    with open("benchmark_predictions/benchmarks_meta.txt", "a") as f:
        f.write(str(benchmark_information) + "\n")


if __name__ == "__main__":
    main()
