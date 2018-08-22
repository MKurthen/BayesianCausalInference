import argparse
import datetime

timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


class BCMParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(description='perform inference tests')

        self.add_argument(
                '--num', type=int, default=1,
                help='number of iterations to test')
        self.add_argument(
                '--seeded', type=bool, default=False,
                help='use fixed random seeds')
        self.add_argument(
                '--name', type=str, help='name for the benchmark results')
        self.add_argument(
                '--nbins', type=int, default=256,
                help='number of bins for inference model')
        self.add_argument(
                '--num_samples', type=int, default=256,
                help='number of samples to draw')
        self.add_argument(
                '--noise_var', type=float, default=0.1,
                help='value for noise_var')
        self.add_argument(
                '--config', type=int, default=None,
                help='id referring to a parameter configuration in '
                '"model_configurations.txt"')
        self.add_argument(
                '--first_id', type=int, default=1,
                help='first id to process, 1 <= id <= 300')
        self.add_argument(
                '--last_id', type=int, default=None,
                help='last id to process, 1 <= id <= 300')
        self.add_argument(
                '--power_spectrum_beta', type=str, default='1/(q**4 + 1)',
                help='assumed power spectrum beta')
        self.add_argument(
                '--power_spectrum_f', type=str, default='1/(q**4 + 1)',
                help='assumed power spectrum f')
        self.add_argument(
                '--rho', type=float, default=1., help='rho value')
        self.add_argument(
                '--iteration_limit', type=int, default=500,
                help='iteration limit for minimization')
        self.add_argument(
                '--tol_rel_gradnorm', type=float, default=1e-3,
                help='tol rel gradnorm parameter for minimization')
        self.add_argument(
                '--verbosity', type=int, default=0,
                help='verbosity of output')
        self.add_argument(
                '--benchmark', type=str, default='Cha',
                help='which benchmark case, "Cha", "Gauss", "tcep",'
                    '"bcs_nvar5e-2", "bcs_nvar1e-1", "bcs_nvar5e-1"')
        self.add_argument(
                '--model', type=int, default=1,
                help='which model to use for inference')
        self.add_argument(
                '--scale_max', type=float, default=0.8,
                help='scale the data to the interval [0, scale_max]')
