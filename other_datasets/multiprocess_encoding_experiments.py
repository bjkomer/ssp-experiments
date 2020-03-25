from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import argparse

from constants import full_continuous, small_continuous, full_continuous_regression, small_continuous_regression
from feature_encoding import encode_dataset, encode_dataset_nd, encode_comparison_dataset

from pmlb import fetch_data, classification_dataset_names

from concurrent.futures import ThreadPoolExecutor as Pool
import subprocess
import time

parser = argparse.ArgumentParser('Encoding experiment on PMLB')

parser.add_argument('--debug', action='store_true', help='if set, just try on a few datasets and variants')
parser.add_argument(
    '--encoding-type', type=str, default='all',
    choices=['independent-ssp', 'combined-ssp', 'combined-simplex-ssp', 'all', 'pc-gauss', 'pc-gauss-tiled', 'one-hot', 'tile-code'],
    help='type of ssp encoding to use'
)
parser.add_argument('--sigma', type=float, default=0.5)
parser.add_argument('--n-tiles', type=int, default=8)
parser.add_argument('--max-iters', type=int, default=600)
parser.add_argument('--limit-low', type=float, default=-3)
parser.add_argument('--limit-high', type=float, default=3)
parser.add_argument('--max-workers', type=int, default=10)
parser.add_argument('--folder', type=str, default='process_output')
parser.add_argument('--regression', action='store_true', help='Use the regression datasets instead of classification')
parser.add_argument('--only-encoding', action='store_true', help='only run the encodings and not the base normalized')
args = parser.parse_args()

params = vars(args)

if args.regression:
    MLP = MLPRegressor
else:
    MLP = MLPClassifier


@ignore_warnings(category=ConvergenceWarning)
def experiment(dataset, exp_args):
    X, y = fetch_data(dataset, return_X_y=True, local_cache_dir='cache')

    # making features 0 mean and unit variance
    scaler = StandardScaler()

    train_X, test_X, train_y, test_y = train_test_split(X, y)

    scaler.fit(train_X)

    train_X_scaled = scaler.transform(train_X)
    test_X_scaled = scaler.transform(test_X)

    # optionally scale the targets, from a single test, overall performance is worse with scaled targets
    # if exp_args.regression:
    #     scaler_reg = StandardScaler()
    #
    #     scaler_reg.fit(train_y.reshape(-1, 1))
    #     train_y = scaler_reg.transform(train_y.reshape(-1, 1)).reshape(-1)
    #     test_y = scaler_reg.transform(test_y.reshape(-1, 1)).reshape(-1)

    # don't use adam on the smaller datasets
    if len(train_X) > 1000:
        solver = 'adam'
    else:
        solver = 'lbfgs'

    # max_iters = [300]
    # hidden_layers = [(512,), (1024,), (256, 256)]
    # seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # scales = [0.25, 1.0]
    # dims = [256]

    max_iters = [args.max_iters]
    scales = [0.25]
    dims = [256]

    if exp_args.encoding_type == 'all':
        enc_types = ['independent-ssp', 'combined-ssp', 'combined-simplex-ssp', 'one-hot', 'tile-code', 'pc-gauss', 'pc-gauss-tiled']
    elif exp_args.encoding_type == 'all-ssp':
        enc_types = ['independent-ssp', 'combined-ssp', 'combined-simplex-ssp']
    elif exp_args.encoding_type == 'all-other':
        enc_types = ['one-hot', 'tile-code', 'pc-gauss', 'pc-gauss-tiled']
    else:
        enc_types = [exp_args.encoding_type]

    if exp_args.debug:
        seeds = [1, 2, 3]
        hidden_layers = [(512,)]
        inter_fname = '{}/debug_enc_{}_results_{}iters_{}.csv'.format(
            exp_args.folder, exp_args.encoding_type, exp_args.max_iters, dataset
        )
    else:
        seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        # hidden_layers = [(512, 512), (1024,)]
        hidden_layers = [(256,), (512,), (1024,), (256, 256), (512, 512), (1024, 1024)]
        inter_fname = '{}/enc_{}_results_{}iters_{}.csv'.format(
            exp_args.folder, exp_args.encoding_type, exp_args.max_iters, dataset
        )

    # only run if the data does not already exist
    if not os.path.exists(inter_fname):
        # contains all results for this dataset
        df = pd.DataFrame()
        for max_iter in max_iters:
            for hidden_layer_sizes in hidden_layers:
                for seed in seeds:
                    for scale in scales:
                        for dim in dims:
                            for enc_type in enc_types:

                                # train_X_enc = encode_dataset(train_X, dim=dim, seed=seed, scale=scale)
                                # test_X_enc = encode_dataset(test_X, dim=dim, seed=seed, scale=scale)

                                if enc_type == 'independent-ssp':
                                    train_X_enc_scaled = encode_dataset(
                                        train_X_scaled, dim=dim, seed=seed, scale=scale
                                    )
                                    test_X_enc_scaled = encode_dataset(
                                        test_X_scaled, dim=dim, seed=seed, scale=scale
                                    )
                                    encoding_name = 'SSP Normalized'
                                elif enc_type == 'combined-ssp':
                                    train_X_enc_scaled = encode_dataset_nd(
                                        train_X_scaled, dim=dim, seed=seed, scale=scale, style='normal'
                                    )
                                    test_X_enc_scaled = encode_dataset_nd(
                                        test_X_scaled, dim=dim, seed=seed, scale=scale, style='normal'
                                    )
                                    encoding_name = 'Combined SSP Normalized'
                                elif enc_type == 'combined-simplex-ssp':
                                    train_X_enc_scaled = encode_dataset_nd(
                                        train_X_scaled, dim=dim, seed=seed, scale=scale, style='simplex'
                                    )
                                    test_X_enc_scaled = encode_dataset_nd(
                                        test_X_scaled, dim=dim, seed=seed, scale=scale, style='simplex'
                                    )
                                    encoding_name = 'Combined Simplex SSP Normalized'
                                elif enc_type in ['one-hot', 'tile-code', 'pc-gauss', 'pc-gauss-tiled']:
                                    train_X_enc_scaled = encode_comparison_dataset(
                                        train_X_scaled, encoding=enc_type, seed=seed, dim=dim, **params
                                    )
                                    test_X_enc_scaled = encode_comparison_dataset(
                                        test_X_scaled, encoding=enc_type, seed=seed, dim=dim, **params
                                    )
                                    if enc_type == 'one-hot':
                                        encoding_name = 'One Hot'
                                    elif enc_type == 'tile-code':
                                        encoding_name = 'Tile Coding'
                                    elif enc_type == 'pc-gauss':
                                        encoding_name = 'RBF'
                                    elif enc_type == 'pc-gauss-tiled':
                                        encoding_name = 'RBF Tiled'
                                else:
                                    raise NotImplementedError('unknown encoding type: {}'.format(enc_type))

                                mlp = MLP(
                                    hidden_layer_sizes=hidden_layer_sizes,
                                    activation='relu',
                                    solver=solver,
                                    max_iter=max_iter,
                                    random_state=seed,
                                    early_stopping=True,
                                    validation_fraction=0.1,
                                )

                                mlp.fit(train_X_enc_scaled, train_y)
                                acc = mlp.score(test_X_enc_scaled, test_y)

                                df = df.append(
                                    {
                                        'Dim': dim,
                                        'Seed': seed,
                                        'Scale': scale if 'ssp' in enc_type else 0,
                                        'N-Tiles': exp_args.n_tiles if enc_type == 'tile-coding' else 0,
                                        'Sigma': exp_args.sigma if ((enc_type == 'pc-guass') or (enc_type == 'pc-guass-tiled')) else 0,
                                        'Encoding': encoding_name,
                                        'Dataset': dataset,
                                        'Model': 'MLP - {}'.format(hidden_layer_sizes),
                                        'Accuracy': acc,
                                        'Solver': solver,
                                        'Max Iter': max_iter,
                                    },
                                    ignore_index=True,
                                )

                    if not args.only_encoding:

                        mlp = MLP(
                            hidden_layer_sizes=hidden_layer_sizes,
                            activation='relu',
                            solver=solver,
                            max_iter=max_iter,
                            random_state=seed,
                            early_stopping=True,
                            validation_fraction=0.1,
                        )

                        mlp.fit(train_X_scaled, train_y)
                        acc = mlp.score(test_X_scaled, test_y)

                        df = df.append(
                            {
                                'Dim': 0,
                                'Seed': seed,
                                'Scale': 0,
                                'N-Tiles': 0,
                                'Sigma': 0,
                                'Encoding': 'Normalized',
                                'Dataset': dataset,
                                'Model': 'MLP - {}'.format(hidden_layer_sizes),
                                'Accuracy': acc,
                                'Solver': solver,
                                'Max Iter': max_iter,
                            },
                            ignore_index=True,
                        )
        # save each dataset individually, in case the run crashes and needs to be restarted
        df.to_csv(inter_fname)

    return dataset


class ExpRunner:

    def __init__(self, args):
        # current number of processes running
        self.n_workers = 0

        # index of the next dataset to run
        self.dataset_index = 0

        self.exp_args = args

        if self.exp_args.regression:
            if self.exp_args.debug:
                # self.datasets = ['1595_poker', '537_houses', '215_2dplanes', '1096_FacultySalaries']
                self.datasets = ['1096_FacultySalaries', '192_vineyard', '690_visualizing_galaxy', '665_sleuth_case2002', '485_analcatdata_vehicle']
            else:
                # just using the ones with small number of features for regression, since there is a lot anyway
                self.datasets = small_continuous_regression
        else:
            if self.exp_args.debug:
                self.datasets = ['banana', 'appendicitis', 'diabetes', 'titanic']
            else:
                self.datasets = full_continuous

        self.n_datasets = len(self.datasets)

        if not os.path.exists(self.exp_args.folder):
            os.makedirs(self.exp_args.folder)

        self.pool = Pool(max_workers=self.exp_args.max_workers)

    def callback(self, e):
        self.n_workers -= 1
        if e.exception() is not None:
            print("got exception: %s" % e.exception())
        else:
            print("Finished process for: {}".format(e.result()))

    def run(self):
        # loop until all datasets are launched and all workers are finished
        while (self.dataset_index < self.n_datasets) or (self.n_workers > 0):
            # if there is room for another worker and still work to be done, launch it
            if (self.n_workers < self.exp_args.max_workers) and (self.dataset_index < self.n_datasets):
                print("Launching process for: {} of {} - {}".format(self.dataset_index + 1, self.n_datasets, self.datasets[self.dataset_index]))
                f = self.pool.submit(experiment, dataset=self.datasets[self.dataset_index], exp_args=self.exp_args)
                f.add_done_callback(self.callback)
                self.n_workers += 1
                self.dataset_index += 1
            else:
                time.sleep(.1)


def main():

    exp_runner = ExpRunner(args)
    exp_runner.run()


if __name__ == '__main__':
    main()
