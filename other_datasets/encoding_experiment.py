from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import argparse

from constants import some_continuous, full_continuous, small_continuous
from feature_encoding import encode_dataset, encode_dataset_nd

from pmlb import fetch_data, classification_dataset_names

parser = argparse.ArgumentParser('Encoding experiment on PMLB')

parser.add_argument('--debug', action='store_true', help='if set, just try on a few datasets and variants')
parser.add_argument(
    '--encoding-type', type=str, default='independent-ssp',
    choices=['independent-ssp', 'combined-ssp', 'combined-simplex-ssp', 'all'],
    help='type of ssp encoding to use'
)
parser.add_argument('--max-iters', type=int, default=500)

args = parser.parse_args()

@ignore_warnings(category=ConvergenceWarning)
def main():

    if args.debug:
        datasets = ['banana', 'appendicitis', 'diabetes', 'titanic']
    else:
        datasets = full_continuous

    n_datasets = len(datasets)

    # for saving partial results in case of crash
    if not os.path.exists('intermediate'):
        os.makedirs('intermediate')

    # contains all results
    df_all = pd.DataFrame()

    for i, classification_dataset in enumerate(datasets):

        print('\x1b[2K\r {} of {}. {}'.format(i + 1, n_datasets, classification_dataset), end="\r")
        X, y = fetch_data(classification_dataset, return_X_y=True)

        # making features 0 mean and unit variance
        scaler = StandardScaler()

        train_X, test_X, train_y, test_y = train_test_split(X, y)

        scaler.fit(train_X)

        train_X_scaled = scaler.transform(train_X)
        test_X_scaled = scaler.transform(test_X)

        # don't use adam on the smaller datasets
        if len(train_X) > 1000:
            solver = 'adam'
        else:
            solver = 'lbfgs'

        #max_iters = [300]
        #hidden_layers = [(512,), (1024,), (256, 256)]
        #seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        #scales = [0.25, 1.0]
        #dims = [256]

        max_iters = [args.max_iters]
        scales = [0.25]
        dims = [256]

        if args.encoding_type == 'all':
            enc_types = ['independent-ssp', 'combined-ssp', 'combined-simplex-ssp']
        else:
            enc_types = [args.encoding_type]


        if args.debug:
            seeds = [1, 2, 3]
            hidden_layers = [(512,)]
            inter_fname = 'intermediate/debug_enc_{}_results_{}iters_{}.csv'.format(
                args.encoding_type, args.max_iters, classification_dataset
            )
        else:
            seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            hidden_layers = [(512, 512), (1024,)]
            inter_fname = 'intermediate/enc_{}_results_{}iters_{}.csv'.format(
                args.encoding_type, args.max_iters, classification_dataset
            )

        # only run if the data does not already exist
        if os.path.exists(inter_fname):
            df = pd.read_csv(inter_fname)
        else:
            # contains all results for this dataset
            df = pd.DataFrame()
            for max_iter in max_iters:
                for hidden_layer_sizes in hidden_layers:
                    for seed in seeds:
                        print(
                            '\x1b[2K\r {} of {}. {} - hs {} - seed {}'.format(
                                i + 1,
                                n_datasets,
                                classification_dataset,
                                hidden_layer_sizes,seed
                            ),
                            end="\r"
                        )
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
                                    else:
                                        raise NotImplementedError('unknown encoding type: {}'.format(enc_type))


                                    mlp = MLPClassifier(
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
                                            'Scale': scale,
                                            'Encoding': encoding_name,
                                            'Dataset': classification_dataset,
                                            'Model': 'MLP - {}'.format(hidden_layer_sizes),
                                            'Accuracy': acc,
                                            'Solver': solver,
                                            'Max Iter': max_iter,
                                        },
                                        ignore_index=True,
                                    )

                        mlp = MLPClassifier(
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
                                'Encoding': 'Normalized',
                                'Dataset': classification_dataset,
                                'Model': 'MLP - {}'.format(hidden_layer_sizes),
                                'Accuracy': acc,
                                'Solver': solver,
                                'Max Iter': max_iter,
                            },
                            ignore_index=True,
                        )
                    # df.to_csv('encoding_exp_results_iter{}_hs{}.csv'.format(
                    #     max_iter,
                    #     str(hidden_layer_sizes).replace(" ", "-").replace(",", ""))
                    # )
                    #
                    # df_all = df_all.append(df, ignore_index=True)
            # save each dataset individually, in case the run crashes and needs to be restarted
            df.to_csv(inter_fname)

        df_all = df_all.append(df, ignore_index=True)

    print("Saving All Results")

    if args.debug:
        df_all.to_csv('debug_enc_{}_results_{}iters.csv'.format(args.encoding_type, args.max_iters))
    else:
        df_all.to_csv('enc_{}_results_{}iters.csv'.format(args.encoding_type, args.max_iters))

if __name__ == '__main__':
    main()
