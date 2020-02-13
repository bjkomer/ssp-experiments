from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from constants import some_continuous, full_continuous, small_continuous
from feature_encoding import encode_dataset

from pmlb import fetch_data, classification_dataset_names


@ignore_warnings(category=ConvergenceWarning)
def main():

    datasets = full_continuous
    # datasets = small_continuous

    # # # TEMP TEST
    # datasets = ['banana']
    # datasets = ['appendicitis']
    # datasets = ['diabetes']
    # datasets = ['titanic']
    datasets = ['banana', 'appendicitis', 'diabetes', 'titanic']

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

        max_iters = [300]
        hidden_layers = [(512,), (1024,), (256, 256)]
        seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        scales = [0.25, 1.0]
        dims = [256]

        # contains all results for this dataset
        df = pd.DataFrame()
        for max_iter in max_iters:
            for hidden_layer_sizes in hidden_layers:
                for seed in seeds:
                    for scale in scales:
                        for dim in dims:

                            # train_X_enc = encode_dataset(train_X, dim=dim, seed=seed, scale=scale)
                            # test_X_enc = encode_dataset(test_X, dim=dim, seed=seed, scale=scale)

                            train_X_enc_scaled = encode_dataset(train_X_scaled, dim=dim, seed=seed, scale=scale)
                            test_X_enc_scaled = encode_dataset(test_X_scaled, dim=dim, seed=seed, scale=scale)

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
                                    'Encoding': 'SSP Normalized',
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
        df.to_csv('intermediate/encoding_exp_results_300iters_{}.csv'.format(classification_dataset))

        df_all = df_all.append(df, ignore_index=True)

    print("Saving All Results")

    df_all.to_csv('encoding_exp_all_results_300iters.csv')

if __name__ == '__main__':
    main()
