from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

from constants import some_continuous, full_continuous, small_continuous
from feature_encoding import encode_dataset

from pmlb import fetch_data, classification_dataset_names

datasets = full_continuous
# datasets = small_continuous

n_datasets = len(datasets)

# contains all results
df = pd.DataFrame()

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

    hidden_layer_sizes = (256,)

    for seed in [1, 2, 3, 4, 5]:
        for scale in [1.0, 2.0, 5.0]:
            for dim in [128, 256, 512]:


                # train_X_enc = encode_dataset(train_X, dim=dim, seed=seed, scale=scale)
                # test_X_enc = encode_dataset(test_X, dim=dim, seed=seed, scale=scale)

                train_X_enc_scaled = encode_dataset(train_X_scaled, dim=dim, seed=seed, scale=scale)
                test_X_enc_scaled = encode_dataset(test_X_scaled, dim=dim, seed=seed, scale=scale)

                mlp = MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation='relu',
                    solver=solver,
                    max_iter=300,
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
                    },
                    ignore_index=True,
                )

        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver=solver,
            max_iter=300,
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
            },
            ignore_index=True,
        )

print("Saving Results")

df.to_csv('encoding_exp_results.csv')
