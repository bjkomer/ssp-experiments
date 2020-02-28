# list the names of the datasets that only include continuous features
from pmlb import fetch_data, classification_dataset_names, regression_dataset_names
from pmlb.write_metadata import count_features_type, imbalance_metrics
from constants import full_continuous, some_continuous, full_continuous_regression, small_continuous_regression
import pandas as pd


def determine_endpoint_type(features):
    """ Determines the type of an endpoint
    :param features: pandas.DataFrame
        A dataset in a panda's data frame
    :returns string
        string with a name of a dataset
    """
    counter={k.name: v for k, v in features.columns.to_series().groupby(features.dtypes).groups.items()}
    if len(features.groupby('target').apply(list))==2:
        return 'binary'
    if 'float64' in counter:
        return 'float'
    return 'integer'

# dataset_names = full_continuous
dataset_names = full_continuous_regression
dataset_names = full_continuous + small_continuous_regression

n_datasets = len(dataset_names)

meta_df = pd.DataFrame()

for i, dataset in enumerate(dataset_names):
    print('\x1b[2K\r {} of {}. {}'.format(i+1, n_datasets, dataset), end="\r")
    df = fetch_data(dataset, return_X_y=False, local_cache_dir='cache')
    # feat = count_features_type(df.ix[:, df.columns != 'class'])
    feat = count_features_type(df.ix[:, df.columns != 'target'])
    n_binary = feat[0]
    n_integer = feat[1]
    n_float = feat[2]

    endpoint = determine_endpoint_type(df.ix[:, df.columns == 'target'])
    mse = imbalance_metrics(df['target'].tolist())

    if dataset in full_continuous:
        dataset_type = 'Classificaion'
    elif dataset in full_continuous_regression:
        dataset_type = 'Regression'
    else:
        assert False

    meta_df = meta_df.append(
        {
            'Dataset': dataset,
            'Float Features': n_float,
            'Integer Features': n_integer,
            'Binary Features': n_binary,
            'Endpoint': endpoint,
            'Number of Samples': len(df.axes[0]),
            'Number of Classes': mse[0],
            'Imbalance Metric': mse[1],
            'Type': dataset_type,
        },
        ignore_index=True
    )

meta_df.to_csv('metadata.csv')
