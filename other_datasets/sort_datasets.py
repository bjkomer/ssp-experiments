# Sort out which encodings work the best on which datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from constants import full_continuous
import sys

across_models = False#True

if len(sys.argv) > 1:
    # one or more files given, load them all into one dataframe
    fnames = sys.argv[1:]
    df = pd.DataFrame()
    for fname in fnames:
        df = df.append(pd.read_csv(fname))

meta_df = pd.read_csv('metadata.csv')
df = df.merge(meta_df, on='Dataset')

# hardcode model of interest here. TODO: make parameter
# df = df[df['Model'] == 'MLP - (512, 512)']
# df = df[df['Model'] == 'MLP - (1024,)']
# df = df[df['Model'] == 'MLP - (512,)']
# df = df[df['Model'] == 'MLP - (1024, 1024)']
# df = df[df['Model'] == 'MLP - (256, 512)']
# df = df[df['Model'] == 'MLP - (512, 256)']

encodings = df['Encoding'].unique()

models = df['Model'].unique()

n_encodings = len(encodings)
n_datasets = len(full_continuous)
n_models = len(models)

# initialize dictionary with empty lists
best_dict = {}
for encoding in encodings:
    best_dict[encoding] = []

if across_models:
    accs = np.zeros((n_encodings, n_models, n_datasets))

    for di, dataset_name in enumerate(full_continuous):
        for ei, encoding in enumerate(encodings):
            for mi, model in enumerate(models):
                accs[ei, mi, di] = df[(df['Dataset'] == dataset_name) & (df['Encoding'] == encoding) & (df['Model'] == model)]['Accuracy'].mean()
                if np.isnan(accs[ei, mi, di]):
                    accs[ei, mi, di] = 0

        if np.sum(accs[:, :, di]) > 0:
            # make sure there is some data
            ind_best = np.unravel_index(np.argmax(accs[:, :, di], axis=None), accs[:, :, di].shape)
            best_dict[encodings[ind_best[0]]].append(dataset_name)
else:
    accs = np.zeros((n_encodings, n_datasets))

    for di, dataset_name in enumerate(full_continuous):
        for ei, encoding in enumerate(encodings):
            accs[ei, di] = df[(df['Dataset'] == dataset_name) & (df['Encoding'] == encoding)]['Accuracy'].mean()
            if np.isnan(accs[ei, di]):
                accs[ei, di] = 0

        if np.sum(accs[:, di]) > 0:
            # make sure there is some data
            ind_best = np.argmax(accs[:, di])
            best_dict[encodings[ind_best]].append(dataset_name)

for encoding in encodings:
    result_df = meta_df[meta_df['Dataset'].isin(best_dict[encoding])]
    print("")
    print("{} best: {}".format(encoding, best_dict[encoding]))
    print("Number of datasets: {}".format(len(best_dict[encoding])))
    print("Avg # of features: {}".format(result_df['Float Features'].mean()))
    print("Avg imbalance metric: {}".format(result_df['Imbalance Metric'].mean()))
    print("Avg # of samples: {}".format(result_df['Number of Samples'].mean()))
    print("Avg overall Acc: {}".format(df[df['Encoding'] == encoding]['Accuracy'].mean()))
    print("")
