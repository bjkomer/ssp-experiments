import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import numpy as np
import glob
from constants import full_continuous, small_continuous_regression

# this just determines whether the label is accuracy or R2
regression = False

data_to_plot = 'all'
# data_to_plot = 'regression'
# data_to_plot = 'classification'

meta_df = pd.read_csv('metadata.csv')

if len(sys.argv) > 1:
    # one or more files given, load them all into one dataframe
    fnames = sys.argv[1:]
else:
    fnames = []
    # results on the majority of the encodings for classification (51)
    fnames += glob.glob('process_output/enc_all_results_600iters_*')
    # results on the majority of the encodings for regression (72)
    fnames += glob.glob('regression_no_target_scaling/enc_all_results_600iters_*')
    # results on legendre encoding for classification (51)
    fnames += glob.glob('legendre_output/*')
    # results on legendre encoding for regression (72)
    fnames += glob.glob('legendre_regression/*')
    # results with SSP Proj encoding (hex ssp) (120)
    fnames += glob.glob('ssp_proj_output/*')

df = pd.DataFrame()
for fname in fnames:
    df = df.append(pd.read_csv(fname))


# df = df.merge(meta_df, on='Dataset')

if not os.path.exists('figures'):
    os.makedirs('figures')

order = [
    'Hex SSP', 'SSP', 'Combined SSP', 'Simplex SSP',
    'RBF', 'Tile Coding', 'One Hot', 'Normalized', 'Legendre'
]

# optional pairwise views

# df = df[df['Encoding'] != 'RBF']
# df = df[df['Encoding'] != 'One Hot']
# df = df[df['Encoding'] != 'Tile Coding']
# df = df[df['Encoding'] != 'Normalized']
# df = df[df['Encoding'] != 'SSP Normalized']
# df = df[df['Encoding'] != 'Combined SSP Normalized']
# df = df[df['Encoding'] != 'Combined Simplex SSP Normalized']


# renaming labels

df = df.replace(
    {
        'SSP Normalized': 'SSP',
        'Combined SSP Normalized': 'Combined SSP',
        'Combined Simplex SSP Normalized': 'Simplex SSP',
        'SSP Projected Axis': 'Hex SSP'
    }
)

if regression:
    metric = 'R2'
    df = df.rename(columns=
        {
            'Accuracy': 'R2',
        }
    )
else:
    metric = 'Accuracy'

# optionally trim the dataframe
if data_to_plot == 'classification':
    df = df[df['Dataset'].isin(full_continuous)]
elif data_to_plot == 'regression':
    df = df[df['Dataset'].isin(small_continuous_regression)]


# # removing original SSP, for debugging
# df = df[df['Encoding'] != 'SSP']


####################
# Overall Averages #
####################

plt.figure()
sns.barplot(data=df, x='Encoding', y=metric, order=order)
sns.despine()

##############################
# Best Encodings per Dataset #
##############################

across_models = True

encodings = df['Encoding'].unique()

models = df['Model'].unique()

datasets = df['Dataset'].unique()

n_encodings = len(encodings)
n_datasets = len(datasets)
n_models = len(models)

# initialize dictionary with empty lists
best_dict = {}
for encoding in encodings:
    best_dict[encoding] = []

if across_models:
    accs = np.zeros((n_encodings, n_models, n_datasets))

    for di, dataset_name in enumerate(datasets):
        for ei, encoding in enumerate(encodings):
            for mi, model in enumerate(models):
                accs[ei, mi, di] = df[(df['Dataset'] == dataset_name) & (df['Encoding'] == encoding) & (df['Model'] == model)][metric].mean()
                if np.isnan(accs[ei, mi, di]):
                    accs[ei, mi, di] = 0

        if np.sum(accs[:, :, di]) != 0:
            # make sure there is some data
            ind_best = np.unravel_index(np.argmax(accs[:, :, di], axis=None), accs[:, :, di].shape)
            best_dict[encodings[ind_best[0]]].append(dataset_name)
else:
    accs = np.zeros((n_encodings, n_datasets))

    for di, dataset_name in enumerate(datasets):
        for ei, encoding in enumerate(encodings):
            accs[ei, di] = df[(df['Dataset'] == dataset_name) & (df['Encoding'] == encoding)][metric].mean()
            if np.isnan(accs[ei, di]):
                accs[ei, di] = 0

        if np.sum(accs[:, di]) != 0:
            # make sure there is some data
            ind_best = np.argmax(accs[:, di])
            best_dict[encodings[ind_best]].append(dataset_name)

df_best = pd.DataFrame()

for encoding in encodings:
    df_best = df_best.append(
        {
            'Encoding': encoding,
            'Datasets': len(best_dict[encoding]),
        },
        ignore_index=True
    )

plt.figure(figsize=(4, 4))
sns.barplot(data=df_best, x='Encoding', y='Datasets', order=order)
sns.despine()

########################
# Pairwise Comparisons #
########################

# get correct colours
default_palette = sns.color_palette()

colour_dict = {}
for ei, encoding in enumerate(encodings):
    colour_dict[encoding] = default_palette[ei]

pairs = [
    # ['SSP', 'Normalized'],
    # ['SSP', 'Tile Coding'],
    # ['SSP', 'One Hot'],
    # ['SSP', 'RBF'],

    # ['SSP', 'RBF Tiled'],
    # ['Combined SSP', 'Normalized'],
    # ['Simplex SSP', 'Normalized'],

    # ['SSP Projected Axis', 'Normalized'],
    # ['SSP Projected Axis', 'Tile Coding'],
    # ['SSP Projected Axis', 'One Hot'],
    # ['SSP Projected Axis', 'RBF'],
    # ['SSP Projected Axis', 'SSP'],

    ['Hex SSP', 'Normalized'],
    ['Hex SSP', 'Tile Coding'],
    ['Hex SSP', 'One Hot'],
    ['Hex SSP', 'RBF'],
    ['Hex SSP', 'Legendre'],
    ['Hex SSP', 'SSP'],

]

df_bests = []

for pair in pairs:
    df_pair = df[df['Encoding'].isin(pair)]

    # initialize dictionary with empty lists
    best_dict = {}
    for encoding in encodings:
        best_dict[encoding] = []

    if across_models:
        accs = np.zeros((n_encodings, n_models, n_datasets))

        for di, dataset_name in enumerate(datasets):
            for ei, encoding in enumerate(encodings):
                for mi, model in enumerate(models):
                    accs[ei, mi, di] = \
                        df_pair[(df_pair['Dataset'] == dataset_name) & (df_pair['Encoding'] == encoding) & (df_pair['Model'] == model)][
                        metric].mean()
                    if np.isnan(accs[ei, mi, di]):
                        accs[ei, mi, di] = 0

            if np.sum(accs[:, :, di]) != 0:
                # make sure there is some data
                ind_best = np.unravel_index(np.argmax(accs[:, :, di], axis=None), accs[:, :, di].shape)
                best_dict[encodings[ind_best[0]]].append(dataset_name)
    else:
        accs = np.zeros((n_encodings, n_datasets))

        for di, dataset_name in enumerate(datasets):
            for ei, encoding in enumerate(encodings):
                accs[ei, di] = df_pair[(df_pair['Dataset'] == dataset_name) & (df_pair['Encoding'] == encoding)][metric].mean()
                if np.isnan(accs[ei, di]):
                    accs[ei, di] = 0

            if np.sum(accs[:, di]) != 0:
                # make sure there is some data
                ind_best = np.argmax(accs[:, di])
                best_dict[encodings[ind_best]].append(dataset_name)

    df_bests.append(pd.DataFrame())

    for encoding in pair:
        df_bests[-1] = df_bests[-1].append(
            {
                'Encoding': encoding,
                'Datasets': len(best_dict[encoding]),
            },
            ignore_index=True
        )

# plt.figure(figsize=(8, 4))
fig, ax = plt.subplots(1, len(pairs), sharey=True, sharex=False, figsize=(8, 4))
for i, df_b in enumerate(df_bests):
    sns.barplot(data=df_b, x='Encoding', y='Datasets', ax=ax[i], palette=colour_dict)
    ax[i].xaxis.set_label_text("")

ax[1].yaxis.set_label_text("")
ax[2].yaxis.set_label_text("")
ax[3].yaxis.set_label_text("")
# fig.suptitle("Pairwise Comparisons")
fig.text(0.5, 0.04, 'Encoding', ha='center')
sns.despine()


################################
# Bootstrapping Best Encodings #
################################

# takes a while, disable by default
if False:

    df_orig = df.copy()

    df_best = pd.DataFrame()

    seed_sets = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [1, 4, 7],
        [2, 5, 8],
        [3, 6, 9],
        [1, 5, 9],
        [2, 6, 7],
        [3, 4, 8],
        [1, 8, 6],
        [4, 2, 9],
        [7, 5, 3],
    ]

    seed_sets = np.random.randint(low=1, high=10, size=(100, 3))

    for seed_set in seed_sets:

        df = df_orig[df_orig['Seed'].isin(seed_set)]

        # initialize dictionary with empty lists
        best_dict = {}
        for encoding in encodings:
            best_dict[encoding] = []

        if across_models:
            accs = np.zeros((n_encodings, n_models, n_datasets))

            for di, dataset_name in enumerate(datasets):
                for ei, encoding in enumerate(encodings):
                    for mi, model in enumerate(models):
                        accs[ei, mi, di] = df[(df['Dataset'] == dataset_name) & (df['Encoding'] == encoding) & (df['Model'] == model)][metric].mean()
                        if np.isnan(accs[ei, mi, di]):
                            accs[ei, mi, di] = 0

                if np.sum(accs[:, :, di]) != 0:
                    # make sure there is some data
                    ind_best = np.unravel_index(np.argmax(accs[:, :, di], axis=None), accs[:, :, di].shape)
                    best_dict[encodings[ind_best[0]]].append(dataset_name)
        else:
            accs = np.zeros((n_encodings, n_datasets))

            for di, dataset_name in enumerate(datasets):
                for ei, encoding in enumerate(encodings):
                    accs[ei, di] = df[(df['Dataset'] == dataset_name) & (df['Encoding'] == encoding)][metric].mean()
                    if np.isnan(accs[ei, di]):
                        accs[ei, di] = 0

                if np.sum(accs[:, di]) != 0:
                    # make sure there is some data
                    ind_best = np.argmax(accs[:, di])
                    best_dict[encodings[ind_best]].append(dataset_name)

        for encoding in encodings:
            df_best = df_best.append(
                {
                    'Encoding': encoding,
                    'Datasets': len(best_dict[encoding]),
                },
                ignore_index=True
            )

    plt.figure(figsize=(4, 4))
    sns.barplot(data=df_best, x='Encoding', y='Datasets')
    sns.despine()

plt.show()
