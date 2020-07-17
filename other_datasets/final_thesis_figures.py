import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import numpy as np
import glob
from constants import full_continuous, small_continuous_regression
import argparse

parser = argparse.ArgumentParser('Figures depicting performance on PMLB datasets')

parser.add_argument('--overwrite-cache', action='store_true', help='overwrite the combined dataframe')
parser.add_argument('--include-bootstrap', action='store_true', help='include bootstapped confidence interval plots')
parser.add_argument('--n-bootstrap-samples', type=int, default=100)

args = parser.parse_args()


meta_df = pd.read_csv('metadata.csv')

fnames = []
# results on the majority of the encodings for classification (51)
fnames += glob.glob('process_output/enc_all_results_600iters_*')
# results on the majority of the encodings for regression (72)
fnames += glob.glob('regression_no_target_scaling/enc_all_results_600iters_*')
# results on legendre encoding for classification (51)
# fnames += glob.glob('legendre_output/*')
# results with SSP Proj encoding (hex ssp) (122)
fnames += glob.glob('ssp_proj_output/*')

old_names = [
    'Hex SSP', 'SSP', 'Combined SSP', 'Simplex SSP',
    'RBF', 'Tile Coding', 'One Hot', 'Normalized',
]

# shortened names
order = [
    'Hex SSP', 'SSP', 'C-SSP', 'S-SSP',
    'RBF', 'TC', 'OH', 'Norm',
]

cache = 'final_thesis_data.csv'

if not os.path.exists(cache) or args.overwrite_cache:
    df_all = pd.DataFrame()
    for fname in fnames:
        df_all = df_all.append(pd.read_csv(fname))

    # renaming labels
    df_all = df_all.replace(
        {
            'SSP Normalized': 'SSP',
            'Combined SSP Normalized': 'Combined SSP',
            'Combined Simplex SSP Normalized': 'Simplex SSP',
            'SSP Projected Axis': 'Hex SSP'
        }
    )

    # Replace with shorter names to fit on the figure
    for i in range(len(old_names)):
        df_all = df_all.replace(old_names[i], order[i])

    df_all.to_csv(cache)
else:
    df_all = pd.read_csv(cache)

if not os.path.exists('figures'):
    os.makedirs('figures')


# optional pairwise views

# df = df[df['Encoding'] != 'RBF']
# df = df[df['Encoding'] != 'One Hot']
# df = df[df['Encoding'] != 'Tile Coding']
# df = df[df['Encoding'] != 'Normalized']
# df = df[df['Encoding'] != 'SSP Normalized']
# df = df[df['Encoding'] != 'Combined SSP Normalized']
# df = df[df['Encoding'] != 'Combined Simplex SSP Normalized']


df_clf = df_all[df_all['Dataset'].isin(full_continuous)]
df_reg = df_all[df_all['Dataset'].isin(small_continuous_regression)]

# # removing original SSP, for debugging
# df = df[df['Encoding'] != 'SSP']

####################
# Overall Averages #
####################

print("Average Result per Encoding")

fig, ax = plt.subplots(1, 2, figsize=(8, 3), tight_layout=True)

for i, df in enumerate([df_clf, df_reg]):
    if i == 1:
        metric = 'R2'
        df = df.rename(columns=
            {
                'Accuracy': 'R2',
            }
        )
        # set negative values to zero as a form of outlier rejection
        # to get a more useful average measure
        df[df['R2'] < 0] = 0
    else:
        metric = 'Accuracy'

    sns.barplot(data=df, x='Encoding', y=metric, ax=ax[i], order=order)
sns.despine()

fig.savefig('figures/pmlb_all_results.pdf')

##############################
# Best Encodings per Dataset #
##############################

print("Best Encodings per Dataset")

# fig, ax = plt.subplots(1, 2, figsize=(8, 3), tight_layout=True)
fig, ax = plt.subplots(2, 1, figsize=(8, 6), tight_layout=True)
for i, df_base in enumerate([df_clf, df_reg]):
    if i == 1:
        metric = 'R2'
        df_base = df_base.rename(columns=
            {
                'Accuracy': 'R2',
            }
        )
    else:
        metric = 'Accuracy'


    print("Bootstrapping Best Encodings Encodings - {}".format(metric))
    rng = np.random.RandomState(seed=13)
    seed_sets = rng.randint(low=1, high=10, size=(args.n_bootstrap_samples, 3))

    cache_fname = 'cache_' + metric + '.csv'

    if not os.path.exists(cache_fname):

        df_best = pd.DataFrame()

        for seed_set in seed_sets:

            df = df_base[df_base['Seed'].isin(seed_set)]

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



            for encoding in encodings:
                df_best = df_best.append(
                    {
                        'Encoding': encoding,
                        'Datasets': len(best_dict[encoding]),
                    },
                    ignore_index=True
                )
        df_best.to_csv(cache_fname)
    else:
        df_best = pd.read_csv(cache_fname)

    sns.barplot(data=df_best, x='Encoding', y='Datasets', ax=ax[i], order=order)
ax[0].set_ylim([0, 30])
sns.despine()

fig.savefig('figures/pmlb_best_encoding.pdf')

########################
# Pairwise Comparisons #
########################

print("Pairwise Comparisons")

encodings = df_all['Encoding'].unique()
models = df_all['Model'].unique()
datasets = df_all['Dataset'].unique()

n_encodings = len(encodings)
n_datasets = len(datasets)
n_models = len(models)

# get correct colours
default_palette = sns.color_palette()

colour_dict = {}
for ei, encoding in enumerate(order):
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

    # ['Hex SSP', 'SSP'],
    # ['Hex SSP', 'Combined SSP'],
    # ['Hex SSP', 'Simplex SSP'],
    # ['Hex SSP', 'RBF'],
    # ['Hex SSP', 'Tile Coding'],
    # ['Hex SSP', 'One Hot'],
    # ['Hex SSP', 'Normalized'],

    ['Hex SSP', 'SSP'],
    ['Hex SSP', 'C-SSP'],
    ['Hex SSP', 'S-SSP'],
    ['Hex SSP', 'RBF'],
    ['Hex SSP', 'TC'],
    ['Hex SSP', 'OH'],
    ['Hex SSP', 'Norm'],

]

df_bests = []

df_all = df_all.rename(columns=
    {
        'R2': 'Accuracy',
    }
)

metric = 'Accuracy'  # just using the same name, since it is just a rank anyway

# takes a while, disable by default
if args.include_bootstrap:
    print("Bootstrapping Paired Encodings")
    rng = np.random.RandomState(seed=13)
    seed_sets = rng.randint(low=1, high=10, size=(args.n_bootstrap_samples, 3))

    for pair in pairs:

        print("Computing bootstrap for pair:")
        print(pair)

        df_pair_all = df_all[df_all['Encoding'].isin(pair)]

        df_bests.append(pd.DataFrame())

        cache_fname = 'cache_' + pair[1] + '.csv'

        if not os.path.exists(cache_fname):

            for seed_set in seed_sets:

                df_pair = df_pair_all[df_pair_all['Seed'].isin(seed_set)]

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

                for encoding in pair:
                    df_bests[-1] = df_bests[-1].append(
                        {
                            'Encoding': encoding,
                            'Datasets': len(best_dict[encoding]),
                        },
                        ignore_index=True
                    )

            df_bests[-1].to_csv(cache_fname)
        else:
            df_bests[-1] = pd.read_csv(cache_fname)

else:
    for pair in pairs:
        df_pair = df_all[df_all['Encoding'].isin(pair)]

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
fig, ax = plt.subplots(1, len(pairs), sharey=True, sharex=False, figsize=(8, 3), tight_layout=True)
for i, df_b in enumerate(df_bests):
    sns.barplot(data=df_b, x='Encoding', y='Datasets', ax=ax[i], palette=colour_dict)
    if i != 0:
        ax[i].yaxis.set_label_text("")
    ax[i].xaxis.set_label_text("")

# fig.suptitle("Pairwise Comparisons")
# fig.text(0.5, 0.04, 'Encoding', ha='center')
sns.despine()

fig.savefig('figures/pmlb_pairwise_results.pdf')

################################
# Bootstrapping Best Encodings #
################################

# takes a while, disable by default
if args.include_bootstrap:
    print("Bootstrapping Best Encodings")

    cache_fname = 'bootstrap_cache.csv'

    if os.path.exists(cache_fname):
        df_best = pd.read_csv(cache_fname)
    else:

        # df_orig = df_all.copy()

        df_best = pd.DataFrame()

        # seed_sets = [
        #     [1, 2, 3],
        #     [4, 5, 6],
        #     [7, 8, 9],
        #     [1, 4, 7],
        #     [2, 5, 8],
        #     [3, 6, 9],
        #     [1, 5, 9],
        #     [2, 6, 7],
        #     [3, 4, 8],
        #     [1, 8, 6],
        #     [4, 2, 9],
        #     [7, 5, 3],
        # ]

        rng = np.random.RandomState(seed=13)
        seed_sets = rng.randint(low=1, high=10, size=(args.n_bootstrap_samples, 3))

        for seed_set in seed_sets:

            df = df_all[df_all['Seed'].isin(seed_set)]

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

        df_best.to_csv(cache_fname)

    fig, ax = plt.subplots(1, 1, figsize=(6, 3), tight_layout=True)
    sns.barplot(data=df_best, x='Encoding', y='Datasets', ax=ax, order=order)
    sns.despine()
    fig.savefig('figures/pmlb_bootstrapped_all_results.pdf')

    for encoding in encodings:
        print(encoding)
        print(df_best[df_best['Encoding'] == encoding]['Datasets'].mean())

plt.show()
