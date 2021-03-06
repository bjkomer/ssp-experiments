import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from itertools import product

# fname = '/home/bjkomer/ssp-navigation/ssp_navigation/eval_data_tt/med_dim_adam_exps/combined_data.csv'
# fname = '/home/bjkomer/ssp-navigation/ssp_navigation/eval_data_tt/bounds_exps/combined_data.csv'

bounds = False#True#False
label_fontsize = 15#20
tick_fontsize = 12#16

folder_bounds = 'data/bounds_exps'
folder_other = 'data/med_dim_adam_exps'
old_names = ['hex-ssp', 'ssp', 'pc-gauss', 'tile-coding', 'one-hot', 'learned', '2d', 'random']
order = ['Hex SSP', 'SSP', 'RBF', 'Tile-Code', 'One-Hot', 'Learned', '2D', 'Random']

fname = 'combined_data.csv'


best_epoch = False

df = pd.DataFrame()

for folder in [folder_bounds, folder_other]:
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

    for file in files:
        if 'combined' in file:
            continue
        df_temp = pd.read_csv(os.path.join(folder, file))

        if 'bounds_exps' in folder:
            # only want to look at the correct sigma results in this case
            if ('pc-gauss' in file) and ('scaling' in file):
                continue
            else:
                df = df.append(df_temp)
        elif 'med_dim_adam_exps' in folder:
            # only want the ones not originally included in bounds exps
            if ('learned' in file) or ('2d' in file) or ('random' in file):
                df = df.append(df_temp)
        else:

            df = df.append(df_temp)

if best_epoch:
    # Consider only the number of epochs trained for that result in the best test RMSE
    columns = [
        'Dimensionality',
        'Hidden Layer Size',
        'Hidden Layers',
        'Encoding',
        'Seed',
        'Maze ID Type',

        # # Other differentiators
        # 'Number of Mazes Tested',
        # 'Number of Goals Tested',

        # Command line supplied tags
        'Dataset',
        # 'Trained On',
        # 'Epochs',  # this will be used to compute the max over
        'Batch Size',
        'Number of Mazes',
        'Loss Function',
        'Learning Rate',
        'Momentum',
        # 'Sigma',
        # 'Hex Freq Coef',
    ]

    unique_column_dict = {}

    for column in columns:
        unique_column_dict[column] = df[column].unique()

    ordered_column_names = list(unique_column_dict.keys())

    # New dataframe containing only the best early-stop rows
    df_new = pd.DataFrame()

    for column_spec in product(*[unique_column_dict[column] for column in ordered_column_names]):
        # print(ordered_column_names)
        # print(column_spec)
        df_tmp = df
        # load only the columns that match the current spec
        for i in range(len(ordered_column_names)):
            # print("{} == {}".format(ordered_column_names[i], column_spec[i]))
            df_tmp = df_tmp[df_tmp[ordered_column_names[i]] == column_spec[i]]

        if df_tmp.empty:
            # It is possible there is no data for this combination. If so, skip it
            continue
        else:
            # Only difference between rows is the epochs trained for.
            # Find the one with the best result (i.e. not overfit or underfit, best time to early-stop)
            df_new = df_new.append(df_tmp.loc[df_tmp['Angular RMSE'].idxmin()])

    # overwrite old dataframe with the new one
    df = df_new

    print(df)


print(df)

# df = df[df['Number of Mazes'] == 10]
# df = df[df['Dimensionality'] == 256]
# df = df[df['Number of Mazes'] == 20]
# df = df[df['Dimensionality'] == 512]
# df = df[df['Number of Mazes'] == 5]
# df = df[df['Number of Mazes'] == 64]

# TODO: possibly group colour-code:
#  SSP Methods: hex-ssp ssp
#  Standard Encoding Methods: pc-gauss, one-hot, tile-coding
#  Standard NN Methods: 2d, learned
#  Worst-Case Baseline: random

# Replace all encoding names with paper friendly versions
for i in range(len(old_names)):
    df = df.replace(old_names[i], order[i])

plt.figure(figsize=(8, 5), tight_layout=True)
ax = sns.barplot(data=df, x='Encoding', y='Angular RMSE', hue='Number of Mazes', order=order)
ax.set_xlabel('Encoding', fontsize=label_fontsize)
ax.set_ylabel('RMSE', fontsize=label_fontsize)
ax.tick_params(labelsize=tick_fontsize)
sns.despine()

plt.figure(figsize=(8, 5), tight_layout=True)
ax = sns.barplot(data=df, x='Encoding', y='Angular RMSE', order=order)
ax.set_xlabel('Encoding', fontsize=label_fontsize)
ax.set_ylabel('RMSE', fontsize=label_fontsize)
ax.tick_params(labelsize=tick_fontsize)
sns.despine()


plt.show()
