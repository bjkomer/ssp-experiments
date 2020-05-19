import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from itertools import product
from scipy.stats import ttest_ind

# fname = '/home/bjkomer/ssp-navigation/ssp_navigation/eval_data_tt/med_dim_adam_exps/combined_data.csv'
# fname = '/home/bjkomer/ssp-navigation/ssp_navigation/eval_data_tt/bounds_exps/combined_data.csv'

bounds = False#True#False
label_fontsize = 15#20
tick_fontsize = 12#16

if bounds:
    folder = 'data/bounds_exps'
    old_names = ['hex-ssp', 'ssp', 'pc-gauss', 'tile-coding', 'one-hot']
    order = ['Hex SSP', 'SSP', 'RBF', 'Tile-Code', 'One-Hot']
else:
    folder = 'data/med_dim_adam_exps'
    old_names = ['hex-ssp', 'ssp', 'pc-gauss', 'tile-coding', 'one-hot', 'learned', '2d', 'random']
    order = ['Hex SSP', 'SSP', 'RBF', 'Tile-Code', 'One-Hot', 'Learned', '2D', 'Random']

fname = 'combined_data.csv'

combined_fname = os.path.join(folder, fname)

best_epoch = False


if os.path.isfile(combined_fname) and False:
    # combined file already exists, load it
    df = pd.read_csv(combined_fname)
else:
    # combined file does not exist, create it
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

    df = pd.DataFrame()

    for file in files:
        if file == fname:
            continue
        df_temp = pd.read_csv(os.path.join(folder, file))

        # if 'SSP Scaling' not in df_temp:
        #     df_temp['SSP Scaling'] = 0.5

        # only want to look at the correct sigma results in this case
        if 'bounds_exps' in folder:
            # if np.any(df_temp['Encoding'] == 'pc-gauss') and np.any(df_temp['Sigma'] != 0.375):
            if ('pc-gauss' in file) and ('scaling' in file):
                continue
            else:
                df = df.append(df_temp)
        else:

            df = df.append(df_temp)

    df.to_csv(combined_fname)


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


# df_test = df[df['Number of Mazes'] == 25]
df_test = df[df['Number of Mazes'] == 50]

data1 = df_test[df_test['Encoding'] == 'Hex SSP']['Angular RMSE']
# data1 = df_test[df_test['Encoding'] == 'RBF']['Angular RMSE']
data2 = df_test[df_test['Encoding'] == 'SSP']['Angular RMSE']

# data1 = df_test[df_test['Encoding'] == 'Learned']['Angular RMSE']
# data2 = df_test[df_test['Encoding'] == 'Tile-Code']['Angular RMSE']

print(data1)
print(data2)

stat, p = ttest_ind(data1, data2)
print('stat=%.3f, p=%.5f' % (stat, p))
if p > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')

plt.show()
