import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import glob
from tensorflow.python.summary.summary_iterator import summary_iterator
import os

encodings = [
    'proj-ssp',
    'sub-toroid-ssp',
]

enc_names = {
    'proj-ssp': 'Proj SSP',
    'sub-toroid-ssp': 'ST SSP',
}

func_names = {
    'distance': 'Distance',
    'direction': 'Direction',
    'centroid': 'Midpoint',
}

functions = [
    'distance',
    'direction',
    'centroid',
]

seeds = [
    1,
    2,
    3,
    4,
    5,
]

n_projs = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    50,
]

dim = 256
limit = 5
n_samples = 20000

fname_cache = "function_learning_proj_results.csv"

if os.path.exists(fname_cache):
    df = pd.read_csv(fname_cache)
else:
    df = pd.DataFrame()
    for seed in seeds:
        for n_proj in n_projs:
            for enc in encodings:
                for concat in [0, 1]:
                    for func in functions:
                        path = "proj_output_function/{}/{}_{}proj_concat{}_seed{}_dim{}_limit{}".format(
                            func,
                            enc,
                            n_proj,
                            concat,
                            seed,
                            dim,
                            limit,
                        )
                        print(os.getcwd() + "/" + path + "/*/events*")
                        fname = glob.glob(os.getcwd() + "/" + path + "/*/events*")[0]
                        train_loss = -1
                        test_loss = -1
                        # go through and get the value from the last epoch for each loss
                        for e in summary_iterator(fname):
                            for v in e.summary.value:
                                if v.tag == 'avg_loss':
                                    train_loss = v.simple_value
                                elif v.tag == 'test_loss':
                                    test_loss = v.simple_value
                        df = df.append(
                            {
                                'Train Loss': train_loss,
                                'Test Loss': test_loss,
                                'Seed': seed,
                                'Encoding': enc_names[enc],
                                'Function': func,
                                'Concat': concat,
                                'Proj Dim': n_proj
                            },
                            ignore_index=True
                        )

    df.to_csv(fname_cache)

# 1 is very bad in general
df = df[df['Proj Dim'] != 1]

fig, ax = plt.subplots(2, 3, figsize=(8, 4), sharex=True, sharey=False)

for concat in [0, 1]:
    for i, func in enumerate(functions):
        df_temp = df[(df['Concat'] == concat) & (df['Function'] == func)]
        sns.barplot(x='Encoding', y='Test Loss', hue='Proj Dim', data=df_temp, ax=ax[concat, i])
        if concat == 0:
            ax[0, i].set_xlabel("")
            ax[0, i].set_title(func_names[func])

        if i != 0:
            ax[concat, i].set_ylabel("")

sns.despine()

plt.show()
