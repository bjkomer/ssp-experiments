import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import glob
from tensorflow.python.summary.summary_iterator import summary_iterator
import os

encodings = [
    'ssp',
    'hex-ssp',
    'pc-gauss',
    'one-hot',
    'tile-coding',
    'legendre',
]

enc_names = {
    'ssp': 'SSP',
    'hex-ssp': 'Hex SSP',
    'pc-gauss': 'RBF',
    'one-hot': 'One-Hot',
    'tile-coding': 'Tile-Code',
    'legendre': 'Legendre',
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

dim = 256
limit = 5.
n_samples = 20000

df = pd.DataFrame()

for seed in seeds:
    for enc in encodings:
        for concat in [0, 1]:
            for func in functions:
                path = "output_function/{}/{}_concat{}_d{}_limit{}_seed{}_{}samples".format(
                    func,
                    enc,
                    concat,
                    dim,
                    limit,
                    seed,
                    n_samples,
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
                    },
                    ignore_index=True
                )

df.to_csv("function_learning_results.csv")

fig, ax = plt.subplots(2, 3, figsize=(6, 4), sharex=True, sharey=False)

for concat in [0, 1]:
    for i, func in enumerate(functions):
        df_temp = df[(df['Concat'] == concat) & (df['Function'] == func)]
        sns.barplot(x='Encoding', y='Test Loss', data=df_temp, ax=ax[concat, i])
        if concat == 0:
            ax[0, i].set_xlabel("")
            ax[0, i].set_title(func_names[func])

        if i != 0:
            ax[concat, i].set_ylabel("")

sns.despine()

plt.show()
