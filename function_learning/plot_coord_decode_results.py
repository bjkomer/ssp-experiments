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
    'random': 'Random',
}


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
fname = "coord_decode_results.csv"
if os.path.exists(fname):
    df = pd.read_csv(fname)
else:
    df = pd.DataFrame()

    for seed in seeds:
        for enc in encodings:

            path = "coord_decode_function/exps_dim256_limit5_20000samples/{}_seed{}".format(
                enc,
                seed,
            )
            print(os.getcwd() + "/" + path + "/*/events*")
            summary_name = glob.glob(os.getcwd() + "/" + path + "/*/events*")[0]
            train_loss = -1
            test_loss = -1
            # go through and get the value from the last epoch for each loss
            for e in summary_iterator(summary_name):
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
                },
                ignore_index=True
            )

    df.to_csv(fname)

df_top = df[(df['Encoding'] == 'SSP') | (df['Encoding'] == 'Hex SSP') | (df['Encoding'] == 'RBF')]

fig, ax = plt.subplots(1, 2, figsize=(8.5, 4), tight_layout=True, gridspec_kw={'width_ratios': [6, 3]})

sns.barplot(x='Encoding', y='Test Loss', data=df, ax=ax[0])
sns.barplot(x='Encoding', y='Test Loss', data=df_top, ax=ax[1])

# fig.suptitle('Learning to Decode')

sns.despine()

plt.show()
