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
limit = 5.
n_samples = 20000
fname_cache = "coord_decode_proj_results.csv"
if os.path.exists(fname_cache):
    df = pd.read_csv(fname_cache)
else:
    df = pd.DataFrame()

    for seed in seeds:
        for n_proj in n_projs:
            for enc in encodings:

                path = "coord_decode_function_proj/exps_dim256_limit5_20000samples/{}_{}proj_seed{}".format(
                    enc,
                    n_proj,
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
                        'Proj Dim': n_proj,
                    },
                    ignore_index=True
                )

    df.to_csv(fname_cache)

# df_top = df[(df['Encoding'] == 'SSP') | (df['Encoding'] == 'Hex SSP') | (df['Encoding'] == 'RBF')]

# 1 is very bad in general
df = df[df['Proj Dim'] != 1]

df_proj = df[(df['Encoding'] == 'Proj SSP')]
df_st = df[(df['Encoding'] == 'ST SSP')]

fig, ax = plt.subplots(1, 2, figsize=(8.5, 4), tight_layout=True, gridspec_kw={'width_ratios': [3, 3]})

sns.barplot(x='Proj Dim', y='Test Loss', data=df_proj, ax=ax[0])
ax[0].set_title("Proj SSP")
sns.barplot(x='Proj Dim', y='Test Loss', data=df_st, ax=ax[1])
ax[1].set_title("Sub-Toroid SSP")

# fig.suptitle('Learning to Decode')

sns.despine()

plt.show()
