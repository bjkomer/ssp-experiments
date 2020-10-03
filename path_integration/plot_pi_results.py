import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from statannot import add_stat_annotation

df = pd.DataFrame()

fname_cconv = 'eval_data_pi/cconv_seed{}_sub-toroid-ssp_d256.npz'

cconv_seeds = [10, 11, 12, 13, 14]

for seed in cconv_seeds:

    data = np.load(fname_cconv.format(seed))
    df = df.append(
        {
            'Encoding': 'CC SSP',
            'RMSE': data['rmse'],
            'Seed': seed
        },
        ignore_index=True,
    )

# fname_base = 'eval_data_pi/{}_normal_d256_100t.npz'
# fname_base = 'eval_data_pi/{}_nonnegative_d256.npz'
# fname_base = 'eval_data_pi/new_seeds/{}_normal_d256_seed{}_500epochs_5000samples_100t.npz'
fname_base = 'eval_data_pi/new_seeds_longer/{}_normal_d256_seed{}_1500epochs_7500samples_100t.npz'

seeds = [
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
]

enc_names = {
    'hex-ssp': 'Hex SSP',
    'pc-gauss': 'RBF',
    'legendre': 'Legendre',
    'one-hot': 'One Hot',
    'tile-coding': 'Tile Code',
    '2d': '2D',
}

encodings = [
    'hex-ssp',
    'pc-gauss',
    'legendre',
    'one-hot',
    'tile-coding',
    '2d',
]

for encoding in encodings:
    for seed in seeds:
        fname = fname_base.format(encoding, seed)
        try:
            data = np.load(fname)
            df = df.append(
                {
                    'Encoding': enc_names[encoding],
                    'RMSE': data['rmse'],
                    'Seed': seed
                },
                ignore_index=True,
            )
        except:
            print("could not find data for: " + fname)

fig, ax = plt.subplots(1, 1, figsize=(6, 4), tight_layout=True)

order = ['CC SSP', 'Hex SSP', 'RBF', 'Legendre', 'One Hot', 'Tile Code', '2D']

sns.barplot(data=df, x='Encoding', y='RMSE', ax=ax, order=order)

test_results = add_stat_annotation(
    ax, data=df, x='Encoding', y='RMSE', order=order,
    comparisons_correction=None,
    box_pairs=[("CC SSP", "Hex SSP"), ("Hex SSP", "RBF"), ("Tile Code", "One Hot"), ("Hex SSP", "2D"), ("2D", "RBF")],
    test='t-test_ind',
    text_format='star',
    loc='inside',
    verbose=2
)


sns.despine()
plt.show()
