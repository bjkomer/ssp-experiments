import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

df = pd.DataFrame()

fname_cconv = 'eval_data_pi/cconv_sub-toroid-ssp_d256.npz'

data = np.load(fname_cconv)
df = df.append(
    {
        'Encoding': 'CC SSP',
        'RMSE': data['rmse'],
    },
    ignore_index=True,
)


fname_base = 'eval_data_pi/{}_normal_d256.npz'

enc_names = {
    'hex-ssp': 'Hex SSP',
    'pc-gauss': 'RBF',
    'legendre': 'Legendre',
    'one-hot': 'One Hot',
    '2d': '2D',
}

encodings = [
    'hex-ssp',
    'pc-gauss',
    'legendre',
    'one-hot',
    '2d',
]

for encoding in encodings:
    fname = fname_base.format(encoding)
    data = np.load(fname)
    df = df.append(
        {
            'Encoding': enc_names[encoding],
            'RMSE': data['rmse'],
        },
        ignore_index=True,
    )

fig, ax = plt.subplots(1, 1, figsize=(6, 4), tight_layout=True)

sns.barplot(data=df, x='Encoding', y='RMSE', ax=ax)
sns.despine()
plt.show()
