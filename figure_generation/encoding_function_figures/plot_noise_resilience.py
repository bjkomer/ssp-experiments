import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

fname_csv = 'noise_resilience_larger_results_5seeds.csv'
df = pd.read_csv(fname_csv)
df = df.replace('Size', 'Environment Length')
df = df.replace('ssp', 'SSP')
df = df.replace('pc-gauss', 'RBF')
# df = df.replace('legendre', 'LG')
df = df.replace('legendre', 'Legendre')
df = df.replace('tile-coding', 'Tile-Code')
df = df.replace('one-hot', 'One-Hot')

# df = df[df['Encoding'] != 'OH']
# df = df[df['Encoding'] != 'TC']

# df = df[df['Noise Level'] == 0.1]
# df = df[df['Noise Level'] != 1.0]
fig, ax = plt.subplots(1, 3, figsize=(8, 3), tight_layout=True)
# noise_levels = [0.001, 0.005, 0.01, 0.05, 0.1]
noise_levels = [0.001, 0.01, 0.1]
for i, noise_level in enumerate(noise_levels):
    df_n = df[df['Noise Level'] == noise_level]
    sns.lineplot(data=df_n, x='Size', y='RMSE', hue='Encoding', ax=ax[i])
    ax[i].set_xscale('log')
    ax[i].set_title('\u03C3 = {}'.format(noise_level))
    ax[i].set_ylim([-.01, 1])
    if i != 0:
        ax[i].get_legend().remove()
sns.despine()

plt.show()