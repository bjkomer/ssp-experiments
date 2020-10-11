import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from tensorflow.python.summary.summary_iterator import summary_iterator
import glob
from matplotlib.lines import Line2D
import pandas as pd

exp_folder = 'final_loc'
# exp_folder = 'final_loc_tiled'

base_folder = '/home/ctnuser/ssp-navigation/ssp_navigation/datasets/mixed_style_100mazes_100goals_64res_13size_13seed//{}/'.format(exp_folder)

# summary_paths = {
#     'mse': glob.glob('ssp_cleanup_thesis/epochs250/mse/*/*/events*'),
#     'cosine': glob.glob('ssp_cleanup_thesis/epochs250/cosine/*/*/events*'),
# }

# n_mazes = [1, 10, 25]
# n_mazes = [10, 25]
n_mazes = [25]
n_mazes = [10]

seeds = [1, 2, 3, 4, 5]

encodings = [
    'pc-gauss',
    'ssp',
]

old_names = ['hex-ssp', 'ssp', 'pc-gauss', 'tile-coding', 'one-hot', 'random']
order = ['Hex SSP', 'SSP', 'RBF', 'Tile-Code', 'One-Hot', 'Random']

name_map = {}
for i in range(len(old_names)):
    name_map[old_names[i]] = order[i]

tags = [
    'avg_cosine_loss',
    'avg_mse_loss',
    'avg_scaled_loss',
    'test_coord_rmse',
    'test_cosine_loss',
    'test_mse_loss',
]

summary_path_base = '{}mazes/{}/seed{}/*/*/events*'

df = pd.DataFrame()

for nmaze in n_mazes:
    for seed in seeds:
        for encoding in encodings:
            summary_path = glob.glob(base_folder + summary_path_base.format(nmaze, encoding, seed))[0]
            # print(summary_path)
            # print(base_folder + summary_path_base.format(nmaze, encoding, seed))
            mse_loss_list = []
            cosine_loss_list = []
            mse_loss = None
            try:
                for e in summary_iterator(summary_path):
                    for v in e.summary.value:
                        if v.tag == 'test_mse_loss':
                            mse_loss = v.simple_value
                        if v.tag == 'test_coord_rmse':
                            coord_rmse = v.simple_value
                            # mse_loss_list.append(v.simple_value)
                        # elif v.tag == 'avg_cosine_loss':
                        #     cosine_loss_list.append(v.simple_value)

                # mse_loss[i, j, :] = np.array(mse_loss_list)
                # cosine_loss[i, j, :] = np.array(cosine_loss_list)
            except:
                print("failed to read record")
                print(summary_path)

            if mse_loss is not None:
                # using the mse_loss from the last epoch
                df = df.append(
                    {
                        'Encoding': name_map[encoding],
                        'Seed': seed,
                        'Mazes': nmaze,
                        'Test MSE Loss': mse_loss,
                        'Test RMSE': coord_rmse,
                    }, ignore_index=True
                )

fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(6, 4))

# sns.barplot('Encoding', 'Test MSE Loss', hue='Mazes', data=df, order=order)
# sns.barplot('Encoding', 'Test RMSE', hue='Mazes', data=df, order=order)
sns.barplot('Encoding', 'Test RMSE', data=df, order=order, ax=ax)
sns.despine()

plt.title("Localization Results")

plt.show()
