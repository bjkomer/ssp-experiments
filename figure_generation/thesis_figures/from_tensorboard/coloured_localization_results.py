import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from tensorflow.python.summary.summary_iterator import summary_iterator
import glob
from matplotlib.lines import Line2D
import pandas as pd

exp_folder = 'loc_colours'

base_folder = '/home/ctnuser/ssp-navigation/ssp_navigation/datasets/mixed_style_100mazes_100goals_64res_13size_13seed/coloured_10mazes_36sensors_360fov_100000samples/coloured_snapshot_network/{}/'.format(exp_folder)

exp_folder = 'final_loc_colours_longer'

base_folder = '/home/ctnuser/ssp-navigation/ssp_navigation/datasets/mixed_style_100mazes_100goals_64res_13size_13seed/coloured_10mazes_36sensors_360fov_250000samples/coloured_snapshot_network/{}/'.format(exp_folder)

n_mazes = [10]

seeds = [11, 12, 13, 14, 15]

encodings = [
    'sub-toroid-ssp',
    'hex-ssp',
    'ssp',
    'pc-gauss',
    'legendre',
    'tile-coding',
    'one-hot',
    # '2d-normalized',
    '2d',
    # 'random',
]

old_names = [
    'sub-toroid-ssp',
    'hex-ssp',
    'ssp',
    'pc-gauss',
    'legendre',
    'tile-coding',
    'one-hot',
    # '2d-normalized',
    '2d',
    # 'random',
]
order = [
    'ST SSP',
    'Hex SSP',
    'SSP',
    'RBF',
    'Legendre',
    'Tile-Code',
    'One-Hot',
    # '2D-Norm',
    '2D',
    # 'Random',
]

name_map = {}
for i in range(len(old_names)):
    name_map[old_names[i]] = order[i]

encoding_means = {
    'ssp': 1.3970652836340025,
    'hex-ssp': 1.3944702178240633,
    'sub-toroid-ssp': 1.3378701820439027,
    'one-hot': 1.40836112900908,
    'legendre': 3.0360506441542032,
    'pc-gauss': 1.9976824434302876,
    'tile-coding': 2.7940363268066943,
    '2d': 5.197245311201659,
}

tags = [
    'avg_cosine_loss',
    'avg_mse_loss',
    'avg_scaled_loss',
    'test_coord_rmse',
    'test_cosine_loss',
    'test_mse_loss',
]

rmse_file_base = '{}/dim256/scaling0.5/mazes10/hidsize2048/seed{}/*/*/rmse.npz'

# summary_path_base = '{}mazes/{}/seed{}/*/*/events*'
summary_path_base = '{}/dim256/scaling0.5/mazes10/hidsize2048/seed{}/*/*/events*'

use_npz_file = False
nmaze = 10

df = pd.DataFrame()

for seed in seeds:
    for encoding in encodings:
        if use_npz_file:
            print(base_folder + rmse_file_base.format(encoding, seed))
            fname = glob.glob(base_folder + rmse_file_base.format(encoding, seed))[0]
            data = np.load(fname)
            # using the mse_loss from the last epoch
            df = df.append(
                {
                    'Encoding': name_map[encoding],
                    'Seed': seed,
                    'Test RMSE': data['coord_rmse'],
                }, ignore_index=True
            )
        else:
            # print(summary_path)
            # print(base_folder + summary_path_base.format(nmaze, encoding, seed))
            # summary_path = base_folder + summary_path_base.format(nmaze, encoding, seed)
            # summary_path = base_folder + summary_path_base.format(encoding, seed)
            summary_path = glob.glob(base_folder + summary_path_base.format(encoding, seed))[0]
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
                        'Test NRMSE Loss': np.sqrt(mse_loss)/encoding_means[encoding],
                        'Test RMSE': coord_rmse,
                    }, ignore_index=True
                )

fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(6, 4))

# sns.barplot('Encoding', 'Test MSE Loss', hue='Mazes', data=df, order=order)
# sns.barplot('Encoding', 'Test RMSE', hue='Mazes', data=df, order=order)
# sns.barplot('Encoding', 'Test RMSE', data=df, order=order, ax=ax)
# sns.barplot('Encoding', 'Test MSE Loss', data=df, order=order, ax=ax)
sns.barplot('Encoding', 'Test NRMSE Loss', data=df, order=order, ax=ax)
sns.despine()

plt.title("Localization Results")

plt.show()
