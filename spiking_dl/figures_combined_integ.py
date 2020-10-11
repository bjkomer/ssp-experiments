import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

blocks_indices = [0, 4, 6, 7, 8, 9]
maze_indices = [1, 2, 3, 5]

mazes = np.arange(10)
# seeds = np.arange(5)
seeds = np.arange(10)

# suc = np.zeros((10,))
#
# for i in range(10):
#     data = np.load('output/maze{}_seed13_spiking_data.npz'.format(i))
#     suc[i] = data['successes'].mean()
#
# print(np.mean(suc))

folder = 'output'
folder = 'output_256'
folder = 'output_512'
folder = 'output_512_noise'
folder = 'output_512_noise_more_trials'

fnames = [
    folder + '/maze{}_seed{}_spiking_data.npz',
    # folder + '/maze{}_seed{}_spiking_data_loc_gt.npz',
    # folder + '/maze{}_seed{}_spiking_data_cleanup_gt.npz',
    # folder + '/maze{}_seed{}_spiking_data_loc_gt_cleanup_gt.npz',
]

# for the overall percent plot with the four conditions
df = pd.DataFrame()

conditions = [
    'Spiking Network',
    'NN-Cleanup\nNN-Localization',
    'NN-Cleanup\nGT-Localization',
    'GT-Cleanup\nNN-Localization',
    'GT-Cleanup\nGT-Localization',
]

# per_condition = np.zeros((4, ))
for i, fname in enumerate(fnames):

    for mi in mazes:
        suc = 0
        for si in seeds:
            data = np.load(fname.format(mi, si))
            suc += data['successes'].mean()
            # df = df.append(
            #     {
            #         'Maze Index': mi,
            #         'Success Rate': data['successes'].mean(),
            #         'Condition': conditions[i],
            #     },
            #     ignore_index=True,
            # )
        df = df.append(
            {
                'Maze Index': mi,
                'Success Rate': suc / len(seeds),
                'Condition': conditions[i],
            },
            ignore_index=True,
        )

df = pd.concat([df, pd.read_csv('integ_results.csv')])

fig_overall, ax_overall = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)

# sns.barplot(data=df, x='Condition', y='Success Rate', ax=ax_overall)
sns.boxplot(data=df, x='Condition', y='Success Rate', ax=ax_overall)
sns.despine()

for i in range(5):
    df_tmp = df[df['Condition'] == conditions[i]]
    print(conditions[i])
    print("mean")
    print(df_tmp['Success Rate'].mean())
    print("std")
    print(df_tmp['Success Rate'].std())
    print("blocks mean")
    print(df_tmp[df_tmp['Maze Index'].isin(blocks_indices)]['Success Rate'].mean())
    print("maze mean")
    print(df_tmp[df_tmp['Maze Index'].isin(maze_indices)]['Success Rate'].mean())
    print("")

plt.show()
