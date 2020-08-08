import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# reward shaping and curriculum learning
cur_fnames = [
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_70seed_500000steps_0gd_100ms_cur_scaling0.5.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_71seed_500000steps_0gd_100ms_cur_scaling0.5.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_72seed_500000steps_0gd_100ms_cur_scaling0.5.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_73seed_500000steps_0gd_100ms_cur_scaling0.5.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_74seed_500000steps_0gd_100ms_cur_scaling0.5.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_75seed_500000steps_0gd_100ms_cur_scaling0.5.npz',

    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_80seed_500000steps_0gd_100ms_cur_scaling0.5.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_81seed_500000steps_0gd_100ms_cur_scaling0.5.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_82seed_500000steps_0gd_100ms_cur_scaling0.5.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_83seed_500000steps_0gd_100ms_cur_scaling0.5.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_84seed_500000steps_0gd_100ms_cur_scaling0.5.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_85seed_500000steps_0gd_100ms_cur_scaling0.5.npz',

    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_120seed_500000steps_0gd_100ms_cur_scaling0.5.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_121seed_500000steps_0gd_100ms_cur_scaling0.5.npz',
]

# no reward shaping and curriculum learning
cur_norew_fnames = [
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_100seed_500000steps_0gd_100ms_cur_scaling0.5_nopsrew.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_101seed_500000steps_0gd_100ms_cur_scaling0.5_nopsrew.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_102seed_500000steps_0gd_100ms_cur_scaling0.5_nopsrew.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_103seed_500000steps_0gd_100ms_cur_scaling0.5_nopsrew.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_104seed_500000steps_0gd_100ms_cur_scaling0.5_nopsrew.npz',

    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_110seed_500000steps_0gd_100ms_cur_scaling0.5_nopsrew.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_111seed_500000steps_0gd_100ms_cur_scaling0.5_nopsrew.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_112seed_500000steps_0gd_100ms_cur_scaling0.5_nopsrew.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_113seed_500000steps_0gd_100ms_cur_scaling0.5_nopsrew.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_114seed_500000steps_0gd_100ms_cur_scaling0.5_nopsrew.npz',

    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_120seed_500000steps_0gd_100ms_cur_scaling0.5_nopsrew.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_121seed_500000steps_0gd_100ms_cur_scaling0.5_nopsrew.npz',
]

# no reward shaping
norew_fnames = [
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_100seed_500000steps_0gd_100ms_scaling0.5_nopsrew.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_101seed_500000steps_0gd_100ms_scaling0.5_nopsrew.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_102seed_500000steps_0gd_100ms_scaling0.5_nopsrew.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_103seed_500000steps_0gd_100ms_scaling0.5_nopsrew.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_104seed_500000steps_0gd_100ms_scaling0.5_nopsrew.npz',

    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_110seed_500000steps_0gd_100ms_scaling0.5_nopsrew.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_111seed_500000steps_0gd_100ms_scaling0.5_nopsrew.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_112seed_500000steps_0gd_100ms_scaling0.5_nopsrew.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_113seed_500000steps_0gd_100ms_scaling0.5_nopsrew.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_114seed_500000steps_0gd_100ms_scaling0.5_nopsrew.npz',

    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_120seed_500000steps_0gd_100ms_scaling0.5_nopsrew.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_121seed_500000steps_0gd_100ms_scaling0.5_nopsrew.npz',
]

# reward shaping
fnames = [
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_70seed_500000steps_0gd_100ms_scaling0.5.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_71seed_500000steps_0gd_100ms_scaling0.5.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_72seed_500000steps_0gd_100ms_scaling0.5.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_73seed_500000steps_0gd_100ms_scaling0.5.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_74seed_500000steps_0gd_100ms_scaling0.5.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_75seed_500000steps_0gd_100ms_scaling0.5.npz',

    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_80seed_500000steps_0gd_100ms_scaling0.5.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_81seed_500000steps_0gd_100ms_scaling0.5.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_82seed_500000steps_0gd_100ms_scaling0.5.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_83seed_500000steps_0gd_100ms_scaling0.5.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_84seed_500000steps_0gd_100ms_scaling0.5.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_85seed_500000steps_0gd_100ms_scaling0.5.npz',

    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_120seed_500000steps_0gd_100ms_scaling0.5.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_121seed_500000steps_0gd_100ms_scaling0.5.npz',
]

df = pd.DataFrame()

conditions = [
    'Standard',
    'RS',
    'CL',
    'RS + CL'
]

# conditions = [
#     'Standard',
#     'Reward Shaping',
#     'Curriculum Learning',
# ]

steps_per_eval = 10000

for i, fname_list in enumerate([norew_fnames, fnames, cur_norew_fnames, cur_fnames]):
    for fname in fname_list:
        data = np.load(fname)['eval_data']
        for j in range(data.shape[0]):
            df = df.append(
                {
                    'Condition': conditions[i],
                    'Training Steps': j*steps_per_eval,
                    'Average Return': data[j, 0]/10.,
                },
                ignore_index=True
            )

fig, ax = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)

sns.lineplot(data=df, x="Training Steps", y="Average Return", hue="Condition", ax=ax)
sns.despine()
plt.show()

