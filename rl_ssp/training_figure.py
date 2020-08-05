import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

fnames = [
    'models_tensorflow/large_td3_256dim_2048x2_0sensors_10seed_5000000steps_0gd_100ms_scaling0.5.npz',
    'models_tensorflow/large_td3_256dim_2048x2_0sensors_11seed_5000000steps_0gd_100ms_scaling0.5.npz',
    'models_tensorflow/large_td3_256dim_2048x2_0sensors_12seed_5000000steps_0gd_100ms_scaling0.5.npz',
]

# fnames = [
#     # 'models_tensorflow/large_td3_256dim_2048x2_0sensors_13seed_2500000steps_0gd_100ms_cur_scaling0.5.npz',
#     'models_tensorflow/large_td3_256dim_2048x2_0sensors_2seed_5000000steps_0gd_100ms_cur_scaling0.5.npz',
#     'models_tensorflow/large_td3_256dim_2048x2_0sensors_3seed_5000000steps_0gd_100ms_cur_scaling0.5.npz',
#     'models_tensorflow/large_td3_256dim_2048x2_0sensors_4seed_5000000steps_0gd_100ms_cur_scaling0.5.npz',
# ]

fnames = [
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_70seed_500000steps_0gd_100ms_scaling0.5.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_71seed_500000steps_0gd_100ms_scaling0.5.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_72seed_500000steps_0gd_100ms_scaling0.5.npz',
]

fnames = [
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_70seed_500000steps_0gd_100ms_cur_scaling0.5.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_71seed_500000steps_0gd_100ms_cur_scaling0.5.npz',
    # 'models_tensorflow/medium_td3_256dim_2048x2_0sensors_72seed_500000steps_0gd_100ms_cur_scaling0.5.npz',
]

fnames = [
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_70seed_500000steps_0gd_100ms_scaling0.5.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_71seed_500000steps_0gd_100ms_scaling0.5.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_72seed_500000steps_0gd_100ms_scaling0.5.npz',

    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_70seed_500000steps_0gd_100ms_cur_scaling0.5.npz',
    'models_tensorflow/medium_td3_256dim_2048x2_0sensors_71seed_500000steps_0gd_100ms_cur_scaling0.5.npz',
    # 'models_tensorflow/medium_td3_256dim_2048x2_0sensors_72seed_500000steps_0gd_100ms_cur_scaling0.5.npz',
]

for fname in fnames:
    data = np.load(fname)['eval_data']
    plt.plot(data[:, 0])

plt.show()
