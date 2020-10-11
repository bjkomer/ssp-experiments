import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from stable_baselines import results_plotter
from stable_baselines.bench.monitor import load_results


def plot_data(
        xy_list,
        # xaxis,
        # title,
        # episode_window=100,
        episode_window=1000,
):

    plt.figure(figsize=(8, 2))
    maxx = max(xy[0][-1] for xy in xy_list)
    minx = 0
    for (i, (x, y)) in enumerate(xy_list):
        # color = COLORS[i]
        # plt.scatter(x, y, s=2)
        # Do not plot the smoothed curve at all if the timeseries is shorter than window size.
        if x.shape[0] >= episode_window:
            # Compute and plot rolling mean with window of size EPISODE_WINDOW
            x, y_mean = results_plotter.window_func(x, y, episode_window, np.mean)
            # plt.plot(x, y_mean, color=color)
            # divide by 10 to make reward scaling correct
            plt.plot(x, y_mean/10.)
            print(x.shape)
    plt.xlim(minx, maxx)
    # plt.title(title)
    # plt.xlabel(xaxis)
    plt.ylabel("Episode Rewards")
    plt.tight_layout()


# fnames = [
#     'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_10seed_5000000steps_0gd_100ms_scaling0.5/monitor.csv',
#     'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_11seed_5000000steps_0gd_100ms_scaling0.5/monitor.csv',
#     'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_12seed_5000000steps_0gd_100ms_scaling0.5/monitor.csv',
# ]
fnames = [
    'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_10seed_5000000steps_0gd_100ms_scaling0.5/',
    'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_11seed_5000000steps_0gd_100ms_scaling0.5/',
    'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_12seed_5000000steps_0gd_100ms_scaling0.5/',
]


# fnames = [
#     'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_10seed_5000000steps_0gd_100ms_scaling0.5/',
#     'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_11seed_5000000steps_0gd_100ms_scaling0.5/',
#     'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_12seed_5000000steps_0gd_100ms_scaling0.5/',
#
#     'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_13seed_5000000steps_0gd_1000ms_scaling0.5/',
#     'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_14seed_2500000steps_0gd_100ms_scaling0.5/',
#
#     'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_2seed_5000000steps_0gd_100ms_cur_scaling0.5/',
#     'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_3seed_5000000steps_0gd_100ms_cur_scaling0.5/',
#     'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_4seed_5000000steps_0gd_100ms_cur_scaling0.5/',
#
#     'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_13seed_2500000steps_0gd_100ms_cur_scaling0.5/',
# ]

fnames = [
    # 'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_10seed_5000000steps_0gd_100ms_scaling0.5/',
    # 'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_11seed_5000000steps_0gd_100ms_scaling0.5/',
    # 'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_12seed_5000000steps_0gd_100ms_scaling0.5/',

    'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_50seed_1000000steps_0gd_100ms_scaling0.5/',
    'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_51seed_1000000steps_0gd_100ms_scaling0.5/',
    'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_52seed_1000000steps_0gd_100ms_scaling0.5/',
]

fnames = [
    # 'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_10seed_5000000steps_0gd_100ms_scaling0.5/',
    'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_11seed_5000000steps_0gd_100ms_scaling0.5/',
    # 'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_12seed_5000000steps_0gd_100ms_scaling0.5/',
    #
    # # 'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_13seed_5000000steps_0gd_1000ms_scaling0.5/',
    # 'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_14seed_2500000steps_0gd_100ms_scaling0.5/',

    'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_2seed_5000000steps_0gd_100ms_cur_scaling0.5/',
    # 'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_3seed_5000000steps_0gd_100ms_cur_scaling0.5/',
    'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_4seed_5000000steps_0gd_100ms_cur_scaling0.5/',

    # 'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_13seed_2500000steps_0gd_100ms_cur_scaling0.5/',
]

# results_plotter.plot_results(fnames, 1e6, results_plotter.X_TIMESTEPS, "Results")

timesteps = load_results(fnames[0])

print(timesteps)
print(type(timesteps))

tslist = []
num_timesteps = 1e7
xaxis=results_plotter.X_TIMESTEPS
for folder in fnames:
    timesteps = load_results(folder)
    if num_timesteps is not None:
        timesteps = timesteps[timesteps.l.cumsum() <= num_timesteps]
    tslist.append(timesteps)
xy_list = [results_plotter.ts2xy(timesteps_item, xaxis) for timesteps_item in tslist]
plot_data(xy_list)

# for fname in fnames:

sns.despine()
plt.show()
