import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from stable_baselines import results_plotter


fnames = [
    'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_10seed_5000000steps_0gd_100ms_scaling0.5/monitor.csv',
    'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_11seed_5000000steps_0gd_100ms_scaling0.5/monitor.csv',
    'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_12seed_5000000steps_0gd_100ms_scaling0.5/monitor.csv',
]
fnames = [
    'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_10seed_5000000steps_0gd_100ms_scaling0.5/',
    'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_11seed_5000000steps_0gd_100ms_scaling0.5/',
    'logdir/models_tensorflow/large_td3_256dim_2048x2_0sensors_12seed_5000000steps_0gd_100ms_scaling0.5/',
]

results_plotter.plot_results(fnames, 1e6, results_plotter.X_TIMESTEPS, "Results")

# for fname in fnames:

plt.show()
