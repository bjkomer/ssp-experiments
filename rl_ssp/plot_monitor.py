from stable_baselines import results_plotter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

logdir = sys.argv[1]

# Helper from the library
results_plotter.plot_results([logdir], 1e5, results_plotter.X_TIMESTEPS, "Results")

plt.show()
