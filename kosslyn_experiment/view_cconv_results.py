import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

tpis = [1.0]
threshs = [0.35, 0.4, 0.45, 0.5]
seeds = [1, 2, 3, 4, 5]
vels = [0.6, 0.8, 1.0]
npds = [5, 10, 15]

vels = [0.6]
npds = [15]

eps = 0.002

fname_base = 'output/kosslyn_ssp_cconv_{}seed_tpi{}_thresh{}_vel{}_npd{}.npy'

for tpi in tpis:
    for thresh in threshs:
        for vel in vels:
            for npd in npds:
                plt.figure()
                for seed in seeds:
                    fname = fname_base.format(seed, tpi, thresh, vel, npd)
                    if not os.path.exists(fname):
                        print("could not find: {}".format(fname))
                        continue
                    # order in data is:
                    # (n_samples, 4)
                    # (self.prev_item_index, self.item_index, elapsed_time, dist)
                    data = np.load(fname)
                    # remove corrupted trials
                    inds = data[:, 2] > eps
                    reaction_time = data[inds, 2]
                    distance = data[inds, 3]
                    plt.scatter(distance, reaction_time)
                plt.title(fname)
plt.show()
