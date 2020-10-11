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

vels = [.2, .4]
npds = [15, 25]
threshs = [0.35, 0.4]

vels = [.15, .1]
npds = [15]
threshs = [0.4]

# vels = [0.6]
# npds = [15]

eps = 0.002

# attractor params
# vels = [1.0, 0.5, 0.125]
vels = [1.0, 0.5]
npds = [25, 50]
seeds = [1, 2, 3, 4, 5]
tpis = [1.5]

threshs = [0.4, 0.45]
vels = [0.4, 0.5, 0.25]
npds = [50]
seeds = [1, 2, 3, 4, 5]
tpis = [2.0]

threshs = [0.4, 0.45]
vels = [0.6, 0.5]
npds = [50]
seeds = [6, 7, 8, 9, 10]
tpis = [2.0]
seeds = [6, 7, 8, 9, 10, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
seeds = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

# threshs = [0.4, 0.45, 0.5]
# vels = [0.6, 0.5]
# npds = [50]
# seeds = [11, 12, 13]
# tpis = [2.5]


#spiking cleanup params
# threshs = [0.4, 0.45]
# vels = [0.25, 0.5]
# npds = [50]
# seeds = [11, 12, 13, 14, 15]
# tpis = [1.5]


# fname_base = 'output/kosslyn_ssp_cconv_{}seed_tpi{}_thresh{}_vel{}_npd{}.npy'
fname_base = 'output/attractor_kosslyn_ssp_cconv_{}seed_tpi{}_thresh{}_vel{}_npd{}.npy'
# fname_base = 'output/spiking_kosslyn_ssp_cconv_{}seed_tpi{}_thresh{}_vel{}_npd{}.npy'

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
