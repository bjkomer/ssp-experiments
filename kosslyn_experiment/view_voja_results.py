import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

tpis = [2, 3, 5]
dims = [7, 13, 19, 25]
seeds = [1, 2, 3, 4, 5]
test_pairs = 200
eps = 0.002

fname_base = 'output/exp_data_kosslyn_{}dim_hex_tpi{}_seed{}.npy'

for tpi in tpis:
    for dim in dims:
        plt.figure()
        for seed in seeds:
            fname = fname_base.format(dim, tpi, seed)
            # order in data is:
            # (n_samples, 4)
            # (self.prev_item_index, self.item_index, elapsed_time, dist)
            data = np.load(fname)
            # remove corrupted trials
            inds = data[:, 2] > eps
            reaction_time = data[inds, 2]
            distance = data[inds, 3]
            plt.scatter(distance, reaction_time)
        plt.title("tpi: {}, dim: {}".format(tpi, dim))
plt.show()
