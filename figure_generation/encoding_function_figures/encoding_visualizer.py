from ssp_navigation.utils.encodings import get_encoding_function, add_encoding_params
from spatial_semantic_pointers.utils import ssp_to_loc_v, ssp_to_loc
import argparse
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


parser = argparse.ArgumentParser()
parser = add_encoding_params(parser)
args = parser.parse_args()

vmin=None
vmax=None
# vmin=.3

res = 64
limit = 15
xs = np.linspace(-limit, limit, res)
ys = np.linspace(-limit, limit, res)

encoding_func, repr_dim = get_encoding_function(args, limit_low=-limit, limit_high=limit)

heatmap_vectors = np.zeros((len(xs), len(ys), repr_dim))

for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        heatmap_vectors[i, j, :] = encoding_func(x, y)

plt.imshow(heatmap_vectors[:, :, 0], vmin=vmin, vmax=vmax)
# plt.imshow(heatmap_vectors[:, :, 44], vmin=0, vmax=1)
plt.colorbar()
plt.show()
