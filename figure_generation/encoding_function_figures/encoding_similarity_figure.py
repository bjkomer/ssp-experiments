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

args.n_tiles = 4
args.n_bins = 8
# args.pc_gauss_sigma = 0.75 * (16./13.)*2
args.pc_gauss_sigma = 1.25
args.dim = 256

vmin=None
vmax=None

limit = 8
outer_limit = 10

res = 128
xs = np.linspace(-outer_limit, outer_limit, res)
ys = np.linspace(-outer_limit, outer_limit, res)

encodings = ['hex-ssp', 'pc-gauss', 'tile-coding', 'legendre', 'random']
encoding_plot_names = {
    'hex-ssp': 'Hex SSP',
    'pc-gauss': 'RBF',
    'tile-coding': 'Tile Code',
    'legendre': 'Legendre',
    'random': 'Random',
}

locations = [
    [0, 0],
    [3.4, -1.6],
    [-4.6, -2.3],
    [-9, 5],
    [8.4, 9.2],
]

# cosine distance
fig, ax = plt.subplots(len(locations), len(encodings) + 1, figsize=(14, 14), tight_layout=True)
# vector distance
fig2, ax2 = plt.subplots(len(locations), len(encodings) + 1, figsize=(14, 14), tight_layout=True)

for ei, encoding in enumerate(encodings):
    print(encoding)
    args.spatial_encoding = encoding
    encoding_func, repr_dim = get_encoding_function(args, limit_low=-limit, limit_high=limit)

    heatmap_vectors = np.zeros((len(xs), len(ys), repr_dim))
    norm_heatmap_vectors = np.zeros((len(xs), len(ys), repr_dim))

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            heatmap_vectors[i, j, :] = encoding_func(x, y)
            norm_heatmap_vectors[i, j, :] = heatmap_vectors[i, j, :] / np.linalg.norm(heatmap_vectors[i, j, :])

    for li, loc in enumerate(locations):
        vec = encoding_func(loc[0], loc[1])

        sim = np.tensordot(vec, norm_heatmap_vectors, axes=([0], [2]))

        ax[li, ei+1].imshow(
            sim.T, vmin=vmin, vmax=vmax,
            origin='lower', interpolation='none', extent=(xs[0], xs[-1], ys[0], ys[-1])
        )

        dist = heatmap_vectors - vec
        dist = np.linalg.norm(dist, axis=2)

        ax2[li, ei + 1].imshow(
            (1 - dist).T, vmin=-1, vmax=1,
            origin='lower', interpolation='none', extent=(xs[0], xs[-1], ys[0], ys[-1])
        )

        if li == 0:
            ax[li, ei + 1].set_title(encoding_plot_names[encoding])
            ax2[li, ei + 1].set_title(encoding_plot_names[encoding])

        # ax2[li, ei + 1].imshow(-dist, vmin=vmin, vmax=vmax)
        # ax2[li, ei + 1].imshow(-np.log(1+dist), vmin=-10, vmax=vmax)

# show the points chosen and the bounds
for li, loc in enumerate(locations):
    ax[li, 0].scatter(loc[0], loc[1], color='g')
    ax2[li, 0].scatter(loc[0], loc[1], color='g')
    ax[li, 0].plot([limit, -limit, -limit, limit, limit], [-limit, -limit, limit, limit, -limit])
    ax2[li, 0].plot([limit, -limit, -limit, limit, limit], [-limit, -limit, limit, limit, -limit])
    ax[li, 0].set_xlim([-outer_limit, outer_limit])
    ax[li, 0].set_ylim([-outer_limit, outer_limit])
    ax2[li, 0].set_xlim([-outer_limit, outer_limit])
    ax2[li, 0].set_ylim([-outer_limit, outer_limit])
    ax[li, 0].set_aspect('equal')
    ax2[li, 0].set_aspect('equal')

ax[0, 0].set_title('Location')
ax2[0, 0].set_title('Location')


plt.show()
