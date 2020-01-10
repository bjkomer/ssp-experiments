import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from spatial_semantic_pointers.utils import make_good_unitary, get_heatmap_vectors, encode_point, power, \
    get_heatmap_vectors_hex, encode_point_hex, get_heatmap_vectors_n
import seaborn as sns

parser = argparse.ArgumentParser('averaged heatmaps for regular 2D and hex encoding')

parser.add_argument('--n-seeds', type=int, default=50)
parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--limit', type=float, default=5)
parser.add_argument('--res', type=int, default=256)
parser.add_argument('--palette', type=str, default='plasma', choices=['default', 'diverging', 'plasma'])
parser.add_argument('--n', type=int, default=3, help='projection dimension')

args = parser.parse_args()

if not os.path.exists('heatmap_output'):
    os.makedirs('heatmap_output')

fname = 'heatmap_output/n{}_{}dim_{}seeds.npz'.format(args.n, args.dim, args.n_seeds)

# square_axes = np.zeros((args.n_seeds, 2, args.dim))
# hex_axes = np.zeros((args.n_seeds, 3, args.dim))

axes = np.zeros((args.n_seeds, args.n, args.dim))

heatmaps = np.zeros((args.n_seeds, args.res, args.res))

# Origin vector is the same in all cases
origin_vec = np.zeros((args.dim,))
origin_vec[0] = 1

xs = np.linspace(-args.limit, args.limit, args.res)
ys = np.linspace(-args.limit, args.limit, args.res)

# if the data already exists, just load it
if os.path.exists(fname):
    data = np.load(fname)
    heatmaps=data['heatmaps']
else:

    for seed in range(args.n_seeds):
        print("\x1b[2K\r Seed {} of {}".format(seed + 1, args.n_seeds), end="\r")

        hmv, n_axes = get_heatmap_vectors_n(xs, ys, args.n, seed=seed, dim=args.dim)

        heatmaps[seed, :, :] = np.tensordot(
            origin_vec,
            hmv,
            axes=([0], [2])
        )

        for i, axis in enumerate(n_axes):
            axes[seed, i, :] = axis.v

    np.savez(
        fname,
        axes=axes,
        heatmaps=heatmaps,
    )

if args.palette == 'diverging':
    cmap = sns.diverging_palette(150, 275, s=80, l=55, as_cmap=True)
elif args.palette == 'plasma':
    cmap = 'plasma'
elif args.palette == 'default':
    cmap = None

# vmin = None
# vmax = None
vmin = -1
vmax = 1

avg_heatmap = heatmaps.mean(axis=0)

plt.figure()
plt.imshow(avg_heatmap, vmin=vmin, vmax=vmax, cmap=cmap)
plt.title("Heatmap")

plt.show()
