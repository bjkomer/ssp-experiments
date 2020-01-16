import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from spatial_semantic_pointers.utils import make_good_unitary, get_heatmap_vectors, encode_point, power, \
    get_heatmap_vectors_hex, encode_point_hex
import seaborn as sns

parser = argparse.ArgumentParser('averaged heatmaps for regular 2D and hex encoding')

parser.add_argument('--n-seeds', type=int, default=50)
parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--limit', type=float, default=5)
parser.add_argument('--res', type=int, default=256)
parser.add_argument('--palette', type=str, default='plasma', choices=['default', 'diverging', 'plasma'])

args = parser.parse_args()

if not os.path.exists('heatmap_output'):
    os.makedirs('heatmap_output')

fname = 'heatmap_output/{}dim_{}seeds.npz'.format(args.dim, args.n_seeds)

# square_axes = np.zeros((args.n_seeds, 2, args.dim))
# hex_axes = np.zeros((args.n_seeds, 3, args.dim))

axes = np.zeros((args.n_seeds, 3, args.dim))

square_heatmaps = np.zeros((args.n_seeds, args.res, args.res))
hex_heatmaps = np.zeros((args.n_seeds, args.res, args.res))

# Origin vector is the same in all cases
origin_vec = np.zeros((args.dim,))
origin_vec[0] = 1

xs = np.linspace(-args.limit, args.limit, args.res)
ys = np.linspace(-args.limit, args.limit, args.res)

# if the data already exists, just load it
if os.path.exists(fname):
    data = np.load(fname)
    square_heatmaps=data['square_heatmaps']
    hex_heatmaps=data['hex_heatmaps']
else:

    for seed in range(args.n_seeds):
        print("\x1b[2K\r Seed {} of {}".format(seed + 1, args.n_seeds), end="\r")
        rng = np.random.RandomState(seed=seed)
        X = make_good_unitary(args.dim, rng=rng)
        Y = make_good_unitary(args.dim, rng=rng)
        Z = make_good_unitary(args.dim, rng=rng)

        axes[seed, 0, :] = X.v
        axes[seed, 1, :] = Y.v
        axes[seed, 2, :] = Z.v

        square_heatmaps[seed, :, :] = np.tensordot(
            origin_vec,
            get_heatmap_vectors(xs=xs, ys=ys, x_axis_sp=X, y_axis_sp=Y),
            axes=([0], [2])
        )

        hex_heatmaps[seed, :, :] = np.tensordot(
            origin_vec,
            get_heatmap_vectors_hex(xs=xs, ys=ys, x_axis_sp=X, y_axis_sp=Y, z_axis_sp=Z),
            axes=([0], [2])
        )

    np.savez(
        fname,
        # square_axes=square_axes,
        # hex_axes=hex_axes,
        axes=axes,
        square_heatmaps=square_heatmaps,
        hex_heatmaps=hex_heatmaps,
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

avg_square_heatmap = square_heatmaps.mean(axis=0)
avg_hex_heatmap = hex_heatmaps.mean(axis=0)

plt.figure()
plt.imshow(square_heatmaps[0, :, :], vmin=vmin, vmax=vmax, cmap=cmap)
plt.title("Single Square Heatmap")
plt.figure()
plt.imshow(hex_heatmaps[0, :, :], vmin=vmin, vmax=vmax, cmap=cmap)
plt.title("Single Hexagonal Heatmap")

plt.figure()
plt.imshow(avg_square_heatmap, vmin=vmin, vmax=vmax, cmap=cmap)
plt.title("Square Heatmap")
plt.figure()
plt.imshow(avg_hex_heatmap, vmin=vmin, vmax=vmax, cmap=cmap)
plt.title("Hexagonal Heatmap")

# fig.savefig("fname.pdf", dpi=600, bbox_inches='tight')

plt.show()
