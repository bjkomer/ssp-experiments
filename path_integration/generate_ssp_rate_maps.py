# Create a 'rate map' of the elements of the SSP itself
import numpy as np
import argparse
from spatial_semantic_pointers.utils import make_good_unitary, encode_point
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser('Run 2D supervised path integration experiment using pytorch')

parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--dim', type=int, default=512)
parser.add_argument('--res', type=int, default=256)
parser.add_argument('--limit', type=int, default=5)

args = parser.parse_args()

folder = 'images/ssp_rate_maps/dim{}_limit{}_seed{}'.format(args.dim, args.limit, args.seed)

if not os.path.exists(folder):
    os.makedirs(folder)

rng = np.random.RandomState(seed=args.seed)

x_axis_sp = make_good_unitary(dim=args.dim, rng=rng)
y_axis_sp = make_good_unitary(dim=args.dim, rng=rng)

xs = np.linspace(-args.limit, args.limit, args.res)
ys = np.linspace(-args.limit, args.limit, args.res)

activations = np.zeros((args.dim, args.res, args.res))

for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        activations[:, i, j] = encode_point(x, y, x_axis_sp, y_axis_sp).v


for n in range(args.dim):
    print("Dimension {} of {}".format(n + 1, args.dim))

    fig, ax = plt.subplots()
    ax.imshow(activations[n, :, :])
    fig.savefig("{}/dimension_{}".format(folder, n))