import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from spatial_semantic_pointers.utils import make_good_unitary, get_heatmap_vectors, encode_point, power, \
    get_heatmap_vectors_hex, encode_point_hex  #, get_heatmap_vectors_n
import seaborn as sns

parser = argparse.ArgumentParser('averaged heatmaps for regular 2D and ND encoding')

parser.add_argument('--n-seeds', type=int, default=25)
parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--limit', type=float, default=5)
parser.add_argument('--res', type=int, default=256)
parser.add_argument('--palette', type=str, default='plasma', choices=['default', 'diverging', 'plasma'])

# ns = [3, 4, 5, 8, 12, 24]
ns = [3, 4, 5, 8, 12, 24]

args = parser.parse_args()



def encode_point_n(x, y, axis_sps, x_axis, y_axis):
    """
    Encodes a given 2D point as a ND SSP
    """

    # 2D point represented as an N dimensional vector in the plane spanned by 'x_axis' and 'y_axis'

    vec = (x_axis * x + y_axis * y)

    # Generate the SSP from the high dimensional vector, by convolving all of the axis vector components together
    ret = power(axis_sps[0], vec[0])
    for i in range(1, len(axis_sps)):
        ret *= power(axis_sps[i], vec[i])

    return ret


def get_heatmap_vectors_n(xs, ys, n, seed=13, dim=512):
    """
    Precompute spatial semantic pointers for every location in the linspace
    Used to quickly compute heat maps by a simple vectorized dot product (matrix multiplication)
    """
    rng = np.random.RandomState(seed=seed)
    axis_sps = []
    for i in range(n):
        axis_sps.append(make_good_unitary(dim, rng=rng))

    vectors = np.zeros((len(xs), len(ys), dim))

    N = len(axis_sps)

    # points_nd = np.zeros((N + 1, N))
    # points_nd[:N, :] = np.eye(N)
    # # points in 2D that will correspond to each axis, plus one at zero
    # points_2d = np.zeros((N + 1, 2))

    points_nd = np.eye(N) * np.sqrt(N)
    # points in 2D that will correspond to each axis
    points_2d = np.zeros((N, 2))

    thetas = np.linspace(0, 2 * np.pi, N + 1)[:-1]
    # TODO: will want a scaling here, or along the high dim axes
    for i, theta in enumerate(thetas):
        points_2d[i, 0] = np.cos(theta)
        points_2d[i, 1] = np.sin(theta)

    transform_mat = np.linalg.lstsq(points_2d, points_nd)

    # apply scaling to the axes based on the singular values. Both should be the same
    x_axis = transform_mat[0][0, :] / transform_mat[3][0]
    y_axis = transform_mat[0][1, :] / transform_mat[3][1]

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            # Note: needed to divide by sqrt(2) in the scaling here to get it to match with the 2D and regular hex 3D method.
            p = encode_point_n(
                x=x*transform_mat[3][0]/np.sqrt(2), y=y*transform_mat[3][0]/np.sqrt(2), axis_sps=axis_sps, x_axis=x_axis, y_axis=y_axis
            )
            vectors[i, j, :] = p.v

    # also return the axis_sps so individual points can be generated
    return vectors, axis_sps




if not os.path.exists('heatmap_output'):
    os.makedirs('heatmap_output')

fname_regular = 'heatmap_output/{}dim_{}seeds.npz'.format(args.dim, args.n_seeds)

# square_axes = np.zeros((args.n_seeds, 2, args.dim))
# hex_axes = np.zeros((args.n_seeds, 3, args.dim))



square_heatmaps = np.zeros((args.n_seeds, args.res, args.res))
hex_heatmaps = np.zeros((args.n_seeds, args.res, args.res))

# Origin vector is the same in all cases
origin_vec = np.zeros((args.dim,))
origin_vec[0] = 1

xs = np.linspace(-args.limit, args.limit, args.res)
ys = np.linspace(-args.limit, args.limit, args.res)

avg_heatmaps = np.zeros((len(ns), args.res, args.res))


for ni, n in enumerate(ns):
    fname = 'heatmap_output/n{}_{}dim_{}seeds.npz'.format(n, args.dim, args.n_seeds)
    if os.path.exists(fname):
        data = np.load(fname)
        # avg_heatmaps[ni, :, :] = data['heatmaps'].mean(axis=0)
        avg_heatmaps[ni, :, :] = data['avg_heatmaps']
    else:
        heatmaps = np.zeros((args.n_seeds, args.res, args.res))
        axes = np.zeros((args.n_seeds, n, args.dim))
        for seed in range(args.n_seeds):
            print("\x1b[2K\r Seed {} of {}".format(seed + 1, args.n_seeds), end="\r")

            hmv, n_axes = get_heatmap_vectors_n(xs, ys, n, seed=seed, dim=args.dim)

            heatmaps[seed, :, :] = np.tensordot(
                origin_vec,
                hmv,
                axes=([0], [2])
            )

            for i, axis in enumerate(n_axes):
                axes[seed, i, :] = axis.v

        avg_heatmaps[ni, :, :] = heatmaps.mean(axis=0)
        np.savez(
            fname,
            # axes=axes,
            # heatmaps=heatmaps,
            avg_heatmaps=avg_heatmaps[ni, :, :],
        )



# if the data already exists, just load it
if os.path.exists(fname_regular):
    data = np.load(fname_regular)
    square_heatmaps=data['square_heatmaps']
    hex_heatmaps=data['hex_heatmaps']
else:
    axes = np.zeros((args.n_seeds, 3, args.dim))

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
        fname_regular,
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


def gaussian_2d(x, y, sigma, meshgrid):
    X, Y = meshgrid
    return np.exp(-((X - x) ** 2 + (Y - y) ** 2) / sigma / sigma)

meshgrid = np.meshgrid(xs, ys)

sigma = 1./np.sqrt(2)#0.5#.7
gauss = gaussian_2d(0, 0, sigma, meshgrid)

fig, ax = plt.subplots(len(ns) + 2, 2)

title_font_size = 16

ax[0, 0].imshow(
    avg_square_heatmap, vmin=vmin, vmax=vmax, cmap=cmap,
    origin='lower', interpolation='none', extent=(xs[0], xs[-1], ys[0], ys[-1])
)
ax[0, 0].set_title("Mean Square Heatmap", fontsize=title_font_size)

square_rmse = np.sqrt(np.mean((avg_square_heatmap - gauss)**2))
ax[0, 1].plot(avg_square_heatmap[args.res // 2, :])
ax[0, 1].plot(gauss[args.res // 2, :])
ax[0, 1].set_title("RMSE: {}".format(square_rmse), fontsize=title_font_size)

ax[1, 0].imshow(
    avg_hex_heatmap, vmin=vmin, vmax=vmax, cmap=cmap,
    origin='lower', interpolation='none', extent=(xs[0], xs[-1], ys[0], ys[-1])
)
ax[1, 0].set_title("Mean Hex Heatmap", fontsize=title_font_size)

hex_rmse = np.sqrt(np.mean((avg_hex_heatmap - gauss)**2))
ax[1, 1].plot(avg_hex_heatmap[args.res // 2, :])
ax[1, 1].plot(gauss[args.res // 2, :])
ax[1, 1].set_title("RMSE: {}".format(hex_rmse), fontsize=title_font_size)

for i, n in enumerate(ns):
    ax[i+2, 0].imshow(
        avg_heatmaps[i, :, :], vmin=vmin, vmax=vmax, cmap=cmap,
        origin='lower', interpolation='none', extent=(xs[0], xs[-1], ys[0], ys[-1])
    )
    ax[i+2, 0].set_title("Mean {}D Heatmap".format(n), fontsize=title_font_size)

    rmse = np.sqrt(np.mean((avg_heatmaps[i, :, :] - gauss) ** 2))
    ax[i+2, 1].plot(avg_heatmaps[i, args.res // 2, :])
    ax[i+2, 1].plot(gauss[args.res // 2, :])
    ax[i+2, 1].set_title("RMSE: {}".format(rmse), fontsize=title_font_size)

# fig, ax = plt.subplots(3, 3, figsize=(10, 9))
#
# title_font_size = 16
#
# im = ax[0, 0].imshow(
#     square_heatmaps[0, :, :], vmin=vmin, vmax=vmax, cmap=cmap,
#     origin='lower', interpolation='none', extent=(xs[0], xs[-1], ys[0], ys[-1])
# )
# ax[0, 0].set_title("Single Square Heatmap", fontsize=title_font_size)
#
# ax[0, 1].imshow(
#     hex_heatmaps[0, :, :], vmin=vmin, vmax=vmax, cmap=cmap,
#     origin='lower', interpolation='none', extent=(xs[0], xs[-1], ys[0], ys[-1])
# )
# ax[0, 1].set_title("Single Hexagonal Heatmap", fontsize=title_font_size)
#
# ax[1, 0].imshow(
#     avg_square_heatmap, vmin=vmin, vmax=vmax, cmap=cmap,
#     origin='lower', interpolation='none', extent=(xs[0], xs[-1], ys[0], ys[-1])
# )
# ax[1, 0].set_title("Mean Square Heatmap", fontsize=title_font_size)
#
# ax[1, 1].imshow(
#     avg_hex_heatmap, vmin=vmin, vmax=vmax, cmap=cmap,
#     origin='lower', interpolation='none', extent=(xs[0], xs[-1], ys[0], ys[-1])
# )
# ax[1, 1].set_title("Mean Hexagonal Heatmap", fontsize=title_font_size)
#
# ax[0, 2].imshow(
#     gauss, vmin=vmin, vmax=vmax, cmap=cmap,
#     origin='lower', interpolation='none', extent=(xs[0], xs[-1], ys[0], ys[-1])
# )
# ax[0, 2].set_title("Gaussian", fontsize=title_font_size)
#
# ax[1, 2].imshow(
#     avg_hex_heatmap - gauss, vmin=vmin, vmax=vmax, cmap=cmap,
#     origin='lower', interpolation='none', extent=(xs[0], xs[-1], ys[0], ys[-1])
# )
# ax[1, 2].set_title("Gaussian Difference", fontsize=title_font_size)
#
# square_rmse = np.sqrt(np.mean((avg_square_heatmap - gauss)**2))
# hex_rmse = np.sqrt(np.mean((avg_hex_heatmap - gauss) ** 2))
#
# # square_rmse = np.sqrt(np.mean((square_heatmaps[0, :, :] - gauss) ** 2))
# # hex_rmse = np.sqrt(np.mean((hex_heatmaps[0, :, :] - gauss) ** 2))
#
# # ax[2, 0].plot(square_heatmaps[0, args.res // 2, :])
# ax[2, 0].plot(avg_square_heatmap[args.res // 2, :])
# ax[2, 0].plot(gauss[args.res // 2, :])
# ax[2, 0].set_title("RMSE: {}".format(square_rmse), fontsize=title_font_size)
#
# # ax[2, 1].plot(hex_heatmaps[0, args.res // 2, :])
# ax[2, 1].plot(avg_hex_heatmap[args.res // 2, :])
# ax[2, 1].plot(gauss[args.res // 2, :])
# ax[2, 1].set_title("RMSE: {}".format(hex_rmse), fontsize=title_font_size)
#
# ax[2, 2].plot(gauss[args.res // 2, :])
#




plt.show()