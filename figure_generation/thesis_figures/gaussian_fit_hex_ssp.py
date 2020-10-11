import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import numpy as np

gaussian_rmse = True
average = False#True

fname = '/home/ctnuser/ssp_navigation_sandbox/axis_vector_explorations/heatmap_output/256dim_32seeds.npz'

dim = 256
res = 256
limit = 5
n_seeds = 32

fontsize = 16

palette = 'default'


square_heatmaps = np.zeros((n_seeds, res, res))
hex_heatmaps = np.zeros((n_seeds, res, res))

# Origin vector is the same in all cases
origin_vec = np.zeros((dim,))
origin_vec[0] = 1

xs = np.linspace(-limit, limit, res)
ys = np.linspace(-limit, limit, res)


data = np.load(fname)
square_heatmaps=data['square_heatmaps']
hex_heatmaps=data['hex_heatmaps']



if palette == 'diverging':
    cmap = sns.diverging_palette(150, 275, s=80, l=55, as_cmap=True)
elif palette == 'plasma':
    cmap = 'plasma'
elif palette == 'default':
    cmap = None

# vmin = None
# vmax = None
vmin = -1
vmax = 1

avg_square_heatmap = square_heatmaps.mean(axis=0)
avg_hex_heatmap = hex_heatmaps.mean(axis=0)

if gaussian_rmse:

    def gaussian_2d(x, y, sigma, meshgrid):
        X, Y = meshgrid
        return np.exp(-((X - x) ** 2 + (Y - y) ** 2) / sigma / sigma)

    meshgrid = np.meshgrid(xs, ys)

    sigma = 1/np.sqrt(2)#.7
    # sigma = .7
    gauss = gaussian_2d(0, 0, sigma, meshgrid)

    fig, ax = plt.subplots(1, 4, figsize=(8, 2), tight_layout=True)

    title_font_size = 10#16

    square_rmse = np.sqrt(np.mean((square_heatmaps[0, :, :] - gauss) ** 2))
    hex_rmse = np.sqrt(np.mean((hex_heatmaps[0, :, :] - gauss) ** 2))

    ax[0].plot(xs, square_heatmaps[0, res // 2, :])
    ax[0].plot(xs, gauss[res // 2, :])
    ax[0].set_title("Single Square Axes\nRMSE: {}".format(np.round(square_rmse,4)), fontsize=title_font_size)
    ax[0].set_ylim([-0.3, 1.1])

    ax[1].plot(xs, hex_heatmaps[0, res // 2, :])
    ax[1].plot(xs, gauss[res // 2, :])
    ax[1].set_title("Single Hexagonal Axes\nRMSE: {}".format(np.round(hex_rmse, 4)), fontsize=title_font_size)
    ax[1].set_ylim([-0.3, 1.1])

    square_rmse = np.sqrt(np.mean((avg_square_heatmap - gauss) ** 2))
    hex_rmse = np.sqrt(np.mean((avg_hex_heatmap - gauss) ** 2))

    ax[2].plot(xs, avg_square_heatmap[res // 2, :])
    ax[2].plot(xs, gauss[res // 2, :])
    ax[2].set_title("Mean Square Axes\nRMSE: {}".format(np.round(square_rmse,4)), fontsize=title_font_size)
    ax[2].set_ylim([-0.3, 1.1])

    ax[3].plot(xs, avg_hex_heatmap[res // 2, :])
    ax[3].plot(xs, gauss[res // 2, :])
    ax[3].set_title("Mean Hexagonal Axes\nRMSE: {}".format(np.round(hex_rmse, 4)), fontsize=title_font_size)
    ax[3].set_ylim([-0.3, 1.1])

    sns.despine()

else:
    fig, ax = plt.subplots(2, 2, figsize=(10, 9))

    title_font_size = 16

    im = ax[0, 0].imshow(
        square_heatmaps[0, :, :], vmin=vmin, vmax=vmax, cmap=cmap,
        origin='lower', interpolation='none', extent=(xs[0], xs[-1], ys[0], ys[-1])
    )
    ax[0, 0].set_title("Single Square Heatmap", fontsize=title_font_size)

    ax[0, 1].imshow(
        hex_heatmaps[0, :, :], vmin=vmin, vmax=vmax, cmap=cmap,
        origin='lower', interpolation='none', extent=(xs[0], xs[-1], ys[0], ys[-1])
    )
    ax[0, 1].set_title("Single Hexagonal Heatmap", fontsize=title_font_size)


    ax[1, 0].imshow(
        avg_square_heatmap, vmin=vmin, vmax=vmax, cmap=cmap,
        origin='lower', interpolation='none', extent=(xs[0], xs[-1], ys[0], ys[-1])
    )
    ax[1, 0].set_title("Mean Square Heatmap", fontsize=title_font_size)

    ax[1, 1].imshow(
        avg_hex_heatmap, vmin=vmin, vmax=vmax, cmap=cmap,
        origin='lower', interpolation='none', extent=(xs[0], xs[-1], ys[0], ys[-1])
    )
    ax[1, 1].set_title("Mean Hexagonal Heatmap", fontsize=title_font_size)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    fig.savefig("averaged_heatmaps.pdf", dpi=600, bbox_inches='tight')



plt.show()