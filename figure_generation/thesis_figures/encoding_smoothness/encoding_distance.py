from ssp_navigation.utils.encodings import get_encoding_function, add_encoding_params
from scipy.spatial.distance import cosine
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

parser = argparse.ArgumentParser('')

parser = add_encoding_params(parser)

palette = sns.color_palette()

encodings = [
    'ssp', 'hex-ssp', #'periodic-hex-ssp', 'grid-ssp', 'ind-ssp',
    # 'random', '2d',
    # '2d-normalized',
    # 'hex-trig',
    #'trig', 'random-trig', 'random-rotated-trig', 'random-proj',
    # 'legendre',
    'pc-gauss',
    # 'pc-dog',
    'one-hot',
    'tile-coding',
    'legendre'
]

# encodings = ['ssp']
# color = palette[0]
# encodings = ['hex-ssp']
# color = palette[1]
# encodings = ['legendre']
# color = palette[5]
# encodings = ['one-hot']
# color = palette[3]
# encodings = ['pc-gauss']
# color = palette[2]
# encodings = ['tile-coding']
# color = palette[4]
# encodings = ['sub-toroid-ssp']
# color = palette[6]

labels = {
    'ssp': 'SSP',
    'hex-ssp': 'Hex SSP',
    'one-hot': 'One-Hot',
    'tile-coding': 'Tile-Code',
    'pc-gauss': 'RBF',
    'random': 'Random',
    'legendre': 'Legendre',
}

limit = 5
res = 512#256
xs = np.linspace(-limit, limit, res)


# distances from 0 to the current point, for each encoding
dists = np.zeros((len(encodings), res))
cosine_dists = np.zeros((len(encodings), res))

for ei, e in enumerate(encodings):
    print(e)

    args = parser.parse_args([])  # grab the defaults
    # overwrite specific values
    args.spatial_encoding = e
    args.limit = limit
    args.dim = 256#512#256
    args.n_tiles = 4#8
    args.n_bins = 8
    args.hilbert_points = 1#0

    encoding_func, repr_dim = get_encoding_function(args, limit_low=-limit, limit_high=limit)

    # x_offset = -187.2
    # y_offset = 32.2
    x_offset = 0
    y_offset = 0
    # x_offset = 1.2
    # y_offset = -3.4
    zero = encoding_func(x_offset, y_offset)
    for i, x in enumerate(xs):

        pt = encoding_func(x+x_offset, y_offset)

        dists[ei, i] = np.linalg.norm(zero - pt)
        cosine_dists[ei, i] = cosine(zero, pt)

    # scale tile-coding so that it can be viewed relative to the other encodings
    # if e == 'tile-coding':
    #     dists[ei, :] /= np.sqrt(args.n_tiles)
    # if e == 'legendre':
    #     dists[ei, :] /= 5

legend = []
for e in encodings:
    if e in labels.keys():
        legend.append(labels[e])
    else:
        legend.append(e)

fig, ax = plt.subplots(1, 2, figsize=(8, 4), tight_layout=True)

# ax[1].plot(xs, dists.T)
ax[1].plot(xs, dists.T, color=color)
ax[1].set_title("Euclidean Distances")
# ax[1].legend(legend)
# ax[1].set_xlabel('Position')
# ax[1].set_ylabel('Distance')
# ax[1].set_xlabel('Input Distance from (1.2, -3.4)')
ax[1].set_xlabel('Input Distance from (0.0, 0.0)')
ax[1].set_ylabel('Encoding Distance')
# ax[1].set_ylim([-.1, 2])
# ax[1].set_ylim([-.1, 10])

# ax[0].plot(xs, cosine_dists.T)
ax[0].plot(xs, cosine_dists.T, color=color)
ax[0].set_title("Cosine Distances")
ax[0].legend(legend)
# ax[0].set_xlabel('Position')
# ax[0].set_ylabel('Distance')
# ax[0].set_xlabel('Input Distance from (1.2, -3.4)')
ax[0].set_xlabel('Input Distance from (0.0, 0.0)')
ax[0].set_ylabel('Encoding Distance')

sns.despine()

plt.show()
