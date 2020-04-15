from ssp_navigation.utils.encodings import get_encoding_function, add_encoding_params
from scipy.spatial.distance import cosine
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

parser = argparse.ArgumentParser('')

parser = add_encoding_params(parser)

encodings = [
    'ssp', 'hex-ssp', #'periodic-hex-ssp', 'grid-ssp', 'ind-ssp',
    # 'random', '2d',
    # '2d-normalized',
    'one-hot',
    # 'hex-trig',
    #'trig', 'random-trig', 'random-rotated-trig', 'random-proj',
    # 'legendre',
    'pc-gauss',
    # 'pc-dog',
    'tile-coding'
]

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

    zero = encoding_func(0, 0)
    for i, x in enumerate(xs):

        pt = encoding_func(x, 0)

        dists[ei, i] = np.linalg.norm(zero - pt)
        cosine_dists[ei, i] = cosine(zero, pt)

    # scale tile-coding so that it can be viewed relative to the other encodings
    if e == 'tile-coding':
        dists[ei, :] /= np.sqrt(args.n_tiles)

legend = []
for e in encodings:
    if e in labels.keys():
        legend.append(labels[e])
    else:
        legend.append(e)

fig, ax = plt.subplots(1, 2, figsize=(8, 4), tight_layout=True)

ax[1].plot(xs, dists.T)
ax[1].set_title("Euclidean Distances")
# ax[1].legend(legend)
ax[1].set_xlabel('Position')
ax[1].set_ylabel('Distance')

ax[0].plot(xs, cosine_dists.T)
ax[0].set_title("Cosine Distances")
ax[0].legend(legend)
ax[0].set_xlabel('Position')
ax[0].set_ylabel('Distance')

sns.despine()

plt.show()