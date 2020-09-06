from ssp_navigation.utils.encodings import get_encoding_function, add_encoding_params
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

parser = argparse.ArgumentParser('')

parser = add_encoding_params(parser)

palette = sns.color_palette()

encodings = [
    'ssp',
    'hex-ssp',
    'sub-toroid-ssp',
    '2d',
    'one-hot',
    'legendre',
    'pc-gauss',
    'tile-coding'
]

# color = palette[5]

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

n_samples = 2500
rng = np.random.RandomState(seed=13)
locations = rng.uniform(-limit, limit, size=(n_samples, 2))
dists = np.zeros((n_samples, n_samples - 1))

avg_dists = np.zeros(len(encodings))

for ei, e in enumerate(encodings):
    print(e)

    args = parser.parse_args([])  # grab the defaults
    # overwrite specific values
    args.spatial_encoding = e
    args.limit = limit
    args.dim = 256
    args.n_tiles = 4
    args.n_bins = 8
    args.hilbert_points = 1

    encoding_func, repr_dim = get_encoding_function(args, limit_low=-limit, limit_high=limit)

    enc_values = np.zeros((n_samples, repr_dim))

    for i in range(n_samples):
        enc_values[i, :] = encoding_func(locations[i, 0], locations[i, 1])

    # distances of random points
    for i in range(n_samples):
        oi = 0
        for j in range(n_samples-1):
            if i == j:
                oi = 1
            dists[i, j] = np.linalg.norm(enc_values[i, :] - enc_values[j+oi, :])

    avg_dists[ei] = np.mean(dists.flatten())
    print(avg_dists[ei])

plt.show()
