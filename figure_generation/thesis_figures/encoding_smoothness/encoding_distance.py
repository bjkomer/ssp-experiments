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
    #'trig', 'random-trig', 'random-rotated-trig', 'random-proj', 'legendre',
    'pc-gauss',
    # 'pc-dog',
    # 'tile-coding'
]

limit = 5
res = 256
xs = np.linspace(-5, 5, res)



# distances from 0 to the current point, for each encoding
dists = np.zeros((len(encodings), res))
cosine_dists = np.zeros((len(encodings), res))

for ei, e in enumerate(encodings):
    print(e)

    args = parser.parse_args([])  # grab the defaults
    # overwrite specific values
    args.spatial_encoding = e
    args.limit = limit
    args.dim = 512#256
    args.n_tiles = 8
    args.n_bins = 8

    encoding_func, repr_dim = get_encoding_function(args, limit_low=-limit, limit_high=limit)

    zero = encoding_func(0, 0)
    for i, x in enumerate(xs):

        pt = encoding_func(x, 0)

        dists[ei, i] = np.linalg.norm(zero - pt)
        cosine_dists[ei, i] = cosine(zero, pt)

plt.figure()
plt.plot(xs, dists.T)
plt.title("Euclidean Distances")
plt.legend(encodings)
plt.figure()
plt.plot(xs, cosine_dists.T)
plt.title("Cosine Distances")
plt.legend(encodings)

plt.show()
