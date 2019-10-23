from ssp_navigation.utils.encodings import get_encoding_function, encoding_func_from_model
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(
    'Create figure showing the heatmap similarity of points encoded with different methods'
)

# parser.add_argument('--spatial-encoding', type=str, default='ssp',
#                     choices=[
#                         'ssp', 'random', '2d', '2d-normalized', 'one-hot', 'hex-trig',
#                         'trig', 'random-trig', 'random-proj', 'learned', 'frozen-learned',
#                         'pc-gauss', 'tile-coding'
#                     ],
#                     help='coordinate encoding for agent location and goal')
parser.add_argument('--hex-freq-coef', type=float, default=2.5, help='constant to scale frequencies by for hex-trig')
parser.add_argument('--pc-gauss-sigma', type=float, default=0.25, help='sigma for the gaussians')
parser.add_argument('--n-tiles', type=int, default=8, help='number of layers for tile coding')
parser.add_argument('--n-bins', type=int, default=8, help='number of bins for tile coding')
parser.add_argument('--res', type=int, default=128, help='resolution of the heatmap')
parser.add_argument('--maze-id-type', type=str, choices=['ssp', 'one-hot', 'random-sp'], default='one-hot',
                    help='ssp: region corresponding to maze layout.'
                         'one-hot: each maze given a one-hot vector.'
                         'random-sp: each maze given a unique random SP as an ID')
parser.add_argument('--seed', type=int, default=13, help='Seed for training and generating axis SSPs')
parser.add_argument('--dim', type=int, default=256, help='Dimensionality of the SSPs')
parser.add_argument('--hidden-size', type=int, default=512, help='Size of the hidden layer in the model')
parser.add_argument('--n-hidden-layers', type=int, default=1, help='Number of hidden layers in the model')
parser.add_argument('--model', type=str, default='', help='Saved model to use')

parser.add_argument('--limit-low', type=float, default=-5.0, help='lower limit for heatmap')
parser.add_argument('--limit-high', type=float, default=5.0, help='upper limit for heatmap')
parser.add_argument('--train-limit-low', type=float, default=-5.0, help='lower limit for defining encoding')
parser.add_argument('--train-limit-high', type=float, default=5.0, help='upper limit for defining encoding')
# parser.add_argument('--x-pos', type=float, default=0.0, help='x-position of the test point')
# parser.add_argument('--y-pos', type=float, default=0.0, help='y-position of the test point')

parser.add_argument('--n-mazes', type=int, default=10)

parser.add_argument('--n-points', type=int, default=3, help='Number of points to view')

parser.add_argument('--noise', type=float, default=0.1, help='gaussian noise to add to the representation')

parser.add_argument('--fname', default='heatmap_decoding.npz', help='file name to save results to for quick reloading')

args = parser.parse_args()

encodings = [
    'ssp',
    'one-hot',
    'pc-gauss',
    'tile-coding',
    # 'hex-trig',
    # 'random-proj',
    # 'random',
]

xs = np.linspace(args.limit_low, args.limit_high, args.res)
ys = np.linspace(args.limit_low, args.limit_high, args.res)

if os.path.isfile(args.fname):
    print("Loading Data from File")
    data = np.load(args.fname)
    heatmaps = data['heatmaps']
    normalized_heatmaps = data['normalized_heatmaps']
else:
    print("Generating Data")
    heatmaps = np.zeros((len(encodings), args.n_points, args.res, args.res))
    normalized_heatmaps = np.zeros((len(encodings), args.n_points, args.res, args.res))

    points = np.random.uniform(low=args.limit_low, high=args.limit_high, size=(args.n_points, 2))

    for ei, encoding in enumerate(encodings):
        args.spatial_encoding = encoding
        encoding_func, repr_dim = get_encoding_function(
            args, limit_low=args.train_limit_low, limit_high=args.train_limit_high
        )

        # # input is maze, loc, goal ssps, output is 2D direction to move
        # if 'learned' in args.spatial_encoding:
        #     enc_func = encoding_func_from_model(args.model, args.dim)
        #
        #
        #     def encoding_func(x, y):
        #         return enc_func(np.array([x, y]))
        # else:
        #     pass

        activations = np.zeros((args.res, args.res, args.dim))
        normalized_activations = np.zeros((args.res, args.res, args.dim))

        for ix, x in enumerate(xs):
            for iy, y in enumerate(ys):
                activations[ix, iy, :] = encoding_func(x, y)
                normalized_activations[ix, iy, :] = activations[ix, iy, :] / np.linalg.norm(activations[ix, iy, :])

        for pi in range(args.n_points):
            encoded_point = encoding_func(points[pi, 0], points[pi, 1])

            # Add optional noise to the encoding
            encoded_point = encoded_point + np.random.normal(loc=0, scale=args.noise, size=(args.dim,))

            heatmaps[ei, pi, :, :] = np.tensordot(encoded_point, activations, axes=([0], [2]))
            normalized_heatmaps[ei, pi, :, :] = np.tensordot(encoded_point, normalized_activations, axes=([0], [2]))

    np.savez(
        args.fname,
        heatmaps=heatmaps,
        normalized_heatmaps=normalized_heatmaps,
    )

# Plot the results

fig, ax = plt.subplots(args.n_points, len(encodings))

for ei in range(len(encodings)):
    for pi in range(args.n_points):
        ax[pi, ei].imshow(normalized_heatmaps[ei, pi, :, :])


fig2, ax2 = plt.subplots(args.n_points, len(encodings))

for ei in range(len(encodings)):
    for pi in range(args.n_points):
        ax2[pi, ei].imshow(heatmaps[ei, pi, :, :])

plt.show()
