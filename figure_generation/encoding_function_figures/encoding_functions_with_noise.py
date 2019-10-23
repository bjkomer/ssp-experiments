import numpy as np
from ssp_navigation.utils.encodings import get_encoding_function, encoding_func_from_model
import numpy as np
from spatial_semantic_pointers.plots import plot_predictions, plot_predictions_v
from spatial_semantic_pointers.utils import ssp_to_loc_v, encode_point, make_good_unitary
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from functools import partial
import os
import argparse


parser = argparse.ArgumentParser('Evaluate encoding methods under various levels of noise')

parser.add_argument('--dim', type=int, default=512)
parser.add_argument('--res', type=int, default=64)
# parser.add_argument('--ssp-scaling', type=float, default=1.0)
parser.add_argument('--limit-low', type=float, default=-5)
parser.add_argument('--limit-high', type=float, default=5)
parser.add_argument('--train-limit-low', type=float, default=-5)
parser.add_argument('--train-limit-high', type=float, default=5)

# parser.add_argument('--pc-gauss-sigma', type=float, default=0.25)
# parser.add_argument('--frozen-model-path', type=str, default='frozen_models/blocks_10_100_model.pt')
# parser.add_argument('--seed', type=int, default=13)

parser.add_argument('--n-samples', type=int, default=5000)


args = parser.parse_args()

fname_csv = 'output/evaluate_encodings_noise_multiseed.csv'

seeds = np.arange(5)

ssp_configs = [
    argparse.Namespace(
        spatial_encoding='ssp', dim=512, seed=seed, limit_low=-5, limit_high=5
    ) for seed in seeds
]

pcgauss_configs = [
    argparse.Namespace(
        spatial_encoding='pc-gauss', dim=512, seed=seed, limit_low=-5, limit_high=5,
        pc_gauss_sigma=0.75,
    ) for seed in seeds
]

# Seed has no effect on one-hot encoding
onehot_configs = [
    argparse.Namespace(
        spatial_encoding='one-hot', dim=4096, seed=0, limit_low=-5, limit_high=5,
    )
]

tilecoding_configs = [
    argparse.Namespace(
        spatial_encoding='tile-coding', dim=4096, seed=seed, limit_low=-5, limit_high=5,
        n_bins=16, n_tiles=16,
    ) for seed in seeds
]


configs = ssp_configs + pcgauss_configs + onehot_configs + tilecoding_configs


if not os.path.exists(fname_csv):

    # # TODO: use multiple seeds for each datapoint
    #
    # # TODO: incorporate dimensionality as another variable
    # #       need a different encoding function for each dim, may be easier to set up as partial functions
    # # TODO: set up to use pandas dataframes, for nice plotting
    # dimensions = [4, 16, 64, 256]
    #
    # seeds = np.arange(10)
    #
    # encodings = [
    #     # '2d',
    #     'ssp',
    #     'pc-gauss',
    #     'one-hot',
    #     'tile-coding',
    #     'hex-trig',
    #     # 'frozen-learned',
    # ]

    # noise_levels = [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000]

    noise_levels = [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0, 10]

    # n_seeds = len(seeds)
    # n_encodings = len(encodings)
    n_noise_levels = len(noise_levels)
    # n_dimensions = len(dimensions)

    df = pd.DataFrame()

    xs = np.linspace(args.limit_low, args.limit_high, args.res)
    ys = np.linspace(args.limit_low, args.limit_high, args.res)

    for config in configs:

        encoding_func, repr_dim = get_encoding_function(
            config, limit_low=args.train_limit_low, limit_high=args.train_limit_high
        )

        # for ix, x in enumerate(xs):
        #     for iy, y in enumerate(ys):
        #         activations[ix, iy, :] = encoding_func(x, y)
        #         normalized_activations[ix, iy, :] = activations[ix, iy, :] / np.linalg.norm(activations[ix, iy, :])
        #
        # encoded_point = encoding_func(args.x_pos, args.y_pos)
        #
        # heatmap = np.tensordot(encoded_point, activations, axes=([0], [2]))
        # normalized_heatmap = np.tensordot(encoded_point, normalized_activations, axes=([0], [2]))

        heatmap_vectors = np.zeros((len(xs), len(ys), config.dim))

        flat_heatmap_vectors = np.zeros((len(xs) * len(ys), config.dim))

        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                heatmap_vectors[i, j, :] = encoding_func(x, y)

                flat_heatmap_vectors[i * len(ys) + j, :] = heatmap_vectors[i, j, :].copy()

                # Normalize. This is required for frozen-learned to work
                heatmap_vectors[i, j, :] /= np.linalg.norm(heatmap_vectors[i, j, :])

        # Generate random samples

        # random positions not aligning with the heatmap
        true_pos = np.random.uniform(low=args.limit_low, high=args.limit_high, size=(args.n_samples, 2))

        encodings = np.zeros((args.n_samples, config.dim))

        for i in range(args.n_samples):
            encodings[i, :] = encoding_func(true_pos[i, 0], true_pos[i, 1])

        for ni, n in enumerate(noise_levels):

            print("Noise level: {}".format(n))

            noisy_encodings = encodings + np.random.normal(loc=0, scale=n, size=(args.n_samples, config.dim))

            predictions = ssp_to_loc_v(
                noisy_encodings,
                heatmap_vectors, xs, ys
            )

            # Root mean squared error
            rmse = np.sqrt(((predictions - true_pos)**2).mean())

            df = df.append(
                {
                    'Dimensions': config.dim,
                    'Noise Level': n,
                    'Encoding': config.spatial_encoding,
                    'RMSE': rmse,
                    'Seed': config.seed
                },
                ignore_index=True,
            )

            print(rmse)

    df.to_csv(fname_csv)

else:

    df = pd.read_csv(fname_csv)

# df = df[df['Dimensions'] == 16]

ax = sns.lineplot(data=df, x='Noise Level', y='RMSE', hue='Encoding')
ax.set_xscale('log')

plt.show()
