# make a plot of heatmap decoding error based on size of gaussian noise injected into the system.
# To be fair, should noise be scaled by the magnitude of the encoding vector? 1 in most cases, but not all
# Can also test 2D 'encoding' as a baseline. It has no inherent noise resilience
# another comparison could be a dimensionally expanded 2D, where each direction is corrupted differently
# can get a comparable measure here
from ssp_navigation.utils.encodings import get_encoding_function, add_encoding_params
from spatial_semantic_pointers.utils import ssp_to_loc_v
import argparse
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


parser = argparse.ArgumentParser()
# parser = add_encoding_params(parser)
parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--ssp-scaling', type=int, default=0.5)
parser.add_argument('--overwrite-output', action='store_true')
args = parser.parse_args()

res_base = 32
# limits = [1, 2, 5, 10, 20, 50, 100]
limits = [1, 2, 4, 8, 16, 32, 64]

dim = 256

noise_levels = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]

seeds = [0, 1, 2]

configs = []

for limit in limits:

    ssp_configs = [
        argparse.Namespace(
            spatial_encoding='ssp', dim=args.dim, seed=seed,
            limit_low=-limit, limit_high=limit,
            ssp_scaling=args.ssp_scaling,
        ) for seed in seeds
    ]

    pcgauss_configs = [
        argparse.Namespace(
            spatial_encoding='pc-gauss', dim=args.dim, seed=seed,
            limit_low=-limit, limit_high=limit,
            pc_gauss_sigma=0.75*(limit/5),
            hilbert_points=1,
        ) for seed in seeds
    ]

    # Seed has no effect on one-hot encoding
    onehot_configs = [
        argparse.Namespace(
            spatial_encoding='one-hot', dim=256, #dim=4096,
            seed=0,
            limit_low=-limit, limit_high=limit,
        )
    ]

    tilecoding_configs = [
        argparse.Namespace(
            spatial_encoding='tile-coding', dim=256, #dim=4096,
            seed=seed,
            limit_low=-limit, limit_high=limit,
            n_bins=8, n_tiles=4,
            # n_bins=16, n_tiles=16,
        ) for seed in seeds
    ]

    configs += ssp_configs + pcgauss_configs + onehot_configs + tilecoding_configs

fname_csv = 'noise_resilience_results.csv'

if os.path.exists(fname_csv) and not args.overwrite_output:
    df = pd.read_csv(fname_csv)
else:
    df = pd.DataFrame()
    for config in configs:

        print(config.spatial_encoding)
        print('Limit: {}'.format(config.limit_high))

        limit = config.limit_high

        res = res_base*limit

        xs = np.linspace(-limit, limit, res)
        ys = np.linspace(-limit, limit, res)

        encoding_func, repr_dim = get_encoding_function(config, limit_low=-limit, limit_high=limit)

        heatmap_vectors = np.zeros((len(xs), len(ys), repr_dim))

        flat_heatmap_vectors = np.zeros((len(xs) * len(ys), repr_dim))
        true_pos = np.zeros((len(xs) * len(ys), 2))

        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                heatmap_vectors[i, j, :] = encoding_func(x, y)

                flat_heatmap_vectors[i * len(ys) + j, :] = heatmap_vectors[i, j, :].copy()
                true_pos[i * len(ys) + j, 0] = x
                true_pos[i * len(ys) + j, 1] = y

                # Normalize. This is required for the dot product to be used
                heatmap_vectors[i, j, :] /= np.linalg.norm(heatmap_vectors[i, j, :])

        for noise_level in noise_levels:

            rng = np.random.RandomState(seed=config.seed)
            noisy_encodings = flat_heatmap_vectors + rng.normal(0, noise_level, size=(len(xs)*len(ys), repr_dim))

            predictions = ssp_to_loc_v(
                noisy_encodings,
                heatmap_vectors, xs, ys
            )

            # Root mean squared error
            rmse = np.sqrt(((predictions - true_pos) ** 2).mean())

            df = df.append(
                {
                    'Dimensions': repr_dim,
                    'Noise Level': noise_level,
                    'Limit': config.limit_high,
                    'Encoding': config.spatial_encoding,
                    'RMSE': rmse,
                    'Seed': config.seed
                },
                ignore_index=True,
            )

            print(rmse)

        df.to_csv(fname_csv)

    df.to_csv(fname_csv)


display = os.environ.get('DISPLAY')
if display is None or 'localhost' in display:
    print("No Display detected, skipping plot")
else:
    ax = sns.lineplot(data=df, x='Limits', y='RMSE', hue='Encoding')
    ax.set_xscale('log')

    plt.show()
