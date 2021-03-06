import numpy as np
from path_integration_utils import encoding_func_from_model, pc_gauss_encoding_func, hex_trig_encoding_func
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
parser.add_argument('--ssp-scaling', type=float, default=1.0)
parser.add_argument('--limit-low', type=float, default=0.0)
parser.add_argument('--limit-high', type=float, default=2.2)

parser.add_argument('--pc-gauss-sigma', type=float, default=0.25)
parser.add_argument('--frozen-model-path', type=str, default='frozen_models/blocks_10_100_model.pt')
parser.add_argument('--seed', type=int, default=13)

parser.add_argument('--n-samples', type=int, default=5000)


args = parser.parse_args()

# fname = 'output/evaluate_encodings_noise.npz'
fname_csv = 'output/evaluate_encodings_noise_multiseed.csv'

if not os.path.exists(fname_csv):

    # TODO: use multiple seeds for each datapoint

    # TODO: incorporate dimensionality as another variable
    #       need a different encoding function for each dim, may be easier to set up as partial functions
    # TODO: set up to use pandas dataframes, for nice plotting
    dimensions = [4, 16, 64, 256]

    seeds = np.arange(10)

    encodings = [
        # '2d',
        'ssp',
        'pc-gauss',
        'hex-trig',
        # 'frozen-learned',
    ]

    noise_levels = [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000]

    n_seeds = len(seeds)
    n_encodings = len(encodings)
    n_noise_levels = len(noise_levels)
    n_dimensions = len(dimensions)

    # Generate an encoding function from the model path
    fl_enc_func = encoding_func_from_model(args.frozen_model_path)


    def ssp_enc_func(coords, dim=256, seed=13):
        rng = np.random.RandomState(seed)
        x_axis_sp = make_good_unitary(dim, rng=rng)
        y_axis_sp = make_good_unitary(dim, rng=rng)
        return encode_point(
            x=coords[0], y=coords[1], x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp,
        ).v


    ssp_funcs = []
    pc_gauss_funcs = []
    hex_trig_funcs = []
    for di, d in enumerate(dimensions):

        for si, s in enumerate(seeds):

            # rng = np.random.RandomState(args.seed)
            # x_axis_sp = make_good_unitary(d, rng=rng)
            # y_axis_sp = make_good_unitary(d, rng=rng)
            #
            # def ssp_enc_func(coords):
            #     return encode_point(
            #         x=coords[0], y=coords[1], x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp,
            #     ).v

            # ssp_funcs.append(ssp_enc_func)
            ssp_funcs.append(partial(ssp_enc_func, dim=d, seed=s))


            use_softmax = False
            rng = np.random.RandomState(s)
            pc_gauss_enc_func = pc_gauss_encoding_func(
                limit_low=args.limit_low, limit_high=args.limit_high, dim=d, sigma=args.pc_gauss_sigma, rng=rng, use_softmax=use_softmax
            )
            pc_gauss_funcs.append(pc_gauss_enc_func)
            # pc_gauss_funcs.append(
            #     partial(
            #         pc_gauss_encoding_func,
            #         limit_low=args.limit_low, limit_high=args.limit_high,
            #         dim=d, sigma=args.pc_gauss_sigma, rng=rng, use_softmax=use_softmax)
            # )

            hex_trig_enc_func = hex_trig_encoding_func(
                dim=d, seed=s,
                frequencies=(2.5, 2.5 * 1.4, 2.5 * 1.4 * 1.4)
            )
            hex_trig_funcs.append(hex_trig_enc_func)

            # hex_trig_funcs.append(
            #     partial(
            #         hex_trig_encoding_func,
            #         dim=d, seed=args.seed,
            #         frequencies=(2.5, 2.5 * 1.4, 2.5 * 1.4 * 1.4)
            #     )
            # )

    encoding_functions = [
        ssp_funcs,
        pc_gauss_funcs,
        hex_trig_funcs,
    ]

    # encoding_functions = {
    #     # '2d': None,
    #     'ssp': ssp_enc_func,
    #     'pc-gauss': pc_gauss_enc_func,
    #     'hex-trig': hex_trig_enc_func,
    #     'frozen-learned': fl_enc_func,
    # }


    # results = np.zeros((n_encodings, n_dimensions, n_noise_levels))

    df = pd.DataFrame()

    xs = np.linspace(args.limit_low, args.limit_high, args.res)
    ys = np.linspace(args.limit_low, args.limit_high, args.res)

    for ei, e in enumerate(encodings):

        print("Encoding: {}".format(e))

        for di, d in enumerate(dimensions):
            print("Dim: {}".format(d))

            heatmap_vectors = np.zeros((len(xs), len(ys), d))

            flat_heatmap_vectors = np.zeros((len(xs) * len(ys), d))

            for si, s in enumerate(seeds):
                print("Seed: {}".format(s))

                for i, x in enumerate(xs):
                    for j, y in enumerate(ys):
                        heatmap_vectors[i, j, :] = encoding_functions[ei][di*n_seeds + si](
                            # no batch dim
                            np.array(
                                [x, y]
                            )
                        )

                        flat_heatmap_vectors[i * len(ys) + j, :] = heatmap_vectors[i, j, :].copy()

                        # Normalize. This is required for frozen-learned to work
                        heatmap_vectors[i, j, :] /= np.linalg.norm(heatmap_vectors[i, j, :])

                # Generate random samples

                # random positions not aligning with the heatmap
                true_pos = np.random.uniform(low=args.limit_low, high=args.limit_high, size=(args.n_samples, 2))

                encodings = np.zeros((args.n_samples, d))

                for i in range(args.n_samples):
                    encodings[i, :] = encoding_functions[ei][di*n_seeds + si](
                        np.array(
                            [true_pos[i, 0], true_pos[i, 1]]
                        )
                    )

                for ni, n in enumerate(noise_levels):

                    print("Noise level: {}".format(n))

                    noisy_encodings = encodings + np.random.normal(loc=0, scale=n, size=(args.n_samples, d))

                    predictions = ssp_to_loc_v(
                        noisy_encodings,
                        heatmap_vectors, xs, ys
                    )

                    # Root mean squared error
                    # results[ei, di, ni] = np.sqrt(((predictions - true_pos)**2).mean())
                    rmse = np.sqrt(((predictions - true_pos)**2).mean())

                    df = df.append(
                        {
                            'Dimensions': d,
                            'Noise Level': n,
                            'Encoding': e,
                            'RMSE': rmse,
                            'Seed': s
                        },
                        ignore_index=True,
                    )

                    print(rmse)


    df.to_csv(fname_csv)

else:

    df = pd.read_csv(fname_csv)

ax = sns.lineplot(data=df[df['Dimensions'] == 16], x='Noise Level', y='RMSE', hue='Encoding')
# ax = sns.lineplot(data=df, x='Noise Level', y='RMSE', hue='Encoding')
ax.set_xscale('log')

plt.show()

# np.savez(fname, results=results)
#
#
# plt.plot(results)
# plt.show()
