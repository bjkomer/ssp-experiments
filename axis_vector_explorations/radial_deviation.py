import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from spatial_semantic_pointers.utils import make_good_unitary, get_heatmap_vectors, encode_point, power, \
    get_heatmap_vectors_hex, encode_point_hex
import seaborn as sns
import pandas as pd

parser = argparse.ArgumentParser('radial deviation for regular 2D and hex encoding')

parser.add_argument('--n-seeds', type=int, default=50)
parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--res', type=int, default=128, help='resolution of radii used')
parser.add_argument('--limit', type=float, default=5, help='highest radial distance to calculate')
parser.add_argument('--samples', type=int, default=100, help='number of samples at each radius to randomly draw')

args = parser.parse_args()

use_pandas = True

def compute_rad_dev(X, Y, Z=None, res=128, limit=5, samples=100, hex=False):
    rs = np.linspace(0, limit, res)

    ret = np.zeros((res, samples))

    # samples at random angles for each radius
    angles = np.random.uniform(low=0, high=2 * np.pi, size=(res, samples))

    for ir, r in enumerate(rs):

        for i in range(samples):

            # shouldn't matter which one is sin and cos here, angles are arbitrary
            x = r * np.cos(angles[ir, i])
            y = r * np.sin(angles[ir, i])
            # dot product with the origin will just end up being value of the first element
            if hex:
                ret[ir, i] = encode_point_hex(x, y, X, Y, Z).v[0]
            else:
                ret[ir, i] = encode_point(x, y, X, Y).v[0]

    return ret


if not os.path.exists('rad_dev_output'):
    os.makedirs('rad_dev_output')

fname = 'rad_dev_output/{}dim_{}res_{}samples_{}seeds.npz'.format(args.dim, args.res, args.samples, args.n_seeds)
fname_pd = 'rad_dev_output/{}dim_{}res_{}samples_{}seeds.csv'.format(args.dim, args.res, args.samples, args.n_seeds)

# square_axes = np.zeros((args.n_seeds, 2, args.dim))
# hex_axes = np.zeros((args.n_seeds, 3, args.dim))

rad_dev_square = np.zeros((args.n_seeds, args.res, args.samples))
rad_dev_hex = np.zeros((args.n_seeds, args.res, args.samples))

axes = np.zeros((args.n_seeds, 3, args.dim))

# if the data already exists, just load it
if os.path.exists(fname):
    data = np.load(fname)
    rad_dev_square=data['rad_dev_square']
    rad_dev_hex=data['rad_dev_hex']
else:

    for seed in range(args.n_seeds):
        print("\x1b[2K\r Seed {} of {}".format(seed + 1, args.n_seeds), end="\r")
        rng = np.random.RandomState(seed=seed)
        X = make_good_unitary(args.dim, rng=rng)
        Y = make_good_unitary(args.dim, rng=rng)
        Z = make_good_unitary(args.dim, rng=rng)

        axes[seed, 0, :] = X.v
        axes[seed, 1, :] = Y.v
        axes[seed, 2, :] = Z.v

        rad_dev_square[seed, :, :] = compute_rad_dev(X, Y, Z=None, res=args.res, limit=args.limit, samples=args.samples, hex=False)

        rad_dev_hex[seed, :, :] = compute_rad_dev(X, Y, Z, res=args.res, limit=args.limit, samples=args.samples, hex=True)

    np.savez(
        fname,
        axes=axes,
        rad_dev_square=rad_dev_square,
        rad_dev_hex=rad_dev_hex,
    )

if use_pandas:
    if os.path.exists(fname_pd):
        df = pd.read_csv(fname_pd)
    else:
        print("Generating Pandas Dataframe")
        df = pd.DataFrame()
        rs = np.linspace(0, args.limit, args.res)
        dfs = []
        for seed in range(args.n_seeds):
            print("\x1b[2K\r Seed {} of {}".format(seed + 1, args.n_seeds), end="\r")
            for ri, r in enumerate(rs):

                df_square = pd.DataFrame(
                    data=rad_dev_square[seed, ri, :],
                    columns=['Similarity']
                )
                df_square['Generation'] = 'Square'
                df_square['Radius'] = r
                df_square['Seed'] = seed

                df_hex = pd.DataFrame(
                    data=rad_dev_hex[seed, ri, :],
                    columns=['Similarity']
                )
                df_hex['Generation'] = 'Hexagonal'
                df_hex['Radius'] = r
                df_hex['Seed'] = seed

                dfs.append(df_square)
                dfs.append(df_hex)

                # df = df.append(df_square, ignore_index=True)
                # df = df.append(df_hex, ignore_index=True)

                # for n in range(args.samples):
                #     df = df.append(
                #         {
                #             'Generation': 'Square',
                #             'Similarity': rad_dev_square[seed, ri, n],
                #             'Radius': r,
                #             'Seed': seed,
                #         },
                #         ignore_index=True
                #     )
                #     df = df.append(
                #         {
                #             'Generation': 'Hexagonal',
                #             'Similarity': rad_dev_hex[seed, ri, n],
                #             'Radius': r,
                #             'Seed': seed,
                #         },
                #         ignore_index=True
                #     )

        # more efficient to do it this way
        df = df.append(dfs)

        df.to_csv(fname_pd)

    # std_square = df[df['Generation'] == 'Square'].groupby(['Radius'])[['Similarity']].std()
    # std_hex = df[df['Generation'] == 'Hexagonal'].groupby(['Radius'])[['Similarity']].std()
    #
    # plt.plot(std_square)
    # plt.plot(std_hex)
    # plt.title('Standard Deviation of Radial Deviation')

    # sns.lineplot(data=df[df['Seed'] == 0], x='Radius', y='Similarity', hue='Generation', ci='sd')
    sns.lineplot(data=df, x='Radius', y='Similarity', hue='Generation', ci='sd')

else:

    avg_rad_dev_square = rad_dev_square.mean(axis=2).mean(axis=0)
    avg_rad_dev_hex = rad_dev_hex.mean(axis=2).mean(axis=0)

    avg_abs_rad_dev_square = np.abs(rad_dev_square).mean(axis=2).mean(axis=0)
    avg_abs_rad_dev_hex = np.abs(rad_dev_hex).mean(axis=2).mean(axis=0)

    # plt.figure()
    # plt.plot(avg_rad_dev_square)
    # plt.title("Square Radial Deviation")
    # plt.figure()
    # plt.plot(avg_rad_dev_hex)
    # plt.title("Hexagonal Radial Deviation")
    # plt.figure()
    # plt.plot(avg_abs_rad_dev_square)
    # plt.title("Square Absolute Radial Deviation")
    # plt.figure()
    # plt.plot(avg_abs_rad_dev_hex)
    # plt.title("Hexagonal Absolute Radial Deviation")

    plt.figure()
    plt.plot(avg_rad_dev_square)
    plt.plot(avg_rad_dev_hex)
    plt.title("Radial Deviation")
    plt.figure()
    plt.plot(avg_abs_rad_dev_square)
    plt.plot(avg_abs_rad_dev_hex)
    plt.title("Absolute Radial Deviation")

plt.show()
