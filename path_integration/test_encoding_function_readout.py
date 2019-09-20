from path_integration_utils import encoding_func_from_model, pc_gauss_encoding_func, hex_trig_encoding_func, \
    one_hot_encoding
import numpy as np
from spatial_semantic_pointers.utils import ssp_to_loc_v, encode_point, make_good_unitary
from spatial_semantic_pointers.plots import plot_predictions, plot_predictions_v
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser('View the effectiveness of an encoding function readout')

parser.add_argument('--dim', type=int, default=512)
parser.add_argument('--res', type=int, default=64)
parser.add_argument('--ssp-scaling', type=float, default=1.0)
parser.add_argument('--limit-low', type=float, default=0.0)
parser.add_argument('--limit-high', type=float, default=2.2)
parser.add_argument('--encoding', type=str, default='ssp',
                    choices=['ssp', 'frozen-learned', 'pc-gauss', 'pc-gauss-softmax', 'hex-trig', 'one-hot'])
parser.add_argument('--pc-gauss-sigma', type=float, default=0.25)
parser.add_argument('--frozen-model-path', type=str, default='frozen_models/blocks_10_100_model.pt')
parser.add_argument('--seed', type=int, default=13)

parser.add_argument('--n-show-activations', type=int, default=3, help='number of activations to show')
parser.add_argument('--noise', type=float, default=0, help='amount of noise to add to the encoding')

args = parser.parse_args()


if args.encoding == 'frozen-learned':
    # Generate an encoding function from the model path
    encoding_func = encoding_func_from_model(args.frozen_model_path)
elif args.encoding == 'ssp':

    rng = np.random.RandomState(args.seed)
    x_axis_sp = make_good_unitary(args.dim, rng=rng)
    y_axis_sp = make_good_unitary(args.dim, rng=rng)

    def encoding_func(coords):
        return encode_point(
            x=coords[0], y=coords[1], x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp,
        ).v
elif args.encoding == 'pc-gauss' or args.encoding == 'pc-gauss-softmax':
    use_softmax = args.encoding == 'pc-gauss-softmax'
    rng = np.random.RandomState(args.seed)
    encoding_func = pc_gauss_encoding_func(
        limit_low=args.limit_low, limit_high=args.limit_high, dim=args.dim, sigma=args.pc_gauss_sigma, rng=rng, use_softmax=use_softmax
    )
elif args.encoding == 'hex-trig':
    encoding_func = hex_trig_encoding_func(
        dim=args.dim, seed=args.seed,
        frequencies=(2.5, 2.5 * 1.4, 2.5 * 1.4 * 1.4)
    )
elif args.encoding == 'one-hot':
    encoding_func = one_hot_encoding(
        dim=args.dim,
        limit_low=args.limit_low,
        limit_high=args.limit_high,
    )


xs = np.linspace(args.limit_low, args.limit_high, args.res)
ys = np.linspace(args.limit_low, args.limit_high, args.res)

# encoding for every point in a 2D linspace, for approximating a readout

# FIXME: inefficient but will work for now
heatmap_vectors = np.zeros((len(xs), len(ys), args.dim))

flat_heatmap_vectors = np.zeros((len(xs) * len(ys), args.dim))

for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        heatmap_vectors[i, j, :] = encoding_func(
            # batch dim
            # np.array(
            #     [[x, y]]
            # )
            # no batch dim
            np.array(
                [x, y]
            )
        )

        flat_heatmap_vectors[i*len(ys)+j, :] = heatmap_vectors[i, j, :].copy()

        # Normalize. This is required for frozen-learned to work
        heatmap_vectors[i, j, :] /= np.linalg.norm(heatmap_vectors[i, j, :])


n_samples = 5000
# random positions not aligning with the heatmap
rand_pos = np.random.uniform(low=args.limit_low, high=args.limit_high, size=(n_samples, 2))

encodings = np.zeros((n_samples, args.dim))

for i in range(n_samples):
    encodings[i, :] = encoding_func(
        np.array(
            [rand_pos[i, 0], rand_pos[i, 1]]
        )
    )

    # encodings[i, :] /= np.linalg.norm(encodings[i, :])

encodings += np.random.normal(loc=0, scale=args.noise, size=(n_samples, args.dim))

# encodings /= encodings.sum(axis=1)[:, np.newaxis]


predictions = ssp_to_loc_v(
    # flat_heatmap_vectors,
    encodings,
    heatmap_vectors, xs, ys
)

print(predictions)

coords = predictions.copy()

fig_pred, ax_pred = plt.subplots()


print("plotting predicted locations")
plot_predictions_v(predictions / args.ssp_scaling, coords / args.ssp_scaling, ax_pred, min_val=args.limit_low, max_val=args.limit_high)

if args.n_show_activations > 0:
    for i in range(min(args.dim, args.n_show_activations)):
        plt.figure()
        plt.imshow(heatmap_vectors[:, :, i])


plt.show()
