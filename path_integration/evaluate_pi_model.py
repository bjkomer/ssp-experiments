import argparse
import numpy as np
from ssp_navigation.utils.encodings import get_encoding_function, add_encoding_params
from models import SSPPathIntegrationModel, CircConvPathIntegrationModel, Simple2DPathIntegrationModel
from spatial_semantic_pointers.utils import get_heatmap_vectors, ssp_to_loc, ssp_to_loc_v
from datasets import train_test_loaders, train_test_loaders_jit
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns


parser = argparse.ArgumentParser(
    'Evaluate a path integration model'
)

parser.add_argument('--use-cconv', action='store_true', help='use cconv instead of a recurrent model')
parser = add_encoding_params(parser)
parser.add_argument('--dataset', type=str, default='data/path_integration_raw_trajectories_1000t_15s_seed13.npz')
parser.add_argument('--load-saved-model', type=str, default='', help='Saved model to load from')
parser.add_argument('--dropout-p', type=float, default=0.5)
parser.add_argument('--use-lmu', action='store_true', help='Use an LMU instead of an LSTM')
parser.add_argument('--lmu-order', type=int, default=6)
parser.add_argument('--batch-size', type=int, default=10)
parser.add_argument('--n-samples', type=int, default=1000)
parser.add_argument('--res', type=int, default=128)
parser.add_argument('--rollout-length', type=int, default=100)
args = parser.parse_args()

limit_low = 0.0
limit_high = 2.2
encoding_func, repr_dim = get_encoding_function(args, limit_low=limit_low, limit_high=limit_high)


xs = np.linspace(limit_low, limit_high, args.res)
ys = np.linspace(limit_low, limit_high, args.res)

heatmap_vectors = np.zeros((len(xs), len(ys), repr_dim))

print("Generating Heatmap Vectors")

for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        heatmap_vectors[i, j, :] = encoding_func(
            x=x, y=y
        )

        heatmap_vectors[i, j, :] /= np.linalg.norm(heatmap_vectors[i, j, :])

print("Heatmap Vector Generation Complete")

if args.use_cconv:
    if args.spatial_encoding == '2d':
        model = Simple2DPathIntegrationModel(
            unroll_length=args.rollout_length,
        )
    else:
        model = CircConvPathIntegrationModel(
            unroll_length=args.rollout_length,
            x_axis_vec=encoding_func(1, 0),
            y_axis_vec=encoding_func(0, 1),
        )
else:
    model = SSPPathIntegrationModel(
        input_size=2,
        unroll_length=args.rollout_length,
        sp_dim=repr_dim, dropout_p=args.dropout_p, use_lmu=args.use_lmu, order=args.lmu_order
    )
    if args.load_saved_model:
        model.load_state_dict(torch.load(args.load_saved_model), strict=False)
    else:
        print("Warning, no model given, random weights will be used")

data = np.load(args.dataset)

rng = np.random.RandomState(seed=13)

# trainloader, testloader = train_test_loaders(
trainloader, testloader = train_test_loaders_jit(
    data,
    n_train_samples=args.n_samples,
    n_test_samples=args.n_samples,
    rollout_length=args.rollout_length,
    batch_size=args.batch_size,
    encoding=args.spatial_encoding,
    encoding_func=encoding_func,
    rng=rng,
)

print("Testing")
with torch.no_grad():
    # Everything is in one batch, so this loop will only happen once
    for i, data in enumerate(testloader):
        velocity_inputs, ssp_inputs, ssp_outputs = data

        ssp_pred = model.forward(velocity_inputs, ssp_inputs)

        # note: first two indices of ssp_red and ssp_outputs are flipped


    # predictions = np.zeros((ssp_pred.shape[1]*ssp_pred.shape[2], 2))
    # coords = np.zeros((ssp_pred.shape[1]*ssp_pred.shape[2], 2))
    predictions = np.zeros((ssp_pred.shape[0]*ssp_pred.shape[1], 2))
    coords = np.zeros((ssp_pred.shape[0]*ssp_pred.shape[1], 2))

    print("Computing predicted locations and true locations")
    # Using all data, one chunk at a time
    for ri in range(args.rollout_length):

        if args.spatial_encoding == '2d':
            # copying 'predicted' coordinates, where the agent thinks it is
            predictions[ri * ssp_pred.shape[1]:(ri + 1) * ssp_pred.shape[1], :] = ssp_pred.detach().numpy()[ri, :, :]

            # copying 'ground truth' coordinates, where the agent should be
            coords[ri * ssp_pred.shape[1]:(ri + 1) * ssp_pred.shape[1], :] = ssp_outputs.detach().numpy()[:, ri, :]
        else:
            # computing 'predicted' coordinates, where the agent thinks it is
            predictions[ri * ssp_pred.shape[1]:(ri + 1) * ssp_pred.shape[1], :] = ssp_to_loc_v(
                ssp_pred.detach().numpy()[ri, :, :],
                heatmap_vectors, xs, ys
            )

            # computing 'ground truth' coordinates, where the agent should be
            coords[ri * ssp_pred.shape[1]:(ri + 1) * ssp_pred.shape[1], :] = ssp_to_loc_v(
                ssp_outputs.detach().numpy()[:, ri, :],
                heatmap_vectors, xs, ys
            )

    fig, ax = plt.subplots(1, args.batch_size)

    if args.batch_size == 1:
        # ax.scatter(
        #     coords[:, 0],
        #     coords[:, 1]
        # )

        ax.scatter(
            predictions[:, 0],
            predictions[:, 1],
            color='blue',
            label='predictions',
        )

        # add a bit of jitter to the coords so they can be seed
        ax.scatter(
            coords[:, 0] + 0.001,
            coords[:, 1] + 0.001,
            color='green',
            label='ground truth',
        )

        ax.set_xlim([0, 2.2])
        ax.set_ylim([0, 2.2])
        ax.set_aspect('equal')

        ax.legend()
    else:
        for bi in range(args.batch_size):
            ax[bi].scatter(
                predictions[bi * args.rollout_length:(bi + 1) * args.rollout_length, 0],
                predictions[bi * args.rollout_length:(bi + 1) * args.rollout_length, 1],
                color='blue',
                label='predictions',
            )
            ax[bi].scatter(
                coords[bi * args.rollout_length:(bi + 1) * args.rollout_length, 0],
                coords[bi * args.rollout_length:(bi + 1) * args.rollout_length, 1],
                color='green',
                label='ground truth',
            )

            ax[bi].set_xlim([0, 2.2])
            ax[bi].set_ylim([0, 2.2])
            ax[bi].set_aspect('equal')

            ax[bi].legend()


    plt.show()
