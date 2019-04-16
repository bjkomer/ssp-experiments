import matplotlib
import os
# allow code to work on machines without a display or in a screen session
display = os.environ.get('DISPLAY')
if display is None or 'localhost' in display:
    matplotlib.use('agg')

import argparse
import numpy as np
# NOTE: this is currently soft-linked to this directory
from arguments import add_parameters
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import train_test_loaders
from models import SSPPathIntegrationModel
from datetime import datetime
from tensorboardX import SummaryWriter
import json
from spatial_semantic_pointers.utils import get_heatmap_vectors, ssp_to_loc, ssp_to_loc_v
from spatial_semantic_pointers.plots import plot_predictions, plot_predictions_v
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser('Run 2D supervised path integration experiment using pytorch')

parser = add_parameters(parser)

parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--n-samples', type=int, default=1)
parser.add_argument('--dataset', type=str, default='../lab/reproducing/data/path_integration_trajectories_logits_200t_15s_seed13.npz')
parser.add_argument('--model', type=str, default='output/ssp_path_integration/clipped/Mar22_15-24-10/ssp_path_integration_model.pt', help='Saved model to load from')

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

data = np.load(args.dataset)

x_axis_vec = data['x_axis_vec']
y_axis_vec = data['y_axis_vec']
ssp_scaling = data['ssp_scaling']

limit_low = 0 * ssp_scaling
limit_high = 2.2 * ssp_scaling
res = 256#512#128 #256

xs = np.linspace(limit_low, limit_high, res)
ys = np.linspace(limit_low, limit_high, res)

# Used for visualization of test set performance using pos = ssp_to_loc(sp, heatmap_vectors, xs, ys)
heatmap_vectors = get_heatmap_vectors(xs, ys, x_axis_vec, y_axis_vec)

# n_samples = 5000
n_samples = args.n_samples#1000
rollout_length = args.trajectory_length#100
batch_size = 1

model = SSPPathIntegrationModel(unroll_length=rollout_length)

if args.model:
    model.load_state_dict(torch.load(args.model), strict=False)


trainloader, testloader = train_test_loaders(
    data,
    n_train_samples=n_samples,
    n_test_samples=n_samples,
    rollout_length=rollout_length,
    batch_size=batch_size
)

print("Testing")
with torch.no_grad():
    # Everything is in one batch, so this loop will only happen once
    for i, data in enumerate(testloader):
        velocity_inputs, ssp_inputs, ssp_outputs = data

        ssp_pred, lstm_outputs = model.forward_activations(velocity_inputs, ssp_inputs)


    print("ssp_pred.shape", ssp_pred.shape)
    print("ssp_outputs.shape", ssp_outputs.shape)
    print("lstm_outputs.shape", lstm_outputs.shape)

    # predictions = np.zeros((ssp_pred.shape[1]*ssp_pred.shape[2], 2))
    # coords = np.zeros((ssp_pred.shape[1]*ssp_pred.shape[2], 2))
    predictions = np.zeros((ssp_pred.shape[0]*ssp_pred.shape[1], 2))
    coords = np.zeros((ssp_pred.shape[0]*ssp_pred.shape[1], 2))

    print("Computing predicted locations and true locations")
    # Using all data, one chunk at a time
    for ri in range(rollout_length):
        # computing 'predicted' coordinates, where the agent thinks it is
        predictions[ri*ssp_pred.shape[1]:(ri+1)*ssp_pred.shape[1], :] = ssp_to_loc_v(
            ssp_pred.detach().numpy()[ri, :, :],
            heatmap_vectors, xs, ys
        )

        # computing 'ground truth' coordinates, where the agent should be
        coords[ri*ssp_pred.shape[1]:(ri+1)*ssp_pred.shape[1], :] = ssp_to_loc_v(
            ssp_outputs.detach().numpy()[:, ri, :],
            heatmap_vectors, xs, ys
        )

    fig, ax = plt.subplots(1, batch_size)

    if batch_size == 1:
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

        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            color='green',
            label='ground truth',
        )

        ax.legend()
    else:
        for bi in range(batch_size):
            ax[bi].scatter(
                predictions[bi * rollout_length:(bi + 1) * rollout_length, 0],
                predictions[bi * rollout_length:(bi + 1) * rollout_length, 1],
                color='blue',
                label='predictions',
            )
            ax[bi].scatter(
                coords[bi * rollout_length:(bi + 1) * rollout_length, 0],
                coords[bi * rollout_length:(bi + 1) * rollout_length, 1],
                color='green',
                label='ground truth',
            )

            ax[bi].legend()

    plt.show()
