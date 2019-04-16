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
parser.add_argument('--n-samples', type=int, default=1000)
parser.add_argument('--dataset', type=str, default='../lab/reproducing/data/path_integration_trajectories_logits_200t_15s_seed13.npz')
parser.add_argument('--model', type=str, default='output/ssp_path_integration/clipped/Mar22_15-24-10/ssp_path_integration_model.pt', help='Saved model to load from')
parser.add_argument('--output', type=str, default='output/rate_maps.npz')
parser.add_argument('--encoding', type=str, default='ssp', choices=['ssp', '2d'])

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

data = np.load(args.dataset)

x_axis_vec = data['x_axis_vec']
y_axis_vec = data['y_axis_vec']

if args.encoding == 'ssp':
    encoding_dim = 512
    ssp_scaling = data['ssp_scaling']
elif args.encoding == '2d':
    encoding_dim = 2
    ssp_scaling = 1
else:
    raise NotImplementedError

limit_low = 0 * ssp_scaling
limit_high = 2.2 * ssp_scaling
res = 128 #256

xs = np.linspace(limit_low, limit_high, res)
ys = np.linspace(limit_low, limit_high, res)

# Used for visualization of test set performance using pos = ssp_to_loc(sp, heatmap_vectors, xs, ys)
heatmap_vectors = get_heatmap_vectors(xs, ys, x_axis_vec, y_axis_vec)

# n_samples = 5000
n_samples = args.n_samples#1000
rollout_length = args.trajectory_length#100
batch_size = args.minibatch_size#10

model = SSPPathIntegrationModel(unroll_length=rollout_length, sp_dim=encoding_dim)

if args.model:
    model.load_state_dict(torch.load(args.model), strict=False)


trainloader, testloader = train_test_loaders(
    data,
    n_train_samples=n_samples,
    n_test_samples=n_samples,
    rollout_length=rollout_length,
    batch_size=batch_size,
    encoding=args.encoding,
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

    # old version, something seems wrong here...
    # predictions = np.zeros((ssp_pred.shape[1]*ssp_pred.shape[2], 2))
    # coords = np.zeros((ssp_pred.shape[1]*ssp_pred.shape[2], 2))
    # activations = np.zeros((ssp_pred.shape[1]*ssp_pred.shape[2], model.lstm_hidden_size))

    predictions = np.zeros((ssp_pred.shape[0]*ssp_pred.shape[1], 2))
    coords = np.zeros((ssp_pred.shape[0]*ssp_pred.shape[1], 2))
    activations = np.zeros((ssp_pred.shape[0]*ssp_pred.shape[1], model.lstm_hidden_size))

    assert rollout_length == ssp_pred.shape[0]

    # For each neuron, contains the average activity at each spatial bin
    # Computing for both ground truth and predicted location
    rate_maps_pred = np.zeros((model.lstm_hidden_size, len(xs), len(ys)))
    rate_maps_truth = np.zeros((model.lstm_hidden_size, len(xs), len(ys)))

    print("Computing predicted locations and true locations")
    # Using all data, one chunk at a time
    for ri in range(rollout_length):

        # print("ssp_pred.shape[1]", ssp_pred.shape[1])
        # print("ri * ssp_pred.shape[1]", ri * ssp_pred.shape[1])
        # print("(ri + 1) * ssp_pred.shape[1]", (ri + 1) * ssp_pred.shape[1])
        # print("predictions.shape", predictions.shape)
        # print("ssp_pred.shape", ssp_pred.shape)
        # print("")

        if args.encoding == 'ssp':
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
        elif args.encoding == '2d':
            # copying 'predicted' coordinates, where the agent thinks it is
            predictions[ri * ssp_pred.shape[1]:(ri + 1) * ssp_pred.shape[1], :] = ssp_pred.detach().numpy()[ri, :, :]

            # copying 'ground truth' coordinates, where the agent should be
            coords[ri * ssp_pred.shape[1]:(ri + 1) * ssp_pred.shape[1], :] = ssp_outputs.detach().numpy()[:, ri, :]

        # reshaping activations and converting to numpy array
        activations[ri*ssp_pred.shape[1]:(ri+1)*ssp_pred.shape[1], :] = lstm_outputs.detach().numpy()[ri, :, :]

    print("Computing spatial firing maps")

    # used for 2D encoding case
    tol = (limit_high - limit_low) / res

    for xi, x in enumerate(xs):
        print("xloc {} of {}".format(xi + 1, len(xs)))
        for yi, y in enumerate(ys):

            if args.encoding == 'ssp':
                # Note: this assumes a discretized mapping using ssp_to_loc_v
                pred_inds = (predictions[:, 0] == x) & (predictions[:, 1] == y)
                truth_inds = (coords[:, 0] == x) & (coords[:, 1] == y)
            elif args.encoding == '2d':
                # set to true if it is the closest match in the linspace
                pred_inds = (np.abs(predictions[:, 0] - x) < tol) & (np.abs(predictions[:, 1] - y) < tol)
                truth_inds = (np.abs(coords[:, 0] - x) < tol) & (np.abs(coords[:, 1] - y) < tol)
            n_pred = np.sum(pred_inds)
            n_truth = np.sum(truth_inds)
            if n_pred > 0:
                for ni in range(model.lstm_hidden_size):
                    # np.sum(inds) should count only the true values in the boolean indexing array
                    rate_maps_pred[ni, xi, yi] = np.sum(activations[pred_inds, ni]) / n_pred
            if n_truth > 0:
                for ni in range(model.lstm_hidden_size):
                    # np.sum(inds) should count only the true values in the boolean indexing array
                    rate_maps_truth[ni, xi, yi] = np.sum(activations[truth_inds, ni]) / n_truth

    # TODO: save the firing maps here, so they can be processed later
    np.savez(
        args.output,
        rate_maps_pred=rate_maps_pred,
        rate_maps_truth=rate_maps_truth,
    )

    for ni in range(model.lstm_hidden_size):
        print("Neuron {} of {}".format(ni + 1, model.lstm_hidden_size))
        plt.imshow(rate_maps_pred[ni, :, :])
        plt.show()

