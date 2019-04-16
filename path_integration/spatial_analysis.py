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

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

data = np.load(args.dataset)

x_axis_vec = data['x_axis_vec']
y_axis_vec = data['y_axis_vec']
ssp_scaling = data['ssp_scaling']

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

    predictions = np.zeros((ssp_pred.shape[1]*ssp_pred.shape[2], 2))
    coords = np.zeros((ssp_pred.shape[1]*ssp_pred.shape[2], 2))
    activations = np.zeros((ssp_pred.shape[1]*ssp_pred.shape[2], model.lstm_hidden_size))

    # For each neuron, contains the average activity at each spatial bin
    # Computing for both ground truth and predicted location
    rate_maps_pred = np.zeros((model.lstm_hidden_size, len(xs), len(ys)))
    rate_maps_truth = np.zeros((model.lstm_hidden_size, len(xs), len(ys)))

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

        # reshaping activations and converting to numpy array
        activations[ri*ssp_pred.shape[1]:(ri+1)*ssp_pred.shape[1], :] = lstm_outputs.detach().numpy()[ri, :, :]

    print("Computing spatial firing maps")

    for xi, x in enumerate(xs):
        print("xloc {} of {}".format(xi + 1, len(xs)))
        for yi, y in enumerate(ys):
            pred_inds = (predictions[:, 0] == x) & (predictions[:, 1] == y)
            truth_inds = (coords[:, 0] == x) & (coords[:, 1] == y)
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

