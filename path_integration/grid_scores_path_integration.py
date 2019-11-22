# Compute grid scores using the new dataset format

import matplotlib
import os
# allow code to work on machines without a display or in a screen session
display = os.environ.get('DISPLAY')
if display is None or 'localhost' in display:
    matplotlib.use('agg')

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import train_test_loaders, angular_train_test_loaders, load_from_cache
from models import SSPPathIntegrationModel
from datetime import datetime
from tensorboardX import SummaryWriter
import json
from spatial_semantic_pointers.utils import get_heatmap_vectors, ssp_to_loc, ssp_to_loc_v
from spatial_semantic_pointers.plots import plot_predictions, plot_predictions_v
import matplotlib.pyplot as plt
from path_integration_utils import pc_to_loc_v, encoding_func_from_model, pc_gauss_encoding_func, ssp_encoding_func, \
    hd_gauss_encoding_func, hex_trig_encoding_func



import grid_scoring.scores as scores
import grid_scoring.utils as utils
# from grid_scoring.run_network import run_and_gather_activations, run_and_gather_localization_activations
from path_integration_utils import encoding_func_from_model, pc_gauss_encoding_func


parser = argparse.ArgumentParser('Compute grid scores for a path integration model')
parser.add_argument('--n-samples', type=int, default=5000)
parser.add_argument('--use-localization', action='store_true')
# TODO: use these parameters
parser.add_argument('--dataset', type=str, default='')
parser.add_argument('--model', type=str, default='')
parser.add_argument('--fname-prefix', type=str, default='sac')
parser.add_argument('--ssp-scaling', type=float, default=1.0)
parser.add_argument('--encoding', type=str, default='ssp',
                    choices=['ssp', '2d', 'frozen-learned', 'pc-gauss', 'pc-gauss-softmax', 'hex-trig', 'hex-trig-all-freq'])
parser.add_argument('--frozen-model', type=str, default='', help='model to use frozen encoding weights from')
parser.add_argument('--pc-gauss-sigma', type=float, default=0.25)
parser.add_argument('--hex-freq-coef', type=float, default=2.5, help='constant to scale frequencies by')


parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--dropout-p', type=float, default=0.5)
parser.add_argument('--encoding-dim', type=int, default=512)
parser.add_argument('--train-split', type=float, default=0.8, help='Training fraction of the train/test split')
parser.add_argument('--allow-cache', action='store_true',
                    help='once the dataset has been generated, it will be saved to a file to be loaded faster')

parser.add_argument('--trajectory-length', type=int, default=100)
parser.add_argument('--minibatch-size', type=int, default=10)

parser.add_argument('--n-bins', type=int, default=20)

parser.add_argument('--n-hd-cells', type=int, default=0, help='If non-zero, use linear and angular velocity as well as HD cell output')
parser.add_argument('--sin-cos-ang', type=int, default=1, choices=[0, 1],
                    help='Use the sin and cos of the angular velocity if angular velocities are used')

args = parser.parse_args()

ssp_scaling = args.ssp_scaling

torch.manual_seed(args.seed)
np.random.seed(args.seed)

data = np.load(args.dataset)

# only used for frozen-learned and other custom encoding functions
encoding_func = None

if args.encoding == 'ssp':
    dim = args.encoding_dim
    encoding_func = ssp_encoding_func(seed=args.seed, dim=dim, ssp_scaling=args.ssp_scaling)
elif args.encoding == '2d':
    dim = 2
    ssp_scaling = 1  # no scaling used for 2D coordinates directly
elif args.encoding == 'pc':
    dim = args.n_place_cells
    ssp_scaling = 1
elif args.encoding == 'frozen-learned':
    dim = args.encoding_dim
    ssp_scaling = 1
    # Generate an encoding function from the model path
    encoding_func = encoding_func_from_model(args.frozen_model)
elif args.encoding == 'pc-gauss' or args.encoding == 'pc-gauss-softmax':
    dim = args.encoding_dim
    ssp_scaling = 1
    use_softmax = args.encoding == 'pc-guass-softmax'
    # Generate an encoding function from the model path
    rng = np.random.RandomState(args.seed)
    encoding_func = pc_gauss_encoding_func(
        limit_low=0 * ssp_scaling, limit_high=2.2 * ssp_scaling,
        dim=dim, rng=rng, sigma=args.pc_gauss_sigma,
        use_softmax=use_softmax
    )
elif args.encoding == 'hex-trig':
    dim = args.encoding_dim
    ssp_scaling = 1
    encoding_func = hex_trig_encoding_func(
        dim=dim, seed=args.seed,
        frequencies=(args.hex_freq_coef, args.hex_freq_coef*1.4, args.hex_freq_coef*1.4 * 1.4)
    )
elif args.encoding == 'hex-trig-all-freq':
    dim = args.encoding_dim
    ssp_scaling = 1
    encoding_func = hex_trig_encoding_func(
        dim=dim, seed=args.seed,
        frequencies=np.linspace(1, 10, 100),
    )
else:
    raise NotImplementedError

limit_low = 0 #* args.ssp_scaling
limit_high = 2.2 #* args.ssp_scaling
res = 128 #256

xs = np.linspace(limit_low, limit_high, res)
ys = np.linspace(limit_low, limit_high, res)

# FIXME: inefficient but will work for now
heatmap_vectors = np.zeros((len(xs), len(ys), dim))

print("Generating Heatmap Vectors")

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

        heatmap_vectors[i, j, :] /= np.linalg.norm(heatmap_vectors[i, j, :])

print("Heatmap Vector Generation Complete")

n_samples = args.n_samples
rollout_length = args.trajectory_length
batch_size = args.minibatch_size


if args.n_hd_cells > 0:
    hd_encoding_func = hd_gauss_encoding_func(dim=args.n_hd_cells, sigma=0.25, use_softmax=False, rng=np.random.RandomState(args.seed))
    if args.sin_cos_ang:
        input_size = 3
    else:
        input_size = 2
    model = SSPPathIntegrationModel(input_size=input_size, unroll_length=rollout_length, sp_dim=dim + args.n_hd_cells, dropout_p=args.dropout_p)
else:
    hd_encoding_func = None
    model = SSPPathIntegrationModel(unroll_length=rollout_length, sp_dim=dim, dropout_p=args.dropout_p)


# model = SSPPathIntegrationModel(unroll_length=rollout_length, sp_dim=dim, dropout_p=args.dropout_p)

model.load_state_dict(torch.load(args.model), strict=False)

model.eval()

# encoding specific cache string
encoding_specific = ''
if args.encoding == 'ssp':
    encoding_specific = args.ssp_scaling
elif args.encoding == 'frozen-learned':
    encoding_specific = args.frozen_model
elif args.encoding == 'pc-gauss' or args.encoding == 'pc-gauss-softmax':
    encoding_specific = args.pc_gauss_sigma
elif args.encoding == 'hex-trig':
    encoding_specific = args.hex_freq_coef

# cache_fname = 'dataset_cache/{}_{}_{}_{}_{}.npz'.format(
#     args.encoding, args.encoding_dim, args.seed, args.n_samples, encoding_specific
# )
cache_fname = 'dataset_cache/{}_{}_{}_{}_{}_{}.npz'.format(
    args.encoding, args.encoding_dim, args.seed, args.n_samples, args.n_hd_cells, encoding_specific
)

# if the file exists, load it from cache
if os.path.exists(cache_fname):
    print("Generating Train and Test Loaders from Cache")
    trainloader, testloader = load_from_cache(cache_fname, batch_size=batch_size, n_samples=n_samples)
else:
    print("Generating Train and Test Loaders")

    if args.n_hd_cells > 0:
        trainloader, testloader = angular_train_test_loaders(
            data,
            n_train_samples=n_samples,
            n_test_samples=n_samples,
            rollout_length=rollout_length,
            batch_size=batch_size,
            encoding=args.encoding,
            encoding_func=encoding_func,
            encoding_dim=args.encoding_dim,
            train_split=args.train_split,
            hd_dim=args.n_hd_cells,
            hd_encoding_func=hd_encoding_func,
            sin_cos_ang=args.sin_cos_ang,
        )
    else:
        trainloader, testloader = train_test_loaders(
            data,
            n_train_samples=n_samples,
            n_test_samples=n_samples,
            rollout_length=rollout_length,
            batch_size=batch_size,
            encoding=args.encoding,
            encoding_func=encoding_func,
            encoding_dim=args.encoding_dim,
            train_split=args.train_split,
        )

    if args.allow_cache:

        if not os.path.exists('dataset_cache'):
            os.makedirs('dataset_cache')

        np.savez(
            cache_fname,
            train_velocity_inputs=trainloader.dataset.velocity_inputs,
            train_ssp_inputs=trainloader.dataset.ssp_inputs,
            train_ssp_outputs=trainloader.dataset.ssp_outputs,
            test_velocity_inputs=testloader.dataset.velocity_inputs,
            test_ssp_inputs=testloader.dataset.ssp_inputs,
            test_ssp_outputs=testloader.dataset.ssp_outputs,
        )

print("Train and Test Loaders Generation Complete")

starts = [0.2] * 10
ends = np.linspace(0.4, 1.0, num=10)
masks_parameters = zip(starts, ends.tolist())
latest_epoch_scorer = scores.GridScorer(
    nbins=args.n_bins,
    coords_range=((0, 2.2), (0, 2.2)),  # data_reader.get_coord_range(),
    mask_parameters=masks_parameters,
)


fname_pred = '{}_{}samples_pred.pdf'.format(args.fname_prefix, args.n_samples)
fname_truth = '{}_{}samples_truth.pdf'.format(args.fname_prefix, args.n_samples)

# Run and gather activations

print("Testing")
with torch.no_grad():
    # Everything is in one batch, so this loop will only happen once
    for i, data in enumerate(testloader):
        velocity_inputs, ssp_inputs, ssp_outputs = data

        ssp_pred, lstm_outputs = model.forward_activations(velocity_inputs, ssp_inputs)

    predictions = np.zeros((ssp_pred.shape[0]*ssp_pred.shape[1], 2))
    coords = np.zeros((ssp_pred.shape[0]*ssp_pred.shape[1], 2))
    activations = np.zeros((ssp_pred.shape[0]*ssp_pred.shape[1], model.lstm_hidden_size))

    assert rollout_length == ssp_pred.shape[0]

    # # For each neuron, contains the average activity at each spatial bin
    # # Computing for both ground truth and predicted location
    # rate_maps_pred = np.zeros((model.lstm_hidden_size, len(xs), len(ys)))
    # rate_maps_truth = np.zeros((model.lstm_hidden_size, len(xs), len(ys)))

    print("Computing predicted locations and true locations")
    # Using all data, one chunk at a time
    for ri in range(rollout_length):

        # trim out head direction info if that was included by only looking up to args.encoding_dim

        # computing 'predicted' coordinates, where the agent thinks it is
        pred = ssp_pred.detach().numpy()[ri, :, :args.encoding_dim]
        # pred = pred / pred.sum(axis=1)[:, np.newaxis]
        predictions[ri * ssp_pred.shape[1]:(ri + 1) * ssp_pred.shape[1], :] = ssp_to_loc_v(
            pred,
            heatmap_vectors, xs, ys
        )

        # computing 'ground truth' coordinates, where the agent should be
        coord = ssp_outputs.detach().numpy()[:, ri, :args.encoding_dim]
        # coord = coord / coord.sum(axis=1)[:, np.newaxis]
        coords[ri * ssp_pred.shape[1]:(ri + 1) * ssp_pred.shape[1], :] = ssp_to_loc_v(
            coord,
            heatmap_vectors, xs, ys
        )

        # reshaping activations and converting to numpy array
        activations[ri*ssp_pred.shape[1]:(ri+1)*ssp_pred.shape[1], :] = lstm_outputs.detach().numpy()[ri, :, :]

# predictions = predictions / args.ssp_scaling
# coords = coords / args.ssp_scaling

print(np.max(predictions))
print(np.min(predictions))

grid_scores_60_pred, grid_scores_90_pred, grid_scores_60_separation_pred, grid_scores_90_separation_pred = utils.get_scores_and_plot(
    scorer=latest_epoch_scorer,
    data_abs_xy=predictions, #res['pos_xy'],
    activations=activations, #res['bottleneck'],
    directory='output_grid_scores', #FLAGS.saver_results_directory,
    filename=fname_pred,
)

grid_scores_60_truth, grid_scores_90_truth, grid_scores_60_separation_truth, grid_scores_90_separation_truth = utils.get_scores_and_plot(
    scorer=latest_epoch_scorer,
    data_abs_xy=coords, #res['pos_xy'],
    activations=activations, #res['bottleneck'],
    directory='output_grid_scores', #FLAGS.saver_results_directory,
    filename=fname_truth,
)


print(grid_scores_60_truth, grid_scores_90_truth, grid_scores_60_separation_truth, grid_scores_90_separation_truth)

# Saving to make grid score values easy to compare for different variations
fname = 'output_grid_scores/{}_{}samples.npz'.format(args.fname_prefix, args.n_samples)
np.savez(
    fname,
    grid_scores_60_pred=grid_scores_60_pred,
    grid_scores_90_pred=grid_scores_90_pred,
    grid_scores_60_separation_pred=grid_scores_60_separation_pred,
    grid_scores_90_separation_pred=grid_scores_90_separation_pred,
    grid_scores_60_truth=grid_scores_60_truth,
    grid_scores_90_truth=grid_scores_90_truth,
    grid_scores_60_separation_truth=grid_scores_60_separation_truth,
    grid_scores_90_separation_truth=grid_scores_90_separation_truth,
)
