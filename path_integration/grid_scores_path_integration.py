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
from datasets import train_test_loaders, angular_train_test_loaders, tf_train_test_loaders, load_from_cache
from models import SSPPathIntegrationModel
from datetime import datetime
from tensorboardX import SummaryWriter
import json
from spatial_semantic_pointers.utils import get_heatmap_vectors, ssp_to_loc, ssp_to_loc_v
from spatial_semantic_pointers.plots import plot_predictions, plot_predictions_v
import matplotlib.pyplot as plt
from path_integration_utils import pc_to_loc_v, encoding_func_from_model, pc_gauss_encoding_func, ssp_encoding_func, \
    hd_gauss_encoding_func, hex_trig_encoding_func
from ssp_navigation.utils.encodings import get_encoding_function

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

parser.add_argument('--spatial-encoding', type=str, default='ssp',
                    choices=[
                        'ssp', 'hex-ssp', 'periodic-hex-ssp', 'grid-ssp', 'ind-ssp', 'orth-proj-ssp',
                        'rec-ssp', 'rec-hex-ssp', 'rec-ind-ssp',
                        'random', '2d', '2d-normalized', 'one-hot', 'hex-trig',
                        'trig', 'random-trig', 'random-rotated-trig', 'random-proj', 'legendre',
                        'learned', 'learned-normalized', 'frozen-learned', 'frozen-learned-normalized',
                        'pc-gauss', 'pc-dog', 'tile-coding'
                    ])
                    # choices=['ssp', '2d', 'frozen-learned', 'pc-gauss', 'pc-dog', 'pc-gauss-softmax', 'hex-trig', 'hex-trig-all-freq'])
parser.add_argument('--frozen-model', type=str, default='', help='model to use frozen encoding weights from')
parser.add_argument('--pc-gauss-sigma', type=float, default=0.25)
parser.add_argument('--pc-diff-sigma', type=float, default=0.5)
parser.add_argument('--hex-freq-coef', type=float, default=2.5, help='constant to scale frequencies by')
parser.add_argument('--n-tiles', type=int, default=8, help='number of layers for tile coding')
parser.add_argument('--n-bins', type=int, default=8, help='number of bins for tile coding')
parser.add_argument('--ssp-scaling', type=float, default=1.0)
parser.add_argument('--grid-ssp-min', type=float, default=0.25, help='minimum plane wave scale')
parser.add_argument('--grid-ssp-max', type=float, default=2.0, help='maximum plane wave scale')
parser.add_argument('--phi', type=float, default=0.5, help='phi as a fraction of pi for orth-proj-ssp')
parser.add_argument('--hilbert-points', type=int, default=1, choices=[0, 1, 2, 3],
                    help='pc centers. 0: random uniform. 1: hilbert curve. 2: evenly spaced grid. 3: hex grid')

parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--dropout-p', type=float, default=0.5)
parser.add_argument('--dim', type=int, default=512)
parser.add_argument('--train-split', type=float, default=0.8, help='Training fraction of the train/test split')
parser.add_argument('--allow-cache', action='store_true',
                    help='once the dataset has been generated, it will be saved to a file to be loaded faster')

parser.add_argument('--trajectory-length', type=int, default=100)
parser.add_argument('--minibatch-size', type=int, default=10)

parser.add_argument('--n-image-bins', type=int, default=20)

parser.add_argument('--n-hd-cells', type=int, default=0, help='If non-zero, use linear and angular velocity as well as HD cell output')
parser.add_argument('--sin-cos-ang', type=int, default=1, choices=[0, 1],
                    help='Use the sin and cos of the angular velocity if angular velocities are used')
parser.add_argument('--use-lmu', action='store_true')
parser.add_argument('--lmu-order', type=int, default=6)

args = parser.parse_args()

ssp_scaling = args.ssp_scaling

torch.manual_seed(args.seed)
np.random.seed(args.seed)

data = np.load(args.dataset)

# only used for frozen-learned and other custom encoding functions
# encoding_func = None

limit_low = 0 #* args.ssp_scaling
limit_high = 2.2 #* args.ssp_scaling
res = 128 #256

encoding_func, dim = get_encoding_function(args, limit_low=limit_low, limit_high=limit_high)

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
            # np.array(
            #     [x, y]
            # )
            # new signature
            x=x, y=y
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
    model = SSPPathIntegrationModel(
        input_size=input_size, unroll_length=rollout_length,
        sp_dim=dim + args.n_hd_cells, dropout_p=args.dropout_p, use_lmu=args.use_lmu, order=args.lmu_order
    )
else:
    hd_encoding_func = None
    model = SSPPathIntegrationModel(
        input_size=2, unroll_length=rollout_length,
        sp_dim=dim, dropout_p=args.dropout_p, use_lmu=args.use_lmu, order=args.lmu_order
    )


# model = SSPPathIntegrationModel(unroll_length=rollout_length, sp_dim=dim, dropout_p=args.dropout_p)

model.load_state_dict(torch.load(args.model), strict=False)

model.eval()

# encoding specific cache string
encoding_specific = ''
if 'ssp' in args.spatial_encoding:
    encoding_specific = args.ssp_scaling
elif args.spatial_encoding == 'frozen-learned':
    encoding_specific = args.frozen_model
elif args.spatial_encoding == 'pc-gauss' or args.spatial_encoding == 'pc-gauss-softmax':
    encoding_specific = args.pc_gauss_sigma
elif args.spatial_encoding == 'pc-dog':
    encoding_specific = '{}-{}'.format(args.pc_gauss_sigma, args.pc_diff_sigma)
elif args.spatial_encoding == 'hex-trig':
    encoding_specific = args.hex_freq_coef

if 'tf' in args.dataset:
    cache_fname = 'dataset_cache/tf_{}_{}_{}_{}_{}_{}.npz'.format(
        args.spatial_encoding, args.dim, args.seed, args.n_samples, args.n_hd_cells, encoding_specific
    )
else:
    cache_fname = 'dataset_cache/{}_{}_{}_{}_{}_{}.npz'.format(
        args.spatial_encoding, args.dim, args.seed, args.n_samples, args.n_hd_cells, encoding_specific
    )

# if the file exists, load it from cache
if os.path.exists(cache_fname):
    print("Generating Train and Test Loaders from Cache")
    trainloader, testloader = load_from_cache(cache_fname, batch_size=batch_size, n_samples=n_samples)
else:
    print("Generating Train and Test Loaders")

    if 'tf' in args.dataset:
        # tfrecord dataset only supports using the sin and cos of angular velocity
        assert args.sin_cos_ang == 1

        trainloader, testloader = tf_train_test_loaders(
            data,
            n_train_samples=n_samples,
            n_test_samples=n_samples,
            rollout_length=rollout_length,
            batch_size=batch_size,
            encoding=args.spatial_encoding,
            encoding_func=encoding_func,
            encoding_dim=args.dim,
            train_split=args.train_split,
            hd_dim=args.n_hd_cells,
            hd_encoding_func=hd_encoding_func,
            sin_cos_ang=args.sin_cos_ang,
        )

    else:

        if args.n_hd_cells > 0:
            trainloader, testloader = angular_train_test_loaders(
                data,
                n_train_samples=n_samples,
                n_test_samples=n_samples,
                rollout_length=rollout_length,
                batch_size=batch_size,
                encoding=args.spatial_encoding,
                encoding_func=encoding_func,
                encoding_dim=args.dim,
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
                encoding=args.spatial_encoding,
                encoding_func=encoding_func,
                encoding_dim=args.dim,
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
    nbins=args.n_image_bins,
    coords_range=((0, 2.2), (0, 2.2)),  # data_reader.get_coord_range(),
    mask_parameters=masks_parameters,
)


fname_lstm_pred = '{}_{}samples_lstm_pred.pdf'.format(args.fname_prefix, args.n_samples)
fname_lstm_truth = '{}_{}samples_lstm_truth.pdf'.format(args.fname_prefix, args.n_samples)
fname_dense_pred = '{}_{}samples_dense_pred.pdf'.format(args.fname_prefix, args.n_samples)
fname_dense_truth = '{}_{}samples_dense_truth.pdf'.format(args.fname_prefix, args.n_samples)

# Run and gather activations

print("Testing")
with torch.no_grad():
    # Everything is in one batch, so this loop will only happen once
    for i, data in enumerate(testloader):
        velocity_inputs, ssp_inputs, ssp_outputs = data

        ssp_pred, lstm_outputs, dense_outputs = model.forward_activations(velocity_inputs, ssp_inputs)

    predictions = np.zeros((ssp_pred.shape[0]*ssp_pred.shape[1], 2))
    coords = np.zeros((ssp_pred.shape[0]*ssp_pred.shape[1], 2))
    lstm_activations = np.zeros((ssp_pred.shape[0]*ssp_pred.shape[1], model.lstm_hidden_size))
    dense_activations = np.zeros((ssp_pred.shape[0] * ssp_pred.shape[1], model.linear_hidden_size))

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
        pred = ssp_pred.detach().numpy()[ri, :, :args.dim]
        # pred = pred / pred.sum(axis=1)[:, np.newaxis]
        predictions[ri * ssp_pred.shape[1]:(ri + 1) * ssp_pred.shape[1], :] = ssp_to_loc_v(
            pred,
            heatmap_vectors, xs, ys
        )

        # computing 'ground truth' coordinates, where the agent should be
        coord = ssp_outputs.detach().numpy()[:, ri, :args.dim]
        # coord = coord / coord.sum(axis=1)[:, np.newaxis]
        coords[ri * ssp_pred.shape[1]:(ri + 1) * ssp_pred.shape[1], :] = ssp_to_loc_v(
            coord,
            heatmap_vectors, xs, ys
        )

        # reshaping activations and converting to numpy array
        lstm_activations[ri*ssp_pred.shape[1]:(ri+1)*ssp_pred.shape[1], :] = lstm_outputs.detach().numpy()[ri, :, :]
        dense_activations[ri * ssp_pred.shape[1]:(ri + 1) * ssp_pred.shape[1], :] = dense_outputs.detach().numpy()[ri, :, :]

# predictions = predictions / args.ssp_scaling
# coords = coords / args.ssp_scaling

print(np.max(predictions))
print(np.min(predictions))

grid_scores_60_pred, grid_scores_90_pred, grid_scores_60_separation_pred, grid_scores_90_separation_pred = utils.get_scores_and_plot(
    scorer=latest_epoch_scorer,
    data_abs_xy=predictions, #res['pos_xy'],
    activations=lstm_activations, #res['bottleneck'],
    directory='output_grid_scores', #FLAGS.saver_results_directory,
    filename=fname_lstm_pred,
)

grid_scores_60_truth, grid_scores_90_truth, grid_scores_60_separation_truth, grid_scores_90_separation_truth = utils.get_scores_and_plot(
    scorer=latest_epoch_scorer,
    data_abs_xy=coords, #res['pos_xy'],
    activations=lstm_activations, #res['bottleneck'],
    directory='output_grid_scores', #FLAGS.saver_results_directory,
    filename=fname_lstm_truth,
)

grid_scores_60_dense_pred, grid_scores_90_dense_pred, grid_scores_60_separation_dense_pred, grid_scores_90_separation_dense_pred = utils.get_scores_and_plot(
    scorer=latest_epoch_scorer,
    data_abs_xy=predictions, #res['pos_xy'],
    activations=dense_activations, #res['bottleneck'],
    directory='output_grid_scores', #FLAGS.saver_results_directory,
    filename=fname_dense_pred,
)

grid_scores_60_dense_truth, grid_scores_90_dense_truth, grid_scores_60_separation_dense_truth, grid_scores_90_separation_dense_truth = utils.get_scores_and_plot(
    scorer=latest_epoch_scorer,
    data_abs_xy=coords, #res['pos_xy'],
    activations=dense_activations, #res['bottleneck'],
    directory='output_grid_scores', #FLAGS.saver_results_directory,
    filename=fname_dense_truth,
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

    grid_scores_60_dense_pred=grid_scores_60_dense_pred,
    grid_scores_90_dense_pred=grid_scores_90_dense_pred,
    grid_scores_60_separation_dense_pred=grid_scores_60_separation_dense_pred,
    grid_scores_90_separation_dense_pred=grid_scores_90_separation_dense_pred,
    grid_scores_60_dense_truth=grid_scores_60_dense_truth,
    grid_scores_90_dense_truth=grid_scores_90_dense_truth,
    grid_scores_60_separation_dense_truth=grid_scores_60_separation_dense_truth,
    grid_scores_90_separation_dense_truth=grid_scores_90_separation_dense_truth,
)
