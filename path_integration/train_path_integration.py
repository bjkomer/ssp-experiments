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


parser = argparse.ArgumentParser(
    'Run 2D supervised path integration experiment using pytorch. Allows various encoding methods'
)

# parser = add_parameters(parser)

parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--n-epochs', type=int, default=20)
parser.add_argument('--n-samples', type=int, default=1000)
parser.add_argument('--spatial-encoding', type=str, default='ssp',
                    choices=[
                        'ssp', 'hex-ssp', 'periodic-hex-ssp', 'grid-ssp', 'ind-ssp', 'orth-proj-ssp',
                        'rec-ssp', 'rec-hex-ssp', 'rec-ind-ssp', 'sub-toroid-ssp', 'var-sub-toroid-ssp',
                        'random', '2d', '2d-normalized', 'one-hot', 'hex-trig',
                        'trig', 'random-trig', 'random-rotated-trig', 'random-proj', 'legendre',
                        'learned', 'learned-normalized', 'frozen-learned', 'frozen-learned-normalized',
                        'pc-gauss', 'pc-dog', 'tile-coding'
                    ])
                    # choices=['ssp', '2d', 'frozen-learned', 'pc-gauss', 'pc-gauss-softmax', 'hex-trig', 'hex-trig-all-freq'])
parser.add_argument('--eval-period', type=int, default=50)
parser.add_argument('--logdir', type=str, default='output/ssp_path_integration',
                    help='Directory for saved model and tensorboard log')
parser.add_argument('--dataset', type=str, default='data/path_integration_raw_trajectories_1000t_15s_seed13.npz')
parser.add_argument('--load-saved-model', type=str, default='', help='Saved model to load from')
parser.add_argument('--loss-function', type=str, default='mse',
                    choices=['mse', 'cosine', 'combined', 'alternating', 'scaled'])
parser.add_argument('--frozen-model', type=str, default='', help='model to use frozen encoding weights from')

# Encoding specific parameters
parser.add_argument('--pc-gauss-sigma', type=float, default=0.75)
parser.add_argument('--pc-diff-sigma', type=float, default=1.50)
parser.add_argument('--hex-freq-coef', type=float, default=2.5, help='constant to scale frequencies by')
parser.add_argument('--n-tiles', type=int, default=8, help='number of layers for tile coding')
parser.add_argument('--n-bins', type=int, default=8, help='number of bins for tile coding')
parser.add_argument('--ssp-scaling', type=float, default=1.0)
parser.add_argument('--grid-ssp-min', type=float, default=0.25, help='minimum plane wave scale')
parser.add_argument('--grid-ssp-max', type=float, default=2.0, help='maximum plane wave scale')
parser.add_argument('--phi', type=float, default=0.5, help='phi as a fraction of pi for orth-proj-ssp')
parser.add_argument('--n-proj', type=int, default=3, help='projection dimension for sub toroids')
parser.add_argument('--scale-ratio', type=float, default=0, help='ratio between sub toroid scales')
parser.add_argument('--hilbert-points', type=int, default=1, choices=[0, 1, 2, 3],
                    help='pc centers. 0: random uniform. 1: hilbert curve. 2: evenly spaced grid. 3: hex grid')

parser.add_argument('--dropout-p', type=float, default=0.5)
parser.add_argument('--dim', type=int, default=512)
parser.add_argument('--train-split', type=float, default=0.8, help='Training fraction of the train/test split')
parser.add_argument('--allow-cache', action='store_true',
                    help='once the dataset has been generated, it will be saved to a file to be loaded faster')
# TODO: add option for random vs evenly spaced HD cells
parser.add_argument('--n-hd-cells', type=int, default=0, help='If non-zero, use linear and angular velocity as well as HD cell output')
parser.add_argument('--grad-clip-thresh', type=float, default=1e-5, help='Gradient clipping threshold')
parser.add_argument('--minibatch-size', type=int, default=10,
                    help='Number of trajectories used in the calculation of a stochastic gradient')
parser.add_argument('--trajectory-length', type=int, default=100,
                    help='Number of time steps in the trajectories used for the supervised learning task')
parser.add_argument('--learning-rate', type=float, default=1e-5, help='Step size multiplier in the RMSProp algorithm')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum parameter of the RMSProp algorithm')
parser.add_argument('--regularization-param', type=float, default=1e-5, help='Regularisation parameter for linear layer')

parser.add_argument('--sin-cos-ang', type=int, default=1, choices=[0, 1],
                    help='Use the sin and cos of the angular velocity if angular velocities are used')

parser.add_argument('--use-lmu', action='store_true', help='Use an LMU instead of an LSTM')
parser.add_argument('--lmu-order', type=int, default=6)

parser.add_argument('--non-negative', action='store_true', help='Restrict weights in the linear layer to be positive')
parser.add_argument('--all-non-negative', action='store_true', help='Restrict all weights to be positive')

parser.add_argument('--optimizer', type=str, default='rmsprop', choices=['rmsprop', 'adam'])

parser.add_argument('--gpu', type=int, default=-1,
                    help="set to an integer corresponding to the gpu to use. Set to -1 to use the CPU")

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
save_dir = os.path.join(args.logdir, current_time)
writer = SummaryWriter(log_dir=save_dir)

data = np.load(args.dataset)

if args.gpu == -1:
    device = torch.device('cpu:0')
    pin_memory = False
else:
    device = torch.device('cuda:{}'.format(int(args.gpu)))
    pin_memory = True

# only used for frozen-learned and other custom encoding functions
# encoding_func = None


limit_low = 0 #* args.ssp_scaling
limit_high = 2.2 #* args.ssp_scaling
res = 128 #256

encoding_func, dim = get_encoding_function(args, limit_low=limit_low, limit_high=limit_high)
# dim = args.encoding_dim

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

# n_samples = 5000
n_samples = args.n_samples#1000
rollout_length = args.trajectory_length#100
batch_size = args.minibatch_size#10
n_epochs = args.n_epochs#20
# n_epochs = 5

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
        input_size=2,
        unroll_length=rollout_length,
        sp_dim=dim, dropout_p=args.dropout_p, use_lmu=args.use_lmu, order=args.lmu_order
    )

if args.load_saved_model:
    model.load_state_dict(torch.load(args.load_saved_model), strict=False)

model.to(device)

# trying out cosine similarity here as well, it works better than MSE as a loss for SSP cleanup
cosine_criterion = nn.CosineEmbeddingLoss()
mse_criterion = nn.MSELoss()

if args.optimizer == 'rmsprop':
    optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
elif args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
else:
    raise NotImplementedError

# encoding specific cache string
encoding_specific = ''
if args.spatial_encoding == 'ssp' or args.spatial_encoding == 'hex-ssp':
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
    trainloader, testloader = load_from_cache(cache_fname, batch_size=batch_size, n_samples=n_samples, device=device)
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
                pin_memory=False,#pin_memory,
                device=device,
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
                pin_memory=False,#pin_memory,
                device=device,
            )

    if args.allow_cache:

        if not os.path.exists('dataset_cache'):
            os.makedirs('dataset_cache')

        np.savez(
            cache_fname,
            train_velocity_inputs=trainloader.dataset.velocity_inputs.cpu().numpy(),
            train_ssp_inputs=trainloader.dataset.ssp_inputs.cpu().numpy(),
            train_ssp_outputs=trainloader.dataset.ssp_outputs.cpu().numpy(),
            test_velocity_inputs=testloader.dataset.velocity_inputs.cpu().numpy(),
            test_ssp_inputs=testloader.dataset.ssp_inputs.cpu().numpy(),
            test_ssp_outputs=testloader.dataset.ssp_outputs.cpu().numpy(),
        )

print("Train and Test Loaders Generation Complete")

params = vars(args)
with open(os.path.join(save_dir, "params.json"), "w") as f:
    json.dump(params, f)


class NonNegativeWeights(object):

    def __init__(self, frequency=1):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, 'weight'):
            # print("has weight attribute")
            w = module.weight.data
            w = w.clamp(min=0)
            module.weight.data = w


non_negative_weights = NonNegativeWeights()

# Keep track of running average losses, to adaptively scale the weight between them
running_avg_cosine_loss = 1.
running_avg_mse_loss = 1.

print("Training")
for epoch in range(n_epochs):
    # print('\x1b[2K\r Epoch {} of {}'.format(epoch + 1, n_epochs), end="\r")
    print('Epoch {} of {}'.format(epoch + 1, n_epochs))

    # TODO: modularize this and clean it up
    # Every 'eval_period' epochs, create a test loss and image
    if epoch % args.eval_period == 0:
        print('Evaluating at Epoch {}'.format(epoch + 1))
        with torch.no_grad():
            # Everything is in one batch, so this loop will only happen once
            for i, data in enumerate(testloader):
                velocity_inputs, ssp_inputs, ssp_outputs = data
                # velocity_inputs = velocity_inputs.to(device)
                # ssp_inputs = ssp_inputs.to(device)
                # ssp_outputs = ssp_outputs.to(device)

                ssp_pred = model(velocity_inputs, ssp_inputs)

                # NOTE: need to permute axes of the targets here because the output is
                #       (sequence length, batch, units) instead of (batch, sequence_length, units)
                #       could also permute the outputs instead
                cosine_loss = cosine_criterion(
                    ssp_pred.reshape(ssp_pred.shape[0] * ssp_pred.shape[1], ssp_pred.shape[2]),
                    ssp_outputs.permute(1, 0, 2).reshape(ssp_pred.shape[0] * ssp_pred.shape[1], ssp_pred.shape[2]),
                    torch.ones(ssp_pred.shape[0] * ssp_pred.shape[1]).to(device)
                )
                mse_loss = mse_criterion(ssp_pred, ssp_outputs.permute(1, 0, 2))

                # TODO: handle loss differently for HD version

                print("test cosine loss", cosine_loss.data.item())
                print("test mse loss", mse_loss.data.item())
                writer.add_scalar('test_cosine_loss', cosine_loss.data.item(), epoch)
                writer.add_scalar('test_mse_loss', mse_loss.data.item(), epoch)
                writer.add_scalar('test_combined_loss', mse_loss.data.item() + cosine_loss.data.item(), epoch)
                c_f = (running_avg_mse_loss / (running_avg_mse_loss + running_avg_cosine_loss))
                m_f = (running_avg_cosine_loss / (running_avg_mse_loss + running_avg_cosine_loss))
                writer.add_scalar(
                    'test_scaled_loss',
                    mse_loss.data.item() * m_f + cosine_loss.data.item() * c_f,
                    epoch
                )

            # print("ssp_pred.shape", ssp_pred.shape)
            # print("ssp_outputs.shape", ssp_outputs.shape)

            # Just use start and end location to save on memory and computation
            predictions_start = np.zeros((ssp_pred.shape[1], 2))
            coords_start = np.zeros((ssp_pred.shape[1], 2))

            predictions_end = np.zeros((ssp_pred.shape[1], 2))
            coords_end = np.zeros((ssp_pred.shape[1], 2))

            if args.spatial_encoding == 'ssp':
                print("computing prediction locations")
                predictions_start[:, :] = ssp_to_loc_v(
                    ssp_pred.detach().cpu().numpy()[0, :, :args.dim],
                    heatmap_vectors, xs, ys
                )
                predictions_end[:, :] = ssp_to_loc_v(
                    ssp_pred.detach().cpu().numpy()[-1, :, :args.dim],
                    heatmap_vectors, xs, ys
                )
                print("computing ground truth locations")
                coords_start[:, :] = ssp_to_loc_v(
                    ssp_outputs.detach().cpu().numpy()[:, 0, :args.dim],
                    heatmap_vectors, xs, ys
                )
                coords_end[:, :] = ssp_to_loc_v(
                    ssp_outputs.detach().cpu().numpy()[:, -1, :args.dim],
                    heatmap_vectors, xs, ys
                )
            elif args.spatial_encoding == '2d':
                print("copying prediction locations")
                predictions_start[:, :] = ssp_pred.detach().cpu().numpy()[0, :, :]
                predictions_end[:, :] = ssp_pred.detach().cpu().numpy()[-1, :, :]
                print("copying ground truth locations")
                coords_start[:, :] = ssp_outputs.detach().cpu().numpy()[:, 0, :]
                coords_end[:, :] = ssp_outputs.detach().cpu().numpy()[:, -1, :]
            else:
                # normalizing is important here
                print("computing prediction locations")
                pred_start = ssp_pred.detach().cpu().numpy()[0, :, :args.dim]
                # pred_start = pred_start / pred_start.sum(axis=1)[:, np.newaxis]
                predictions_start[:, :] = ssp_to_loc_v(
                    pred_start,
                    heatmap_vectors, xs, ys
                )
                pred_end = ssp_pred.detach().cpu().numpy()[-1, :, :args.dim]
                # pred_end = pred_end / pred_end.sum(axis=1)[:, np.newaxis]
                predictions_end[:, :] = ssp_to_loc_v(
                    pred_end,
                    heatmap_vectors, xs, ys
                )
                print("computing ground truth locations")
                coord_start = ssp_outputs.detach().cpu().numpy()[:, 0, :args.dim]
                # coord_start = coord_start / coord_start.sum(axis=1)[:, np.newaxis]
                coords_start[:, :] = ssp_to_loc_v(
                    coord_start,
                    heatmap_vectors, xs, ys
                )
                coord_end = ssp_outputs.detach().cpu().numpy()[:, -1, :args.dim]
                # coord_end = coord_end / coord_end.sum(axis=1)[:, np.newaxis]
                coords_end[:, :] = ssp_to_loc_v(
                    coord_end,
                    heatmap_vectors, xs, ys
                )

            fig_pred_start, ax_pred_start = plt.subplots()
            fig_truth_start, ax_truth_start = plt.subplots()
            fig_pred_end, ax_pred_end = plt.subplots()
            fig_truth_end, ax_truth_end = plt.subplots()

            # print("plotting predicted locations")
            # plot_predictions_v(predictions_start / args.ssp_scaling, coords_start / args.ssp_scaling, ax_pred_start, min_val=0, max_val=2.2, fixed_axes=True)
            # plot_predictions_v(predictions_end / args.ssp_scaling, coords_end / args.ssp_scaling, ax_pred_end, min_val=0, max_val=2.2, fixed_axes=True)
            # print("plotting ground truth locations")
            # plot_predictions_v(coords_start / args.ssp_scaling, coords_start / args.ssp_scaling, ax_truth_start, min_val=0, max_val=2.2, fixed_axes=True)
            # plot_predictions_v(coords_end / args.ssp_scaling, coords_end / args.ssp_scaling, ax_truth_end, min_val=0, max_val=2.2, fixed_axes=True)

            print("plotting predicted locations")
            plot_predictions_v(predictions_start, coords_start, ax_pred_start, min_val=0, max_val=2.2, fixed_axes=True)
            plot_predictions_v(predictions_end, coords_end, ax_pred_end, min_val=0, max_val=2.2, fixed_axes=True)
            # only plot ground truth once at the start
            if epoch == 0:
                print("plotting ground truth locations")
                plot_predictions_v(coords_start, coords_start, ax_truth_start, min_val=0, max_val=2.2, fixed_axes=True)
                plot_predictions_v(coords_end, coords_end, ax_truth_end, min_val=0, max_val=2.2, fixed_axes=True)
                writer.add_figure("ground truth start", fig_truth_start, epoch)
                writer.add_figure("ground truth end", fig_truth_end, epoch)

            writer.add_figure("predictions start", fig_pred_start, epoch)
            writer.add_figure("predictions end", fig_pred_end, epoch)

            torch.save(
                model.state_dict(),
                os.path.join(save_dir, '{}_path_integration_model_epoch_{}.pt'.format(args.spatial_encoding, epoch))
            )

    avg_bce_loss = 0
    avg_cosine_loss = 0
    avg_mse_loss = 0
    avg_combined_loss = 0
    avg_scaled_loss = 0
    n_batches = 0
    for i, data in enumerate(trainloader):
        velocity_inputs, ssp_inputs, ssp_outputs = data
        # velocity_inputs = velocity_inputs.to(device)
        # ssp_inputs = ssp_inputs.to(device)
        # ssp_outputs = ssp_outputs.to(device)

        if ssp_inputs.size()[0] != batch_size:
            continue  # Drop data, not enough for a batch
        optimizer.zero_grad()
        # model.zero_grad()

        ssp_pred = model(velocity_inputs, ssp_inputs)

        # NOTE: need to permute axes of the targets here because the output is
        #       (sequence length, batch, units) instead of (batch, sequence_length, units)
        #       could also permute the outputs instead
        cosine_loss = cosine_criterion(
            ssp_pred.reshape(ssp_pred.shape[0] * ssp_pred.shape[1], ssp_pred.shape[2]),
            ssp_outputs.permute(1, 0, 2).reshape(ssp_pred.shape[0] * ssp_pred.shape[1], ssp_pred.shape[2]),
            torch.ones(ssp_pred.shape[0] * ssp_pred.shape[1]).to(device)
        )
        mse_loss = mse_criterion(ssp_pred, ssp_outputs.permute(1, 0, 2))
        loss = cosine_loss + mse_loss

        # adaptive weighted combination of the two loss functions
        c_f = (running_avg_mse_loss / (running_avg_mse_loss + running_avg_cosine_loss))
        m_f = (running_avg_cosine_loss / (running_avg_mse_loss + running_avg_cosine_loss))
        scaled_loss = cosine_loss * c_f + mse_loss * m_f

        if args.loss_function == 'cosine':
            cosine_loss.backward()
        elif args.loss_function == 'mse':
            mse_loss.backward()
        elif args.loss_function == 'combined':
            loss.backward()
        elif args.loss_function == 'alternating':
            if epoch % 2 == 0:
                cosine_loss.backward()
            else:
                mse_loss.backward()
        elif args.loss_function == 'scaled':
            scaled_loss.backward()

        avg_cosine_loss += cosine_loss.data.item()
        avg_mse_loss += mse_loss.data.item()
        avg_combined_loss += (cosine_loss.data.item() + mse_loss.data.item())
        avg_scaled_loss += scaled_loss.data.item()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_thresh)

        optimizer.step()

        if args.non_negative:
            if args.all_non_negative:
                # Set any negative weights to 0
                model.apply(non_negative_weights)
            else:
                # only targeting the one set of weights
                w = model.linear.weight.data
                w = w.clamp(min=0)
                model.linear.weight.data = w

        # avg_loss += loss.data.item()
        n_batches += 1

    avg_cosine_loss /= n_batches
    avg_mse_loss /= n_batches
    avg_combined_loss /= n_batches
    avg_scaled_loss /= n_batches
    print("cosine loss:", avg_cosine_loss)
    print("mse loss:", avg_mse_loss)
    print("combined loss:", avg_combined_loss)
    print("scaled loss:", avg_scaled_loss)

    running_avg_cosine_loss = 0.9 * running_avg_cosine_loss + 0.1 * avg_cosine_loss
    running_avg_mse_loss = 0.9 * running_avg_mse_loss + 0.1 * avg_mse_loss
    print("running_avg_cosine_loss", running_avg_cosine_loss)
    print("running_avg_mse_loss", running_avg_mse_loss)

    writer.add_scalar('avg_cosine_loss', avg_cosine_loss, epoch + 1)
    writer.add_scalar('avg_mse_loss', avg_mse_loss, epoch + 1)
    writer.add_scalar('avg_combined_loss', avg_combined_loss, epoch + 1)
    writer.add_scalar('avg_scaled_loss', avg_scaled_loss, epoch + 1)


print("Testing")
with torch.no_grad():
    # Everything is in one batch, so this loop will only happen once
    for i, data in enumerate(testloader):
        velocity_inputs, ssp_inputs, ssp_outputs = data

        ssp_pred = model(velocity_inputs, ssp_inputs)

        # NOTE: need to permute axes of the targets here because the output is
        #       (sequence length, batch, units) instead of (batch, sequence_length, units)
        #       could also permute the outputs instead
        cosine_loss = cosine_criterion(
            ssp_pred.reshape(ssp_pred.shape[0] * ssp_pred.shape[1], ssp_pred.shape[2]),
            ssp_outputs.permute(1, 0, 2).reshape(ssp_pred.shape[0] * ssp_pred.shape[1], ssp_pred.shape[2]),
            torch.ones(ssp_pred.shape[0] * ssp_pred.shape[1]).to(device)
        )
        mse_loss = mse_criterion(ssp_pred, ssp_outputs.permute(1, 0, 2))

        print("final test cosine loss", cosine_loss.data.item())
        print("final test mse loss", mse_loss.data.item())
        writer.add_scalar('final_test_cosine_loss', cosine_loss.data.item(), epoch)
        writer.add_scalar('final_test_mse_loss', mse_loss.data.item(), epoch)
        writer.add_scalar('final_test_combined_loss', mse_loss.data.item() + cosine_loss.data.item(), epoch)
        c_f = (running_avg_mse_loss / (running_avg_mse_loss + running_avg_cosine_loss))
        m_f = (running_avg_cosine_loss / (running_avg_mse_loss + running_avg_cosine_loss))
        writer.add_scalar(
            'final_test_scaled_loss',
            mse_loss.data.item() * m_f + cosine_loss.data.item() * c_f,
            epoch
        )

    # Just use start and end location to save on memory and computation
    predictions_start = np.zeros((ssp_pred.shape[1], 2))
    coords_start = np.zeros((ssp_pred.shape[1], 2))

    predictions_end = np.zeros((ssp_pred.shape[1], 2))
    coords_end = np.zeros((ssp_pred.shape[1], 2))

    if args.spatial_encoding == 'ssp':
        print("computing prediction locations")
        predictions_start[:, :] = ssp_to_loc_v(
            ssp_pred.detach().cpu().numpy()[0, :, :args.dim],
            heatmap_vectors, xs, ys
        )
        predictions_end[:, :] = ssp_to_loc_v(
            ssp_pred.detach().cpu().numpy()[-1, :, :args.dim],
            heatmap_vectors, xs, ys
        )
        print("computing ground truth locations")
        coords_start[:, :] = ssp_to_loc_v(
            ssp_outputs.detach().cpu().numpy()[:, 0, :args.dim],
            heatmap_vectors, xs, ys
        )
        coords_end[:, :] = ssp_to_loc_v(
            ssp_outputs.detach().cpu().numpy()[:, -1, :args.dim],
            heatmap_vectors, xs, ys
        )
    elif args.spatial_encoding == '2d':
        print("copying prediction locations")
        predictions_start[:, :] = ssp_pred.detach().cpu().numpy()[0, :, :]
        predictions_end[:, :] = ssp_pred.detach().cpu().numpy()[-1, :, :]
        print("copying ground truth locations")
        coords_start[:, :] = ssp_outputs.detach().cpu().numpy()[:, 0, :]
        coords_end[:, :] = ssp_outputs.detach().cpu().numpy()[:, -1, :]
    else:
        # normalizing is important here
        print("computing prediction locations")
        pred_start = ssp_pred.detach().cpu().numpy()[0, :, :args.dim]
        # pred_start = pred_start / pred_start.sum(axis=1)[:, np.newaxis]
        predictions_start[:, :] = ssp_to_loc_v(
            pred_start,
            heatmap_vectors, xs, ys
        )
        pred_end = ssp_pred.detach().cpu().numpy()[-1, :, :args.dim]
        # pred_end = pred_end / pred_end.sum(axis=1)[:, np.newaxis]
        predictions_end[:, :] = ssp_to_loc_v(
            pred_end,
            heatmap_vectors, xs, ys
        )
        print("computing ground truth locations")
        coord_start = ssp_outputs.detach().cpu().numpy()[:, 0, :args.dim]
        # coord_start = coord_start / coord_start.sum(axis=1)[:, np.newaxis]
        coords_start[:, :] = ssp_to_loc_v(
            coord_start,
            heatmap_vectors, xs, ys
        )
        coord_end = ssp_outputs.detach().cpu().numpy()[:, -1, :args.dim]
        # coord_end = coord_end / coord_end.sum(axis=1)[:, np.newaxis]
        coords_end[:, :] = ssp_to_loc_v(
            coord_end,
            heatmap_vectors, xs, ys
        )

    fig_pred_start, ax_pred_start = plt.subplots()
    fig_truth_start, ax_truth_start = plt.subplots()
    fig_pred_end, ax_pred_end = plt.subplots()
    fig_truth_end, ax_truth_end = plt.subplots()

    # print("plotting predicted locations")
    # plot_predictions_v(predictions_start / args.ssp_scaling, coords_start / args.ssp_scaling, ax_pred_start, min_val=0, max_val=2.2, fixed_axes=True)
    # plot_predictions_v(predictions_end / args.ssp_scaling, coords_end / args.ssp_scaling, ax_pred_end, min_val=0, max_val=2.2, fixed_axes=True)
    # print("plotting ground truth locations")
    # plot_predictions_v(coords_start / args.ssp_scaling, coords_start / args.ssp_scaling, ax_truth_start, min_val=0, max_val=2.2, fixed_axes=True)
    # plot_predictions_v(coords_end / args.ssp_scaling, coords_end / args.ssp_scaling, ax_truth_end, min_val=0, max_val=2.2, fixed_axes=True)

    print("plotting predicted locations")
    plot_predictions_v(predictions_start, coords_start, ax_pred_start, min_val=0, max_val=2.2, fixed_axes=True)
    plot_predictions_v(predictions_end, coords_end, ax_pred_end, min_val=0, max_val=2.2, fixed_axes=True)
    print("plotting ground truth locations")
    plot_predictions_v(coords_start, coords_start, ax_truth_start, min_val=0, max_val=2.2, fixed_axes=True)
    plot_predictions_v(coords_end, coords_end, ax_truth_end, min_val=0, max_val=2.2, fixed_axes=True)

    writer.add_figure("final predictions start", fig_pred_start)
    writer.add_figure("final ground truth start", fig_truth_start)

    writer.add_figure("final predictions end", fig_pred_end)
    writer.add_figure("final ground truth end", fig_truth_end)

torch.save(model.state_dict(), os.path.join(save_dir, '{}_path_integration_model.pt'.format(args.spatial_encoding)))
