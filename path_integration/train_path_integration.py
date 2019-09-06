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
from path_integration_utils import pc_to_loc_v, encoding_func_from_model, pc_gauss_encoding_func, ssp_encoding_func


parser = argparse.ArgumentParser(
    'Run 2D supervised path integration experiment using pytorch. Allows various encoding methods'
)

parser = add_parameters(parser)

parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--n-epochs', type=int, default=20)
parser.add_argument('--n-samples', type=int, default=1000)
parser.add_argument('--encoding', type=str, default='ssp',
                    choices=['ssp', '2d', 'frozen-learned', 'pc-gauss', 'pc-gauss-softmax'])
parser.add_argument('--eval-period', type=int, default=50)
parser.add_argument('--logdir', type=str, default='output/ssp_path_integration',
                    help='Directory for saved model and tensorboard log')
parser.add_argument('--dataset', type=str, default='data/path_integration_raw_trajectories_1000t_15s_seed13.npz')
parser.add_argument('--load-saved-model', type=str, default='', help='Saved model to load from')
parser.add_argument('--loss-function', type=str, default='mse',
                    choices=['mse', 'cosine', 'combined', 'alternating', 'scaled'])
parser.add_argument('--frozen-model', type=str, default='', help='model to use frozen encoding weights from')
parser.add_argument('--pc-gauss-sigma', type=float, default=0.01)
parser.add_argument('--dropout-p', type=float, default=0.5)
parser.add_argument('--ssp-scaling', type=float, default=1.0)
parser.add_argument('--encoding-dim', type=int, default=512)
parser.add_argument('--train-split', type=float, default=0.8, help='Training fraction of the train/test split')

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
save_dir = os.path.join(args.logdir, current_time)
writer = SummaryWriter(log_dir=save_dir)

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
else:
    raise NotImplementedError

limit_low = 0 * args.ssp_scaling
limit_high = 2.2 * args.ssp_scaling
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

# n_samples = 5000
n_samples = args.n_samples#1000
rollout_length = args.trajectory_length#100
batch_size = args.minibatch_size#10
n_epochs = args.n_epochs#20
# n_epochs = 5

model = SSPPathIntegrationModel(unroll_length=rollout_length, sp_dim=dim, dropout_p=args.dropout_p)

if args.load_saved_model:
    model.load_state_dict(torch.load(args.load_saved_model), strict=False)


# trying out cosine similarity here as well, it works better than MSE as a loss for SSP cleanup
cosine_criterion = nn.CosineEmbeddingLoss()
mse_criterion = nn.MSELoss()

optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

print("Generating Train and Test Loaders")

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

print("Train and Test Loaders Generation Complete")

params = vars(args)
with open(os.path.join(save_dir, "params.json"), "w") as f:
    json.dump(params, f)

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

                ssp_pred = model(velocity_inputs, ssp_inputs)

                # NOTE: need to permute axes of the targets here because the output is
                #       (sequence length, batch, units) instead of (batch, sequence_length, units)
                #       could also permute the outputs instead
                cosine_loss = cosine_criterion(
                    ssp_pred.reshape(ssp_pred.shape[0] * ssp_pred.shape[1], ssp_pred.shape[2]),
                    ssp_outputs.permute(1, 0, 2).reshape(ssp_pred.shape[0] * ssp_pred.shape[1], ssp_pred.shape[2]),
                    torch.ones(ssp_pred.shape[0] * ssp_pred.shape[1])
                )
                mse_loss = mse_criterion(ssp_pred, ssp_outputs.permute(1, 0, 2))

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

            if args.encoding == 'ssp':
                print("computing prediction locations")
                predictions_start[:, :] = ssp_to_loc_v(
                    ssp_pred.detach().numpy()[0, :, :],
                    heatmap_vectors, xs, ys
                )
                predictions_end[:, :] = ssp_to_loc_v(
                    ssp_pred.detach().numpy()[-1, :, :],
                    heatmap_vectors, xs, ys
                )
                print("computing ground truth locations")
                coords_start[:, :] = ssp_to_loc_v(
                    ssp_outputs.detach().numpy()[:, 0, :],
                    heatmap_vectors, xs, ys
                )
                coords_end[:, :] = ssp_to_loc_v(
                    ssp_outputs.detach().numpy()[:, -1, :],
                    heatmap_vectors, xs, ys
                )
            elif args.encoding == '2d':
                print("copying prediction locations")
                predictions_start[:, :] = ssp_pred.detach().numpy()[0, :, :]
                predictions_end[:, :] = ssp_pred.detach().numpy()[-1, :, :]
                print("copying ground truth locations")
                coords_start[:, :] = ssp_outputs.detach().numpy()[:, 0, :]
                coords_end[:, :] = ssp_outputs.detach().numpy()[:, -1, :]
            else:
                # normalizing is important here
                print("computing prediction locations")
                pred_start = ssp_pred.detach().numpy()[0, :, :]
                pred_start = pred_start / pred_start.sum(axis=1)[:, np.newaxis]
                predictions_start[:, :] = ssp_to_loc_v(
                    pred_start,
                    heatmap_vectors, xs, ys
                )
                pred_end = ssp_pred.detach().numpy()[-1, :, :]
                pred_end = pred_end / pred_end.sum(axis=1)[:, np.newaxis]
                predictions_end[:, :] = ssp_to_loc_v(
                    pred_end,
                    heatmap_vectors, xs, ys
                )
                print("computing ground truth locations")
                coord_start = ssp_outputs.detach().numpy()[:, 0, :]
                coord_start = coord_start / coord_start.sum(axis=1)[:, np.newaxis]
                coords_start[:, :] = ssp_to_loc_v(
                    coord_start,
                    heatmap_vectors, xs, ys
                )
                coord_end = ssp_outputs.detach().numpy()[:, -1, :]
                coord_end = coord_end / coord_end.sum(axis=1)[:, np.newaxis]
                coords_end[:, :] = ssp_to_loc_v(
                    coord_end,
                    heatmap_vectors, xs, ys
                )

            fig_pred_start, ax_pred_start = plt.subplots()
            fig_truth_start, ax_truth_start = plt.subplots()
            fig_pred_end, ax_pred_end = plt.subplots()
            fig_truth_end, ax_truth_end = plt.subplots()

            print("plotting predicted locations")
            plot_predictions_v(predictions_start / args.ssp_scaling, coords_start / args.ssp_scaling, ax_pred_start, min_val=0, max_val=2.2, fixed_axes=True)
            plot_predictions_v(predictions_end / args.ssp_scaling, coords_end / args.ssp_scaling, ax_pred_end, min_val=0, max_val=2.2, fixed_axes=True)
            print("plotting ground truth locations")
            plot_predictions_v(coords_start / args.ssp_scaling, coords_start / args.ssp_scaling, ax_truth_start, min_val=0, max_val=2.2, fixed_axes=True)
            plot_predictions_v(coords_end / args.ssp_scaling, coords_end / args.ssp_scaling, ax_truth_end, min_val=0, max_val=2.2, fixed_axes=True)

            writer.add_figure("predictions start", fig_pred_start, epoch)
            writer.add_figure("ground truth start", fig_truth_start, epoch)

            writer.add_figure("predictions end", fig_pred_end, epoch)
            writer.add_figure("ground truth end", fig_truth_end, epoch)

            torch.save(
                model.state_dict(),
                os.path.join(save_dir, '{}_path_integration_model_epoch_{}.pt'.format(args.encoding, epoch))
            )

    avg_bce_loss = 0
    avg_cosine_loss = 0
    avg_mse_loss = 0
    avg_combined_loss = 0
    avg_scaled_loss = 0
    n_batches = 0
    for i, data in enumerate(trainloader):
        velocity_inputs, ssp_inputs, ssp_outputs = data

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
            torch.ones(ssp_pred.shape[0] * ssp_pred.shape[1])
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
            torch.ones(ssp_pred.shape[0] * ssp_pred.shape[1])
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

    if args.encoding == 'ssp':
        print("computing prediction locations")
        predictions_start[:, :] = ssp_to_loc_v(
            ssp_pred.detach().numpy()[0, :, :],
            heatmap_vectors, xs, ys
        )
        predictions_end[:, :] = ssp_to_loc_v(
            ssp_pred.detach().numpy()[-1, :, :],
            heatmap_vectors, xs, ys
        )
        print("computing ground truth locations")
        coords_start[:, :] = ssp_to_loc_v(
            ssp_outputs.detach().numpy()[:, 0, :],
            heatmap_vectors, xs, ys
        )
        coords_end[:, :] = ssp_to_loc_v(
            ssp_outputs.detach().numpy()[:, -1, :],
            heatmap_vectors, xs, ys
        )
    elif args.encoding == '2d':
        print("copying prediction locations")
        predictions_start[:, :] = ssp_pred.detach().numpy()[0, :, :]
        predictions_end[:, :] = ssp_pred.detach().numpy()[-1, :, :]
        print("copying ground truth locations")
        coords_start[:, :] = ssp_outputs.detach().numpy()[:, 0, :]
        coords_end[:, :] = ssp_outputs.detach().numpy()[:, -1, :]
    else:
        # normalizing is important here
        print("computing prediction locations")
        pred_start = ssp_pred.detach().numpy()[0, :, :]
        pred_start = pred_start / pred_start.sum(axis=1)[:, np.newaxis]
        predictions_start[:, :] = ssp_to_loc_v(
            pred_start,
            heatmap_vectors, xs, ys
        )
        pred_end = ssp_pred.detach().numpy()[-1, :, :]
        pred_end = pred_end / pred_end.sum(axis=1)[:, np.newaxis]
        predictions_end[:, :] = ssp_to_loc_v(
            pred_end,
            heatmap_vectors, xs, ys
        )
        print("computing ground truth locations")
        coord_start = ssp_outputs.detach().numpy()[:, 0, :]
        coord_start = coord_start / coord_start.sum(axis=1)[:, np.newaxis]
        coords_start[:, :] = ssp_to_loc_v(
            coord_start,
            heatmap_vectors, xs, ys
        )
        coord_end = ssp_outputs.detach().numpy()[:, -1, :]
        coord_end = coord_end / coord_end.sum(axis=1)[:, np.newaxis]
        coords_end[:, :] = ssp_to_loc_v(
            coord_end,
            heatmap_vectors, xs, ys
        )

    fig_pred_start, ax_pred_start = plt.subplots()
    fig_truth_start, ax_truth_start = plt.subplots()
    fig_pred_end, ax_pred_end = plt.subplots()
    fig_truth_end, ax_truth_end = plt.subplots()

    print("plotting predicted locations")
    plot_predictions_v(predictions_start / args.ssp_scaling, coords_start / args.ssp_scaling, ax_pred_start, min_val=0, max_val=2.2, fixed_axes=True)
    plot_predictions_v(predictions_end / args.ssp_scaling, coords_end / args.ssp_scaling, ax_pred_end, min_val=0, max_val=2.2, fixed_axes=True)
    print("plotting ground truth locations")
    plot_predictions_v(coords_start / args.ssp_scaling, coords_start / args.ssp_scaling, ax_truth_start, min_val=0, max_val=2.2, fixed_axes=True)
    plot_predictions_v(coords_end / args.ssp_scaling, coords_end / args.ssp_scaling, ax_truth_end, min_val=0, max_val=2.2, fixed_axes=True)

    writer.add_figure("final predictions start", fig_pred_start)
    writer.add_figure("final ground truth start", fig_truth_start)

    writer.add_figure("final predictions end", fig_pred_end)
    writer.add_figure("final ground truth end", fig_truth_end)

torch.save(model.state_dict(), os.path.join(save_dir, '{}_path_integration_model.pt'.format(args.encoding)))
