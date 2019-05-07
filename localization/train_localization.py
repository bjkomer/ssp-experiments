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
from datasets import localization_train_test_loaders
from models import SSPPathIntegrationModel
from datetime import datetime
from tensorboardX import SummaryWriter
import json
from spatial_semantic_pointers.utils import get_heatmap_vectors, ssp_to_loc, ssp_to_loc_v
from spatial_semantic_pointers.plots import plot_predictions, plot_predictions_v
import matplotlib.pyplot as plt
from path_integration_utils import pc_to_loc_v


parser = argparse.ArgumentParser('Run 2D supervised localization experiment using pytorch')

parser = add_parameters(parser)

parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--n-epochs', type=int, default=20)
parser.add_argument('--n-samples', type=int, default=1000)
parser.add_argument('--encoding', type=str, default='ssp', choices=['ssp', '2d', 'pc'])
parser.add_argument('--eval-period', type=int, default=50)
parser.add_argument('--logdir', type=str, default='output/ssp_localization',
                    help='Directory for saved model and tensorboard log')
# TODO: update default to use dataset with distance sensor measurements (or boundary cell activations)
parser.add_argument('--dataset', type=str, default='../lab/reproducing/data/path_integration_trajectories_logits_200t_15s_seed13.npz')
parser.add_argument('--load-saved-model', type=str, default='', help='Saved model to load from')
parser.add_argument('--use-cosine-loss', action='store_true')

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.use_cosine_loss:
    print("Using Cosine Loss")
else:
    print("Using MSE Loss")

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
save_dir = os.path.join(args.logdir, current_time)
writer = SummaryWriter(log_dir=save_dir)

data = np.load(args.dataset)

x_axis_vec = data['x_axis_vec']
y_axis_vec = data['y_axis_vec']
ssp_scaling = data['ssp_scaling']

pc_centers = data['pc_centers']
pc_activations = data['pc_activations']

n_sensors = data['n_sensors']

if args.encoding == 'ssp':
    dim = 512
elif args.encoding == '2d':
    dim = 2
    ssp_scaling = 1  # no scaling used for 2D coordinates directly
elif args.encoding == 'pc':
    dim = args.n_place_cells
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
n_epochs = args.n_epochs#20
# n_epochs = 5



# Input is x and y velocity plus the distance sensor measurements
model = SSPPathIntegrationModel(
    inputs_size=2 + n_sensors,
    unroll_length=rollout_length,
    sp_dim=dim
)

if args.load_saved_model:
    model.load_state_dict(torch.load(args.load_saved_model), strict=False)

if args.encoding == 'pc':
    # Binary cross-entropy loss
    # criterion = nn.BCELoss()
    # more numerically stable. Do not use softmax beforehand
    criterion = nn.BCEWithLogitsLoss()
else:
    cosine_criterion = nn.CosineEmbeddingLoss()
    mse_criterion = nn.MSELoss()

optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

trainloader, testloader = localization_train_test_loaders(
    data,
    n_train_samples=n_samples,
    n_test_samples=n_samples,
    rollout_length=rollout_length,
    batch_size=batch_size,
    encoding=args.encoding,
)

params = vars(args)
with open(os.path.join(save_dir, "params.json"), "w") as f:
    json.dump(params, f)

print("Training")
for epoch in range(n_epochs):
    print("Epoch {} of {}".format(epoch + 1, n_epochs))


    # TODO: modularize this and clean it up ++
    # Every 'eval_period' epochs, create a test loss and image
    if epoch % args.eval_period == 0:
        print("Evaluating")
        with torch.no_grad():
            # Everything is in one batch, so this loop will only happen once
            for i, data in enumerate(testloader):
                velocity_inputs, ssp_inputs, ssp_outputs = data

                ssp_pred = model(velocity_inputs, ssp_inputs)

                # NOTE: need to permute axes of the targets here because the output is
                #       (sequence length, batch, units) instead of (batch, sequence_length, units)
                #       could also permute the outputs instead
                if args.encoding == 'pc':
                    # place cell version needs to explicitly do the softmax here
                    loss = criterion(ssp_pred, F.softmax(ssp_outputs.permute(1, 0, 2), dim=2))
                else:
                    cosine_loss = criterion(ssp_pred, ssp_outputs.permute(1, 0, 2),
                                     torch.ones(ssp_pred.shape[0], ssp_pred.shape[0]))
                    mse_loss = criterion(ssp_pred, ssp_outputs.permute(1, 0, 2))

                print("test loss", loss.data.item())

            writer.add_scalar('test_loss', loss.data.item(), epoch)

            print("ssp_pred.shape", ssp_pred.shape)
            print("ssp_outputs.shape", ssp_outputs.shape)

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
            elif args.encoding == 'pc':
                # (quick hack is to just use the most activated place cell center)
                predictions_start[:, :] = pc_to_loc_v(
                    pc_activations=ssp_pred.detach().numpy()[0, :, :],
                    centers=pc_centers,
                    jitter=0.01
                )
                predictions_end[:, :] = pc_to_loc_v(
                    pc_activations=ssp_pred.detach().numpy()[-1, :, :],
                    centers=pc_centers,
                    jitter=0.01
                )

                coords_start[:, :] = pc_to_loc_v(
                    pc_activations=ssp_outputs.detach().numpy()[:, 0, :],
                    centers=pc_centers,
                    jitter=0.01
                )
                coords_end[:, :] = pc_to_loc_v(
                    pc_activations=ssp_outputs.detach().numpy()[:, -1, :],
                    centers=pc_centers,
                    jitter=0.01
                )

            fig_pred_start, ax_pred_start = plt.subplots()
            fig_truth_start, ax_truth_start = plt.subplots()
            fig_pred_end, ax_pred_end = plt.subplots()
            fig_truth_end, ax_truth_end = plt.subplots()

            print("plotting predicted locations")
            plot_predictions_v(predictions_start / ssp_scaling, coords_start / ssp_scaling, ax_pred_start, min_val=0, max_val=2.2)
            plot_predictions_v(predictions_end / ssp_scaling, coords_end / ssp_scaling, ax_pred_end, min_val=0, max_val=2.2)
            print("plotting ground truth locations")
            plot_predictions_v(coords_start / ssp_scaling, coords_start / ssp_scaling, ax_truth_start, min_val=0, max_val=2.2)
            plot_predictions_v(coords_end / ssp_scaling, coords_end / ssp_scaling, ax_truth_end, min_val=0, max_val=2.2)

            writer.add_figure("predictions start", fig_pred_start, epoch)
            writer.add_figure("ground truth start", fig_truth_start, epoch)

            writer.add_figure("predictions end", fig_pred_end, epoch)
            writer.add_figure("ground truth end", fig_truth_end, epoch)


    avg_loss = 0
    n_batches = 0
    for i, data in enumerate(trainloader):
        velocity_inputs, ssp_inputs, ssp_outputs = data

        if ssp_inputs.size()[0] != batch_size:
            continue  # Drop data, not enough for a batch
        optimizer.zero_grad()
        # model.zero_grad()

        ssp_pred = model(velocity_inputs, sensor_inputs, ssp_inputs)

        # NOTE: need to permute axes of the targets here because the output is
        #       (sequence length, batch, units) instead of (batch, sequence_length, units)
        #       could also permute the outputs instead
        if args.encoding == 'pc':
            # place cell version needs to explicitly do the softmax here
            loss = criterion(ssp_pred, F.softmax(ssp_outputs.permute(1, 0, 2), dim=2))
        else:
            cosine_loss = criterion(ssp_pred, ssp_outputs.permute(1, 0, 2),
                             torch.ones(ssp_pred.shape[0], ssp_pred.shape[0]))
            mse_loss = criterion(ssp_pred, ssp_outputs.permute(1, 0, 2))

        if args.use_cosine_loss:
            cosine_loss.backward()
        else:
            mse_loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_thresh)

        optimizer.step()

        avg_loss += loss.data.item()
        n_batches += 1

    avg_loss /= n_batches
    print("loss:", avg_loss)
    writer.add_scalar('avg_loss', avg_loss, epoch + 1)




print("Testing")
with torch.no_grad():
    # Everything is in one batch, so this loop will only happen once
    for i, data in enumerate(testloader):
        velocity_inputs, ssp_inputs, ssp_outputs = data

        ssp_pred = model(velocity_inputs, ssp_inputs)

        # NOTE: need to permute axes of the targets here because the output is
        #       (sequence length, batch, units) instead of (batch, sequence_length, units)
        #       could also permute the outputs instead
        if args.encoding == 'pc':
            # place cell version needs to explicitly do the softmax here
            loss = criterion(ssp_pred, F.softmax(ssp_outputs.permute(1, 0, 2), dim=2))
        else:
            if args.use_cosine_loss:
                loss = criterion(ssp_pred, ssp_outputs.permute(1, 0, 2),
                                 torch.ones(ssp_pred.shape[0], ssp_pred.shape[0]))
            else:
                loss = criterion(ssp_pred, ssp_outputs.permute(1, 0, 2))

        print("test loss", loss.data.item())

    writer.add_scalar('final_test_loss', loss.data.item())

    print("ssp_pred.shape", ssp_pred.shape)
    print("ssp_outputs.shape", ssp_outputs.shape)

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
    elif args.encoding == 'pc':
        # (quick hack is to just use the most activated place cell center)
        predictions_start[:, :] = pc_to_loc_v(
            pc_activations=ssp_pred.detach().numpy()[0, :, :],
            centers=pc_centers,
            jitter=0.01
        )
        predictions_end[:, :] = pc_to_loc_v(
            pc_activations=ssp_pred.detach().numpy()[-1, :, :],
            centers=pc_centers,
            jitter=0.01
        )

        coords_start[:, :] = pc_to_loc_v(
            pc_activations=ssp_outputs.detach().numpy()[:, 0, :],
            centers=pc_centers,
            jitter=0.01
        )
        coords_end[:, :] = pc_to_loc_v(
            pc_activations=ssp_outputs.detach().numpy()[:, -1, :],
            centers=pc_centers,
            jitter=0.01
        )

    fig_pred_start, ax_pred_start = plt.subplots()
    fig_truth_start, ax_truth_start = plt.subplots()
    fig_pred_end, ax_pred_end = plt.subplots()
    fig_truth_end, ax_truth_end = plt.subplots()

    print("plotting predicted locations")
    plot_predictions_v(predictions_start / ssp_scaling, coords_start / ssp_scaling, ax_pred_start, min_val=0, max_val=2.2)
    plot_predictions_v(predictions_end / ssp_scaling, coords_end / ssp_scaling, ax_pred_end, min_val=0, max_val=2.2)
    print("plotting ground truth locations")
    plot_predictions_v(coords_start / ssp_scaling, coords_start / ssp_scaling, ax_truth_start, min_val=0, max_val=2.2)
    plot_predictions_v(coords_end / ssp_scaling, coords_end / ssp_scaling, ax_truth_end, min_val=0, max_val=2.2)

    writer.add_figure("final predictions start", fig_pred_start)
    writer.add_figure("final ground truth start", fig_truth_start)

    writer.add_figure("final predictions end", fig_pred_end)
    writer.add_figure("final ground truth end", fig_truth_end)



if args.encoding == 'ssp':
    torch.save(model.state_dict(), os.path.join(save_dir, 'ssp_localization_model.pt'))
elif args.encoding == '2d':
    torch.save(model.state_dict(), os.path.join(save_dir, '2d_localization_model.pt'))