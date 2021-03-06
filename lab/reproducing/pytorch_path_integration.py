import matplotlib
import os
# allow code to work on machines without a display or in a screen session
display = os.environ.get('DISPLAY')
if display is None or 'localhost' in display:
    matplotlib.use('agg')
import matplotlib.pyplot as plt

import argparse
import numpy as np
from arguments import add_parameters
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from trajectory_dataset import train_test_loaders
from pytorch_models import PathIntegrationModel
import os
from datetime import datetime
from tensorboardX import SummaryWriter
import json
from utils import pc_to_loc_v, hd_to_ang_v
from spatial_semantic_pointers.plots import plot_predictions_v


parser = argparse.ArgumentParser('Run 2D supervised path integration experiment using pytorch')

parser = add_parameters(parser)

parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--n-epochs', type=int, default=20)
parser.add_argument('--n-samples', type=int, default=1000)
parser.add_argument('--logdir', type=str, default='output/pytorch_path_integration',
                    help='Directory for saved model and tensorboard log')
parser.add_argument('--dataset', type=str, default='data/path_integration_trajectories_logits_200t_15s.npz')
parser.add_argument('--load-saved-model', type=str, default='', help='Saved model to load from')

args = parser.parse_args()

torch.manual_seed(args.seed)

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
save_dir = os.path.join(args.logdir, current_time)
writer = SummaryWriter(log_dir=save_dir)

data = np.load(args.dataset)

pc_centers = data['pc_centers']
hd_centers = data['hd_centers']

# n_samples = 5000
n_samples = args.n_samples#1000
rollout_length = args.trajectory_length#100
batch_size = args.minibatch_size#10
n_epochs = args.n_epochs#20
# n_epochs = 5

model = PathIntegrationModel(unroll_length=rollout_length)

if args.load_saved_model:
    model.load_state_dict(torch.load(args.load_saved_model), strict=False)

# Binary cross-entropy loss
# criterion = nn.BCELoss()
# more numerically stable. Do not use softmax beforehand
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

trainloader, testloader = train_test_loaders(
    data,
    n_train_samples=n_samples,
    n_test_samples=n_samples,
    rollout_length=rollout_length,
    batch_size=batch_size
)

print("Training")
for epoch in range(n_epochs):
    print("Epoch {} of {}".format(epoch + 1, n_epochs))

    if epoch % 100 == 0:
        print("Evaluation")
        with torch.no_grad():
            # Everything is in one batch, so this loop will only happen once
            for i, data in enumerate(testloader):
                velocity_inputs, pc_inputs, hd_inputs, pc_outputs, hd_outputs = data

                pc_pred, hd_pred = model(velocity_inputs, pc_inputs, hd_inputs)

                # NOTE: need to permute axes of the targets here because the output is
                #       (sequence length, batch, units) instead of (batch, sequence_length, units)
                #       could also permute the outputs instead
                loss = criterion(pc_pred, F.softmax(pc_outputs.permute(1, 0, 2), dim=2)) + criterion(hd_pred, F.softmax(
                    hd_outputs.permute(1, 0, 2), dim=2))
                # loss = criterion(pc_pred, pc_outputs.permute(1, 0, 2)) + criterion(hd_pred, hd_outputs.permute(1, 0, 2))

                print("test loss", loss.data.item())

            writer.add_scalar('test_loss', loss.data.item(), epoch)

            # Just use start and end location to save on memory and computation
            pc_predictions_start = np.zeros((pc_pred.shape[1], 2))
            pc_coords_start = np.zeros((pc_pred.shape[1], 2))
            hd_predictions_start = np.zeros((hd_pred.shape[1], 2))
            hd_coords_start = np.zeros((hd_pred.shape[1], 2))

            pc_predictions_end = np.zeros((pc_pred.shape[1], 2))
            pc_coords_end = np.zeros((pc_pred.shape[1], 2))
            hd_predictions_end = np.zeros((hd_pred.shape[1], 2))
            hd_coords_end = np.zeros((hd_pred.shape[1], 2))

            print("computing prediction locations")
            pc_predictions_start[:, :] = pc_to_loc_v(
                pc_pred.detach().numpy()[0, :, :],
                centers=pc_centers,
                jitter=0.01
            )
            pc_predictions_end[:, :] = pc_to_loc_v(
                pc_pred.detach().numpy()[-1, :, :],
                centers=pc_centers,
                jitter=0.01
            )
            hd_predictions_start[:, :] = hd_to_ang_v(
                hd_pred.detach().numpy()[0, :, :],
                centers=hd_centers,
                jitter=0.01
            )
            hd_predictions_end[:, :] = hd_to_ang_v(
                hd_pred.detach().numpy()[-1, :, :],
                centers=hd_centers,
                jitter=0.01
            )

            pc_outputs_sm = F.softmax(pc_outputs.permute(1, 0, 2), dim=2)
            hd_outputs_sm = F.softmax(hd_outputs.permute(1, 0, 2), dim=2)

            print("computing ground truth locations")
            pc_coords_start[:, :] = pc_to_loc_v(
                pc_outputs_sm.detach().numpy()[0, :, :],
                centers=pc_centers,
                jitter=0.01
            )
            pc_coords_end[:, :] = pc_to_loc_v(
                pc_outputs_sm.detach().numpy()[-1, :, :],
                centers=pc_centers,
                jitter=0.01
            )
            hd_coords_start[:, :] = hd_to_ang_v(
                hd_outputs_sm.detach().numpy()[0, :, :],
                centers=hd_centers,
                jitter=0.01
            )
            hd_coords_end[:, :] = hd_to_ang_v(
                hd_outputs_sm.detach().numpy()[-1, :, :],
                centers=hd_centers,
                jitter=0.01
            )

            fig_pc_pred_start, ax_pc_pred_start = plt.subplots()
            fig_pc_truth_start, ax_pc_truth_start = plt.subplots()
            fig_pc_pred_end, ax_pc_pred_end = plt.subplots()
            fig_pc_truth_end, ax_pc_truth_end = plt.subplots()
            fig_hd_pred_start, ax_hd_pred_start = plt.subplots()
            fig_hd_truth_start, ax_hd_truth_start = plt.subplots()
            fig_hd_pred_end, ax_hd_pred_end = plt.subplots()
            fig_hd_truth_end, ax_hd_truth_end = plt.subplots()

            print("plotting predicted locations")
            plot_predictions_v(pc_predictions_start, pc_coords_start, ax_pc_pred_start, min_val=0, max_val=2.2)
            plot_predictions_v(pc_predictions_end, pc_coords_end, ax_pc_pred_end, min_val=0, max_val=2.2)
            plot_predictions_v(hd_predictions_start, hd_coords_start, ax_hd_pred_start, min_val=-1, max_val=1)
            plot_predictions_v(hd_predictions_end, hd_coords_end, ax_hd_pred_end, min_val=-1, max_val=1)

            writer.add_figure("pc predictions start", fig_pc_pred_start, epoch)
            writer.add_figure("pc predictions end", fig_pc_pred_end, epoch)
            writer.add_figure("hd predictions start", fig_hd_pred_start, epoch)
            writer.add_figure("hd predictions end", fig_hd_pred_end, epoch)

    avg_loss = 0
    n_batches = 0
    for i, data in enumerate(trainloader):
        velocity_inputs, pc_inputs, hd_inputs, pc_outputs, hd_outputs = data

        if pc_inputs.size()[0] != batch_size:
            continue  # Drop data, not enough for a batch
        optimizer.zero_grad()
        # model.zero_grad()

        # pc_pred, hd_pred = model(velocity_inputs[i, :, :], pc_inputs[i, :], hd_inputs[i, :])
        pc_pred, hd_pred = model(velocity_inputs, pc_inputs, hd_inputs)

        # loss = criterion(pc_pred, pc_outputs[i, :]) + criterion(hd_pred, hd_outputs[i, :])

        # print("pc_pred.shape", pc_pred.shape)
        # print("pc_outputs.shape", pc_outputs.shape)
        # print("hd_pred.shape", hd_pred.shape)
        # print("hd_outputs.shape", hd_outputs.shape)

        # NOTE: need to permute axes of the targets here because the output is
        #       (sequence length, batch, units) instead of (batch, sequence_length, units)
        #       could also permute the outputs instead
        loss = criterion(pc_pred, F.softmax(pc_outputs.permute(1, 0, 2), dim=2)) + criterion(hd_pred, F.softmax(hd_outputs.permute(1, 0, 2), dim=2))
        # loss = criterion(pc_pred, pc_outputs.permute(1, 0, 2)) + criterion(hd_pred, hd_outputs.permute(1, 0, 2))
        loss.backward()

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
        velocity_inputs, pc_inputs, hd_inputs, pc_outputs, hd_outputs = data

        pc_pred, hd_pred = model(velocity_inputs, pc_inputs, hd_inputs)

        # NOTE: need to permute axes of the targets here because the output is
        #       (sequence length, batch, units) instead of (batch, sequence_length, units)
        #       could also permute the outputs instead
        loss = criterion(pc_pred, F.softmax(pc_outputs.permute(1, 0, 2), dim=2)) + criterion(hd_pred, F.softmax(hd_outputs.permute(1, 0, 2), dim=2))
        # loss = criterion(pc_pred, pc_outputs.permute(1, 0, 2)) + criterion(hd_pred, hd_outputs.permute(1, 0, 2))

        print("test loss", loss.data.item())

    writer.add_scalar('final_test_loss', loss.data.item())

    # Just use start and end location to save on memory and computation
    pc_predictions_start = np.zeros((pc_pred.shape[1], 2))
    pc_coords_start = np.zeros((pc_pred.shape[1], 2))
    hd_predictions_start = np.zeros((hd_pred.shape[1], 2))
    hd_coords_start = np.zeros((hd_pred.shape[1], 2))

    pc_predictions_end = np.zeros((pc_pred.shape[1], 2))
    pc_coords_end = np.zeros((pc_pred.shape[1], 2))
    hd_predictions_end = np.zeros((hd_pred.shape[1], 2))
    hd_coords_end = np.zeros((hd_pred.shape[1], 2))

    print("computing prediction locations")
    pc_predictions_start[:, :] = pc_to_loc_v(
        pc_pred.detach().numpy()[0, :, :],
        centers=pc_centers,
        jitter=0.01
    )
    pc_predictions_end[:, :] = pc_to_loc_v(
        pc_pred.detach().numpy()[-1, :, :],
        centers=pc_centers,
        jitter=0.01
    )
    hd_predictions_start[:, :] = hd_to_ang_v(
        hd_pred.detach().numpy()[0, :, :],
        centers=hd_centers,
        jitter=0.01
    )
    hd_predictions_end[:, :] = hd_to_ang_v(
        hd_pred.detach().numpy()[-1, :, :],
        centers=hd_centers,
        jitter=0.01
    )

    pc_outputs_sm = F.softmax(pc_outputs.permute(1, 0, 2), dim=2)
    hd_outputs_sm = F.softmax(hd_outputs.permute(1, 0, 2), dim=2)

    print("computing ground truth locations")
    pc_coords_start[:, :] = pc_to_loc_v(
        pc_outputs_sm.detach().numpy()[0, :, :],
        centers=pc_centers,
        jitter=0.01
    )
    pc_coords_end[:, :] = pc_to_loc_v(
        pc_outputs_sm.detach().numpy()[-1, :, :],
        centers=pc_centers,
        jitter=0.01
    )
    hd_coords_start[:, :] = hd_to_ang_v(
        hd_outputs_sm.detach().numpy()[0, :, :],
        centers=hd_centers,
        jitter=0.01
    )
    hd_coords_end[:, :] = hd_to_ang_v(
        hd_outputs_sm.detach().numpy()[-1, :, :],
        centers=hd_centers,
        jitter=0.01
    )

    fig_pc_pred_start, ax_pc_pred_start = plt.subplots()
    fig_pc_truth_start, ax_pc_truth_start = plt.subplots()
    fig_pc_pred_end, ax_pc_pred_end = plt.subplots()
    fig_pc_truth_end, ax_pc_truth_end = plt.subplots()
    fig_hd_pred_start, ax_hd_pred_start = plt.subplots()
    fig_hd_truth_start, ax_hd_truth_start = plt.subplots()
    fig_hd_pred_end, ax_hd_pred_end = plt.subplots()
    fig_hd_truth_end, ax_hd_truth_end = plt.subplots()

    print("plotting predicted locations")
    plot_predictions_v(pc_predictions_start, pc_coords_start, ax_pc_pred_start, min_val=0, max_val=2.2)
    plot_predictions_v(pc_predictions_end, pc_coords_end, ax_pc_pred_end, min_val=0, max_val=2.2)
    plot_predictions_v(hd_predictions_start, hd_coords_start, ax_hd_pred_start, min_val=-1, max_val=1)
    plot_predictions_v(hd_predictions_end, hd_coords_end, ax_hd_pred_end, min_val=-1, max_val=1)
    print("plotting ground truth locations")
    plot_predictions_v(pc_coords_start, pc_coords_start, ax_pc_truth_start, min_val=0, max_val=2.2)
    plot_predictions_v(pc_coords_end, pc_coords_end, ax_pc_truth_end, min_val=0, max_val=2.2)
    plot_predictions_v(hd_coords_start, hd_coords_start, ax_hd_truth_start, min_val=-1, max_val=1)
    plot_predictions_v(hd_coords_end, hd_coords_end, ax_hd_truth_end, min_val=-1, max_val=1)

    writer.add_figure("pc predictions start", fig_pc_pred_start, epoch)
    writer.add_figure("pc predictions end", fig_pc_pred_end, epoch)
    writer.add_figure("hd predictions start", fig_hd_pred_start, epoch)
    writer.add_figure("hd predictions end", fig_hd_pred_end, epoch)

    writer.add_figure("pc ground truth start", fig_pc_truth_start)
    writer.add_figure("pc ground truth end", fig_pc_truth_end)
    writer.add_figure("hd ground truth start", fig_hd_truth_start)
    writer.add_figure("hd ground truth end", fig_hd_truth_end)

torch.save(model.state_dict(), os.path.join(save_dir, 'path_integration_model.pt'))

params = vars(args)
with open(os.path.join(save_dir, "params.json"), "w") as f:
    json.dump(params, f)
