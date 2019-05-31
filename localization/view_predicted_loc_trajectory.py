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
from datetime import datetime
from tensorboardX import SummaryWriter
from spatial_semantic_pointers.utils import get_heatmap_vectors, ssp_to_loc, ssp_to_loc_v
import matplotlib.pyplot as plt
from localization_training_utils import TrajectoryValidationSet, localization_train_test_loaders, LocalizationModel, pc_to_loc_v

parser = argparse.ArgumentParser('Run 2D supervised path integration experiment using pytorch')

parser = add_parameters(parser)

parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--n-samples', type=int, default=1)
parser.add_argument('--dataset', type=str, default='data/localization_trajectories_5m_200t_250s_seed13.npz')
parser.add_argument('--model', type=str, default='output/ssp_trajectory_localization/May13_16-00-27/ssp_trajectory_localization_model.pt', help='Saved model to load from')
parser.add_argument('--encoding', type=str, default='ssp', choices=['ssp', '2d'])

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

data = np.load(args.dataset)

x_axis_vec = data['x_axis_vec']
y_axis_vec = data['y_axis_vec']
ssp_scaling = data['ssp_scaling']
ssp_offset = data['ssp_offset']


n_samples = args.n_samples#1000
rollout_length = args.trajectory_length#100
batch_size = 1

# shape of coarse maps is (n_maps, env_size, env_size)
coarse_maps = data['coarse_maps']
n_maps = coarse_maps.shape[0]
env_size = coarse_maps.shape[1]

# shape of dist_sensors is (n_maps, n_trajectories, n_steps, n_sensors)
n_sensors = data['dist_sensors'].shape[3]

# shape of ssps is (n_maps, n_trajectories, n_steps, dim)
dim = data['ssps'].shape[3]

limit_low = -ssp_offset * ssp_scaling
limit_high = (env_size - ssp_offset) * ssp_scaling
res = 256

print("ssp limits")
print(limit_low, limit_high)

xs = np.linspace(limit_low, limit_high, res)
ys = np.linspace(limit_low, limit_high, res)

# Used for visualization of test set performance using pos = ssp_to_loc(sp, heatmap_vectors, xs, ys)
heatmap_vectors = get_heatmap_vectors(xs, ys, x_axis_vec, y_axis_vec)

model = LocalizationModel(
    input_size=2 + n_sensors + n_maps,
    unroll_length=rollout_length,
    sp_dim=dim
)

model.load_state_dict(torch.load(args.model), strict=False)


trainloader, testloader = localization_train_test_loaders(
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
        combined_inputs, ssp_inputs, ssp_outputs = data

        # ssp_pred, lstm_outputs = model.forward_activations(velocity_inputs, ssp_inputs)
        ssp_pred = model(combined_inputs, ssp_inputs)


    print("ssp_pred.shape", ssp_pred.shape)
    print("ssp_outputs.shape", ssp_outputs.shape)
    # print("lstm_outputs.shape", lstm_outputs.shape)

    # predictions = np.zeros((ssp_pred.shape[1]*ssp_pred.shape[2], 2))
    # coords = np.zeros((ssp_pred.shape[1]*ssp_pred.shape[2], 2))
    predictions = np.zeros((ssp_pred.shape[0]*ssp_pred.shape[1], 2))
    coords = np.zeros((ssp_pred.shape[0]*ssp_pred.shape[1], 2))

    print("Computing predicted locations and true locations")
    # Using all data, one chunk at a time
    for ri in range(rollout_length):

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
