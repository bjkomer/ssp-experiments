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
parser.add_argument('--encoding', type=str, default='ssp', choices=['ssp', '2d'])

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

data = np.load(args.dataset)

# shape is (n_maps, n_trajectories, trajectory_steps, 2)
positions = data['positions']
n_maps = positions.shape[0]
n_trajectories = positions.shape[1]
trajectory_steps = positions.shape[2]

fig, ax = plt.subplots(1, 1)

for mi in range(n_maps):
    for ti in range(n_trajectories):

        ax.scatter(
            positions[mi, ti, :, 0],
            positions[mi, ti, :, 1],
            color='blue',
            label='ground truth',
        )

plt.show()
