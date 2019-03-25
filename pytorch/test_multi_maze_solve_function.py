import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import json
from tensorboardX import SummaryWriter
from datetime import datetime
from spatial_semantic_pointers.utils import make_good_unitary, encode_point
from path_utils import plot_path_predictions, generate_maze_sp, solve_maze
from models import FeedForward
from datasets import MazeDataset
import nengo.spa as spa
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(
    'Test a function that given a maze and a goal location, computes the direction to move to get to that goal'
)

parser.add_argument('--seed', type=int, default=13, help='Seed for training and generating axis SSPs')
parser.add_argument('--limit-low', type=float, default=-5, help='lowest coordinate value')
parser.add_argument('--limit-high', type=float, default=5, help='highest coordinate value')
parser.add_argument('--view-activations', action='store_true', help='view spatial activations of each neuron')
parser.add_argument('--dataset', type=str, default='maze_datasets/maze_dataset_10mazes_25goals_64res_13seed.npz')
parser.add_argument('--logdir', type=str, default='test_multi_maze_solve_function',
                    help='Directory for saved model and tensorboard log')
parser.add_argument('--load-saved-model', type=str, default='', help='Saved model to load from')

args = parser.parse_args()

assert(args.limit_low < args.limit_high)

data = np.load(args.dataset)

rng = np.random.RandomState(seed=args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

x_axis_sp = spa.SemanticPointer(data=data['x_axis_sp'])
y_axis_sp = spa.SemanticPointer(data=data['y_axis_sp'])


# n_mazes by size by size
coarse_mazes = data['coarse_mazes']

# n_mazes by res by res
fine_mazes = data['fine_mazes']

# n_mazes by res by res by 2
solved_mazes = data['solved_mazes']

# n_mazes by dim
maze_sps = data['maze_sps']

# n_mazes by n_goals by dim
goal_sps = data['goal_sps']

# n_mazes by n_goals by 2
goals = data['goals']

n_goals = goals.shape[1]

if 'xs' in data.keys():
    xs = data['xs']
    ys = data['ys']
else:
    # backwards compatibility
    xs = np.linspace(args.limit_low, args.limit_high, args.res)
    ys = np.linspace(args.limit_low, args.limit_high, args.res)


n_mazes = goals.shape[0]
n_goals = goals.shape[1]
res = fine_mazes.shape[1]
dim = goal_sps.shape[2]
n_samples = res * res * n_mazes

# Visualization
viz_locs = np.zeros((n_samples, 2))
viz_goals = np.zeros((n_samples, 2))
viz_loc_sps = np.zeros((n_samples, dim))
viz_goal_sps = np.zeros((n_samples, dim))
viz_output_dirs = np.zeros((n_samples, 2))
viz_maze_sps = np.zeros((n_samples, dim))


# For each maze, just choose the first goal to visualize
gi = 0  # goal index, stays constant
si = 0  # sample index, increments each time
for mi in range(n_mazes):
    for xi in range(res):
        for yi in range(res):
            loc_x = xs[xi]
            loc_y = ys[yi]

            viz_locs[si, 0] = loc_x
            viz_locs[si, 1] = loc_y
            viz_goals[si, :] = goals[mi, gi, :]
            viz_loc_sps[si, :] = encode_point(loc_x, loc_y, x_axis_sp, y_axis_sp).v
            viz_goal_sps[si, :] = goal_sps[mi, gi, :]

            viz_output_dirs[si, :] = solved_mazes[mi, gi, xi, yi, :]

            viz_maze_sps[si, :] = maze_sps[mi]

            si += 1


dataset_viz = MazeDataset(
    maze_ssp=viz_maze_sps,
    loc_ssps=viz_loc_sps,
    goal_ssps=viz_goal_sps,
    locs=viz_locs,
    goals=viz_goals,
    direction_outputs=viz_output_dirs,
)

# Each batch will contain the samples for one maze. Must not be shuffled
vizloader = torch.utils.data.DataLoader(
    dataset_viz, batch_size=res*res, shuffle=False, num_workers=0,
)

# input is maze, loc, goal ssps, output is 2D direction to move
model = FeedForward(input_size=dim * 3, output_size=2)

if args.load_saved_model:
    model.load_state_dict(torch.load(args.load_saved_model), strict=False)

model.eval()

# Open a tensorboard writer if a logging directory is given
if args.logdir != '':
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    save_dir = os.path.join(args.logdir, current_time)
    writer = SummaryWriter(log_dir=save_dir)

criterion = nn.MSELoss()

print("Visualization")
with torch.no_grad():
    # Each maze is in one batch
    for i, data in enumerate(vizloader):
        maze_loc_goal_ssps, directions, locs, goals = data

        if args.view_activations:
            outputs, activations = model.forward_activations(maze_loc_goal_ssps)
            # shape of activations is batch_size by n_neurons

            # Loop through each neuron to create a spatial map
            for n in range(activations.shape[1]):
                fig, ax = plt.subplots()
                # Note that this works because the batch contains one example for each location
                # in a res by res grid and is ordered and not shuffled.
                ax.imshow(activations[:, n].view(res, res))
                writer.add_figure('activations for batch {}'.format(i), fig, n)

                fig, ax = plt.subplots()
                ax.imshow(maze_loc_goal_ssps[:, dim + n].view(res, res))
                writer.add_figure('loc inputs for batch {}'.format(i), fig, n)
        else:
            outputs = model(maze_loc_goal_ssps)

        loss = criterion(outputs, directions)

        print(loss.data.item())

        if args.logdir != '':
            fig_pred = plot_path_predictions(
                directions=outputs, coords=locs, type='colour'
            )
            writer.add_figure('viz set predictions', fig_pred, i)
            fig_truth = plot_path_predictions(
                directions=directions, coords=locs, type='colour'
            )
            writer.add_figure('ground truth', fig_truth, i)

            fig_pred_quiver = plot_path_predictions(
                directions=outputs, coords=locs, dcell=xs[1] - xs[0]
            )
            writer.add_figure('viz set predictions quiver', fig_pred_quiver, i)
            fig_truth_quiver = plot_path_predictions(
                directions=directions, coords=locs, dcell=xs[1] - xs[0]
            )
            writer.add_figure('ground truth quiver', fig_truth_quiver, i)

            writer.add_scalar('viz_loss', loss.data.item(), i)
