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

parser = argparse.ArgumentParser(
    'Train a function that given a maze and a goal location, computes the direction to move to get to that goal'
)

parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train for')
parser.add_argument('--seed', type=int, default=13, help='Seed for training and generating axis SSPs')
parser.add_argument('--dim', type=int, default=512, help='Dimensionality of the SSPs')
parser.add_argument('--n-train-samples', type=int, default=1000, help='Number of training samples')
parser.add_argument('--n-test-samples', type=int, default=1000, help='Number of testing samples')
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight-histogram', action='store_true', help='Save histogram of the weights')
parser.add_argument('--maze-type', type=str, default='random', choices=['example', 'random'], help='maze to learn')
parser.add_argument('--maze-size', type=int, default=10, help='Size of the coarse maze structure')
parser.add_argument('--res', type=int, default=64, help='resolution of the fine maze')
parser.add_argument('--limit-low', type=float, default=-5, help='lowest coordinate value')
parser.add_argument('--limit-high', type=float, default=5, help='highest coordinate value')
parser.add_argument('--dataset', type=str, default='maze_datasets/maze_dataset_10mazes_25goals_64res_13seed.npz')
parser.add_argument('--logdir', type=str, default='multi_maze_solve_function',
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



# Get a list of possible start locations to choose (will correspond to all free spaces in all fine mazes)
free_spaces = np.argwhere(fine_mazes == 0)
print(free_spaces.shape)
n_free_spaces = free_spaces.shape[0]

# Training
train_locs = np.zeros((args.n_train_samples, 2))
train_goals = np.zeros((args.n_train_samples, 2))
train_loc_sps = np.zeros((args.n_train_samples, args.dim))
train_goal_sps = np.zeros((args.n_train_samples, args.dim))
train_output_dirs = np.zeros((args.n_train_samples, 2))
train_maze_sps = np.zeros((args.n_train_samples, args.dim))

train_indices = np.random.randint(low=0, high=n_free_spaces, size=args.n_train_samples)

# Testing
test_locs = np.zeros((args.n_test_samples, 2))
test_goals = np.zeros((args.n_test_samples, 2))
test_loc_sps = np.zeros((args.n_test_samples, args.dim))
test_goal_sps = np.zeros((args.n_test_samples, args.dim))
test_output_dirs = np.zeros((args.n_test_samples, 2))
test_maze_sps = np.zeros((args.n_test_samples, args.dim))

test_indices = np.random.randint(low=0, high=n_free_spaces, size=args.n_test_samples)

# Visualization
viz_locs = np.zeros((args.n_test_samples, 2))
viz_goals = np.zeros((args.n_test_samples, 2))
viz_loc_sps = np.zeros((args.n_test_samples, args.dim))
viz_goal_sps = np.zeros((args.n_test_samples, args.dim))
viz_output_dirs = np.zeros((args.n_test_samples, 2))
viz_maze_sps = np.zeros((args.n_test_samples, args.dim))

viz_free_spaces = np.argwhere(fine_mazes[0, :, :] == 0)
print(viz_free_spaces.shape)
n_viz_free_spaces = viz_free_spaces.shape[0]
viz_indices = np.random.randint(low=0, high=n_viz_free_spaces, size=args.n_test_samples)
viz_goal_index = np.random.randint(low=0, high=n_viz_free_spaces)
viz_goal_index = viz_free_spaces[viz_goal_index, :]


for n in range(args.n_train_samples):
    print("Train Sample {} of {}".format(n+1, args.n_train_samples))

    # n_mazes by res by res
    indices = free_spaces[train_indices[n], :]
    maze_index = indices[0]
    x_index = indices[1]
    y_index = indices[2]
    goal_index = np.random.randint(low=0, high=n_goals)

    # 2D coordinate of the agent's current location
    loc_x = xs[x_index]
    loc_y = ys[y_index]

    train_locs[n, 0] = loc_x
    train_locs[n, 1] = loc_y
    train_goals[n, :] = goals[maze_index, goal_index, :]
    train_loc_sps[n, :] = encode_point(loc_x, loc_y, x_axis_sp, y_axis_sp).v
    train_goal_sps[n, :] = goal_sps[maze_index, goal_index, :]

    train_output_dirs[n, :] = solved_mazes[maze_index, goal_index, x_index, y_index, :]

    train_maze_sps[n, :] = maze_sps[maze_index]

for n in range(args.n_test_samples):
    print("Test Sample {} of {}".format(n+1, args.n_test_samples))
    # n_mazes by res by res
    indices = free_spaces[test_indices[n], :]
    maze_index = indices[0]
    x_index = indices[1]
    y_index = indices[2]
    goal_index = np.random.randint(low=0, high=n_goals)

    # 2D coordinate of the agent's current location
    loc_x = xs[x_index]
    loc_y = ys[y_index]

    test_locs[n, 0] = loc_x
    test_locs[n, 1] = loc_y
    test_goals[n, :] = goals[maze_index, goal_index, :]
    test_loc_sps[n, :] = encode_point(loc_x, loc_y, x_axis_sp, y_axis_sp).v
    test_goal_sps[n, :] = goal_sps[maze_index, goal_index, :]

    test_output_dirs[n, :] = solved_mazes[maze_index, goal_index, x_index, y_index, :]

    test_maze_sps[n, :] = maze_sps[maze_index]

# Note: for ease of plotting for testing, would be helpful to just have a single or few goals
#       because of this a separate 'visualization' set will be used
maze_index = 0
goal_index = 0
for n in range(args.n_test_samples):
    print("Viz Sample {} of {}".format(n+1, args.n_test_samples))
    # res by res
    indices = viz_free_spaces[viz_indices[n], :]

    x_index = indices[0]
    y_index = indices[1]

    # 2D coordinate of the agent's current location
    loc_x = xs[x_index]
    loc_y = ys[y_index]

    viz_locs[n, 0] = loc_x
    viz_locs[n, 1] = loc_y
    viz_goals[n, :] = goals[maze_index, goal_index, :]
    viz_loc_sps[n, :] = encode_point(loc_x, loc_y, x_axis_sp, y_axis_sp).v
    viz_goal_sps[n, :] = goal_sps[maze_index, goal_index, :]

    viz_output_dirs[n, :] = solved_mazes[maze_index, goal_index, x_index, y_index, :]

    viz_maze_sps[n, :] = maze_sps[maze_index]


# Reset seeds here after generating data
torch.manual_seed(args.seed)
np.random.seed(args.seed)

#TODO fix to correct datasets
dataset_train = MazeDataset(
    maze_ssp=train_maze_sps,
    loc_ssps=train_loc_sps,
    goal_ssps=train_goal_sps,
    locs=train_locs,
    goals=train_goals,
    direction_outputs=train_output_dirs,
)
dataset_test = MazeDataset(
    maze_ssp=test_mazez_sps,
    loc_ssps=test_loc_sps,
    goal_ssps=test_goal_sps,
    locs=test_locs,
    goals=test_goals,
    direction_outputs=test_output_dirs,
)
dataset_viz = MazeDataset(
    maze_ssp=viz_maze_sps,
    loc_ssps=viz_loc_sps,
    goal_ssps=viz_goal_sps,
    locs=viz_locs,
    goals=viz_goals,
    direction_outputs=viz_output_dirs,
)

trainloader = torch.utils.data.DataLoader(
    dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0,
)

# For testing just do everything in one giant batch
testloader = torch.utils.data.DataLoader(
    dataset_test, batch_size=len(dataset_test), shuffle=False, num_workers=0,
)

vizloader = torch.utils.data.DataLoader(
    dataset_viz, batch_size=len(dataset_viz), shuffle=False, num_workers=0,
)

# input is maze, loc, goal ssps, output is 2D direction to move
model = FeedForward(input_size=args.dim * 3, output_size=2)

if args.load_saved_model:
    model.load_state_dict(torch.load(args.load_saved_model), strict=False)

# Open a tensorboard writer if a logging directory is given
if args.logdir != '':
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    save_dir = os.path.join(args.logdir, current_time)
    writer = SummaryWriter(log_dir=save_dir)
    if args.weight_histogram:
        # Log the initial parameters
        for name, param in model.named_parameters():
            writer.add_histogram('parameters/' + name, param.clone().cpu().data.numpy(), 0)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

for e in range(args.epochs):
    print('Epoch: {0}'.format(e + 1))

    avg_loss = 0
    n_batches = 0
    for i, data in enumerate(trainloader):
        maze_loc_goal_ssps, directions, locs, goals = data

        if maze_loc_goal_ssps.size()[0] != args.batch_size:
            continue  # Drop data, not enough for a batch
        optimizer.zero_grad()

        outputs = model(maze_loc_goal_ssps)

        loss = criterion(outputs, directions)
        # print(loss.data.item())
        avg_loss += loss.data.item()
        n_batches += 1

        loss.backward()

        optimizer.step()

    if args.logdir != '':
        if n_batches > 0:
            avg_loss /= n_batches
            print(avg_loss)
            writer.add_scalar('avg_loss', avg_loss, e + 1)

        if args.weight_histogram and (e + 1) % 10 == 0:
            for name, param in model.named_parameters():
                writer.add_histogram('parameters/' + name, param.clone().cpu().data.numpy(), e + 1)

print("Testing")
with torch.no_grad():
    # Everything is in one batch, so this loop will only happen once
    for i, data in enumerate(testloader):
        maze_loc_goal_ssps, directions, locs, goals = data

        outputs = model(maze_loc_goal_ssps)

        loss = criterion(outputs, directions)

        # print(loss.data.item())

    if args.logdir != '':
        writer.add_scalar('test_loss', loss.data.item())

print("Visualization")
with torch.no_grad():
    # Everything is in one batch, so this loop will only happen once
    for i, data in enumerate(vizloader):
        maze_loc_goal_ssps, directions, locs, goals = data

        outputs = model(maze_loc_goal_ssps)

        loss = criterion(outputs, directions)

        # print(loss.data.item())

    if args.logdir != '':
        fig_pred = plot_path_predictions(
            directions=outputs, coords=locs, type='colour'
        )
        writer.add_figure('viz set predictions', fig_pred)
        fig_truth = plot_path_predictions(
            directions=directions, coords=locs, type='colour'
        )
        writer.add_figure('ground truth', fig_truth)

        fig_pred_quiver = plot_path_predictions(
            directions=outputs, coords=locs, dcell=xs[1] - xs[0]
        )
        writer.add_figure('viz set predictions quiver', fig_pred_quiver)
        fig_truth_quiver = plot_path_predictions(
            directions=directions, coords=locs, dcell=xs[1] - xs[0]
        )
        writer.add_figure('ground truth quiver', fig_truth_quiver)

        writer.add_scalar('viz_loss', loss.data.item())

# Close tensorboard writer
if args.logdir != '':
    writer.close()

    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))

    params = vars(args)
    with open(os.path.join(save_dir, "params.json"), "w") as f:
        json.dump(params, f)
