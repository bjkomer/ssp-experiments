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
parser.add_argument('--logdir', type=str, default='maze_solve_function',
                    help='Directory for saved model and tensorboard log')
parser.add_argument('--load-saved-model', type=str, default='', help='Saved model to load from')
parser.add_argument('--save-generated-data', action='store_true',
                    help='Save train/test/vis data so it does not need to be regenerated')

args = parser.parse_args()


assert(args.limit_low < args.limit_high)

rng = np.random.RandomState(seed=args.seed)
torch.manual_seed(args.seed)

x_axis_sp = make_good_unitary(dim=args.dim, rng=rng)
y_axis_sp = make_good_unitary(dim=args.dim, rng=rng)

np.random.seed(args.seed)
torch.manual_seed(args.seed)

xs = np.linspace(args.limit_low, args.limit_high, args.res)
ys = np.linspace(args.limit_low, args.limit_high, args.res)

# Generate a random maze
maze_ssp, coarse_maze, fine_maze = generate_maze_sp(
    size=args.maze_size,
    xs=xs,
    ys=ys,
    x_axis_sp=x_axis_sp,
    y_axis_sp=y_axis_sp,
    normalize=True,
    obstacle_ratio=.2,
    map_style='blocks'
)

# Get a list of possible goal locations to choose (will correspond to all free spaces in the coarse maze)
# free_spaces = np.argwhere(coarse_maze == 0)
# Get a list of possible goal locations to choose (will correspond to all free spaces in the fine maze)
free_spaces = np.argwhere(fine_maze == 0)
print(free_spaces.shape)
n_free_spaces = free_spaces.shape[0]

# Training
train_locs = np.zeros((args.n_train_samples, 2))
train_goals = np.zeros((args.n_train_samples, 2))
train_loc_sps = np.zeros((args.n_train_samples, args.dim))
train_goal_sps = np.zeros((args.n_train_samples, args.dim))
train_output_dirs = np.zeros((args.n_train_samples, 2))

train_current_loc_indices = np.random.randint(low=0, high=n_free_spaces, size=args.n_train_samples)
train_goal_indices = np.random.randint(low=0, high=n_free_spaces, size=args.n_train_samples)

# Testing
test_locs = np.zeros((args.n_train_samples, 2))
test_goals = np.zeros((args.n_train_samples, 2))
test_loc_sps = np.zeros((args.n_train_samples, args.dim))
test_goal_sps = np.zeros((args.n_train_samples, args.dim))
test_output_dirs = np.zeros((args.n_train_samples, 2))

test_current_loc_indices = np.random.randint(low=0, high=n_free_spaces, size=args.n_test_samples)
test_goal_indices = np.random.randint(low=0, high=n_free_spaces, size=args.n_test_samples)

# Visualization
viz_locs = np.zeros((args.n_test_samples, 2))
viz_goals = np.zeros((args.n_test_samples, 2))
viz_loc_sps = np.zeros((args.n_test_samples, args.dim))
viz_goal_sps = np.zeros((args.n_test_samples, args.dim))
viz_output_dirs = np.zeros((args.n_test_samples, 2))

viz_current_loc_indices = np.random.randint(low=0, high=n_free_spaces, size=args.n_test_samples)
viz_goal_index = np.random.randint(low=0, high=n_free_spaces)
viz_goal_index = free_spaces[viz_goal_index, :]

# TODO: since optimal paths are expensive to generate, save the data and load it if it already exists
data_fname = os.path.join(
    args.logdir, 'data_seed{}_dim{}_ntrain{}_ntest{}.npz'.format(
        args.seed, args.dim, args.n_train_samples, args.n_test_samples
    )
)
if os.path.isfile(data_fname):
    data = np.load(data_fname)
    maze_ssp = spa.SemanticPointer(data['maze_ssp'])
    train_loc_ssps = data['train_loc_ssps']
    train_goal_ssps = data['train_goal_ssps']
    train_locs = data['train_locs']
    train_goals = data['train_goals']
    train_output_dirs = data['train_output_dirs']
    test_loc_ssps = data['test_loc_ssps']
    test_goal_ssps = data['test_goal_ssps']
    test_locs = data['test_locs']
    test_goals = data['test_goals']
    test_output_dirs = data['test_output_dirs']
    viz_loc_ssps = data['viz_loc_ssps']
    viz_goal_ssps = data['viz_goal_ssps']
    viz_locs = data['viz_locs']
    viz_goals = data['viz_goals']
    viz_output_dirs = data['viz_output_dirs']
else:

    for n in range(args.n_train_samples):
        print("Train Sample {} of {}".format(n+1, args.n_train_samples))
        # 2D coordinate of the goal
        goal_index = free_spaces[train_goal_indices[n], :]
        goal_x = xs[goal_index[0]]
        goal_y = ys[goal_index[1]]

        # 2D coordinate of the agent's current location
        loc_index = free_spaces[train_current_loc_indices[n], :]
        loc_x = xs[loc_index[0]]
        loc_y = ys[loc_index[1]]

        # Compute the optimal path given this goal
        solved_maze = solve_maze(fine_maze, start_indices=loc_index, goal_indices=goal_index)

        train_locs[n, 0] = loc_x
        train_locs[n, 1] = loc_y
        train_goals[n, 0] = goal_x
        train_goals[n, 1] = goal_y
        train_loc_sps[n, :] = encode_point(loc_x, loc_y, x_axis_sp, y_axis_sp).v
        train_goal_sps[n, :] = encode_point(goal_x, goal_y, x_axis_sp, y_axis_sp).v

        train_output_dirs[n, :] = solved_maze[loc_index[0], loc_index[1], :]

    for n in range(args.n_test_samples):
        print("Test Sample {} of {}".format(n+1, args.n_train_samples))
        # 2D coordinate of the goal
        goal_index = free_spaces[test_goal_indices[n], :]
        goal_x = xs[goal_index[0]]
        goal_y = ys[goal_index[1]]

        # 2D coordinate of the agent's current location
        loc_index = free_spaces[test_current_loc_indices[n], :]
        loc_x = xs[loc_index[0]]
        loc_y = ys[loc_index[1]]

        # Compute the optimal path given this goal
        solved_maze = solve_maze(fine_maze, start_indices=loc_index, goal_indices=goal_index)

        test_locs[n, 0] = loc_x
        test_locs[n, 1] = loc_y
        test_goals[n, 0] = goal_x
        test_goals[n, 1] = goal_y
        test_loc_sps[n, :] = encode_point(loc_x, loc_y, x_axis_sp, y_axis_sp).v
        test_goal_sps[n, :] = encode_point(goal_x, goal_y, x_axis_sp, y_axis_sp).v

        test_output_dirs[n, :] = solved_maze[loc_index[0], loc_index[1], :]

    # Note: for ease of plotting for testing, would be helpful to just have a single or few goals
    #       because of this a separate 'visualization' set will be used
    for n in range(args.n_test_samples):
        print("Viz Sample {} of {}".format(n+1, args.n_train_samples))
        # 2D coordinate of the goal
        goal_x = xs[viz_goal_index[0]]
        goal_y = ys[viz_goal_index[1]]

        # 2D coordinate of the agent's current location
        loc_index = free_spaces[viz_current_loc_indices[n], :]
        loc_x = xs[loc_index[0]]
        loc_y = ys[loc_index[1]]

        # Compute the optimal path given this goal
        solved_maze = solve_maze(fine_maze, start_indices=loc_index, goal_indices=viz_goal_index)

        viz_locs[n, 0] = loc_x
        viz_locs[n, 1] = loc_y
        viz_goals[n, 0] = goal_x
        viz_goals[n, 1] = goal_y
        viz_loc_sps[n, :] = encode_point(loc_x, loc_y, x_axis_sp, y_axis_sp).v
        viz_goal_sps[n, :] = encode_point(goal_x, goal_y, x_axis_sp, y_axis_sp).v

        viz_output_dirs[n, :] = solved_maze[loc_index[0], loc_index[1], :]

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    if args.save_generated_data:
        np.savez(
            data_fname,
            maze_ssp=maze_ssp.v,
            train_loc_ssps=train_loc_sps,
            train_goal_ssps=train_goal_sps,
            train_locs=train_locs,
            train_goals=train_goals,
            train_output_dirs=train_output_dirs,
            test_loc_ssps=test_loc_sps,
            test_goal_ssps=test_goal_sps,
            test_locs=test_locs,
            test_goals=test_goals,
            test_output_dirs=test_output_dirs,
            viz_loc_ssps=viz_loc_sps,
            viz_goal_ssps=viz_goal_sps,
            viz_locs=viz_locs,
            viz_goals=viz_goals,
            viz_output_dirs=viz_output_dirs,
        )


#TODO fix to correct datasets
dataset_train = MazeDataset(
    maze_ssp=maze_ssp.v,
    loc_ssps=train_loc_sps,
    goal_ssps=train_goal_sps,
    locs=train_locs,
    goals=train_goals,
    direction_outputs=train_output_dirs,
)
dataset_test = MazeDataset(
    maze_ssp=maze_ssp.v,
    loc_ssps=test_loc_sps,
    goal_ssps=test_goal_sps,
    locs=test_locs,
    goals=test_goals,
    direction_outputs=test_output_dirs,
)
dataset_viz = MazeDataset(
    maze_ssp=maze_ssp.v,
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