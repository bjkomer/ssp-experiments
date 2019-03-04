from training_utils import ValidationSet, create_dataloader
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
from models import FeedForward, MLP
from datasets import MazeDataset
import nengo.spa as spa

parser = argparse.ArgumentParser(
    'Train a function that given a maze and a goal location, computes the direction to move to get to that goal'
)

parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train for')
parser.add_argument('--epoch-offset', type=int, default=0,
                    help='Optional offset to start epochs counting from. To be used when continuing training')
parser.add_argument('--validation-period', type=int, default=250, help='number of epochs before a validation set run')
parser.add_argument('--subsample', type=int, default=2, help='amount to subsample for the visualization validation')
parser.add_argument('--maze-id-type', type=str, choices=['ssp', 'one-hot', 'random-sp'], default='ssp',
                    help='ssp: region corresponding to maze layout.'
                         'one-hot: each maze given a one-hot vector.'
                         'random-sp: each maze given a unique random SP as an ID')
parser.add_argument('--seed', type=int, default=13, help='Seed for training and generating axis SSPs')
parser.add_argument('--dim', type=int, default=512, help='Dimensionality of the SSPs')
parser.add_argument('--n-train-samples', type=int, default=1000, help='Number of training samples')
parser.add_argument('--n-test-samples', type=int, default=1000, help='Number of testing samples')
parser.add_argument('--hidden-size', type=int, default=512, help='Size of the hidden layer in the model')
parser.add_argument('--n-hidden-layers', type=int, default=1, help='Number of hidden layers in the model')
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
n_mazes = fine_mazes.shape[0]

if args.maze_id_type == 'ssp':
    id_size = args.dim
elif args.maze_id_type == 'one-hot':
    id_size = n_mazes
    # overwrite data
    maze_sps = np.eye(n_mazes)
elif args.maze_id_type == 'random-sp':
    id_size = args.dim
    # overwrite data
    for mi in range(n_mazes):
        maze_sps[mi, :] = spa.SemanticPointer(args.dim).v
else:
    raise NotImplementedError

# Create a validation/visualization set to run periodically while training and at the end
# validation_set = ValidationSet(data=data, maze_indices=np.arange(n_mazes), goal_indices=[0])
validation_set = ValidationSet(data=data, maze_sps=maze_sps, maze_indices=[0, 1, 2, 3], goal_indices=[0, 1], subsample=args.subsample)

trainloader = create_dataloader(data=data, n_samples=args.n_train_samples, maze_sps=maze_sps, args=args)

testloader = create_dataloader(data=data, n_samples=args.n_test_samples, maze_sps=maze_sps, args=args)

# Reset seeds here after generating data
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# input is maze, loc, goal ssps, output is 2D direction to move
if args.n_hidden_layers > 1:
    model = MLP(input_size=id_size + args.dim * 2, hidden_size=args.hidden_size, output_size=2, n_layers=args.n_hidden_layers)
else:
    model = FeedForward(input_size=id_size + args.dim * 2, hidden_size=args.hidden_size, output_size=2)

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

validation_set.run_ground_truth(writer)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

for e in range(args.epoch_offset, args.epochs + args.epoch_offset):
    print('Epoch: {0}'.format(e + 1))

    if e % args.validation_period == 0:
        # do a validation run and save images
        validation_set.run_validation(model, writer, e)

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
validation_set.run_validation(model, writer, args.epochs + args.epoch_offset)


# Close tensorboard writer
if args.logdir != '':
    writer.close()

    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))

    params = vars(args)
    with open(os.path.join(save_dir, "params.json"), "w") as f:
        json.dump(params, f)
