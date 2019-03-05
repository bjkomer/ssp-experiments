import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import json
from tensorboardX import SummaryWriter
from datetime import datetime
from spatial_semantic_pointers.utils import make_good_unitary, encode_point
from path_utils import path_function, plot_path_predictions, example_path, example_xs, example_ys
from models import FeedForward
from datasets import PathDataset

parser = argparse.ArgumentParser(
    'Train a function that maps from a spatial semantic pointer to the direction an agent should move to follow a path'
)

parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train for')
parser.add_argument('--seed', type=int, default=13, help='Seed for training and generating axis SSPs')
parser.add_argument('--dim', type=int, default=512, help='Dimensionality of the SSPs')
parser.add_argument('--n-train-samples', type=int, default=1000, help='Number of training samples')
parser.add_argument('--n-test-samples', type=int, default=1000, help='Number of testing samples')
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-histogram', action='store_true', help='Save histogram of the weights')
parser.add_argument('--path-type', type=str, default='example', choices=['example', 'random'], help='path to learn')
parser.add_argument('--logdir', type=str, default='path_function',
                    help='Directory for saved model and tensorboard log')
parser.add_argument('--load-saved-model', type=str, default='', help='Saved model to load from')

args = parser.parse_args()

# rng = np.random.seed(args.seed)
rng = np.random.RandomState(seed=args.seed)
torch.manual_seed(args.seed)

# x_axis_sp, y_axis_sp = get_axes(
#     dim=args.dim,
#     theta=np.pi/2.,
#     seed=args.seed,
#     vectors_in_fourier=False,
#     make_unitary=True
# )

x_axis_sp = make_good_unitary(dim=args.dim, rng=rng)
y_axis_sp = make_good_unitary(dim=args.dim, rng=rng)

np.random.seed(args.seed)
torch.manual_seed(args.seed)


# Choose path and linspace
if args.path_type == 'example':
    path = example_path
elif args.path_type == 'random':
    path = np.random.randint(low=0, high=5, size=example_path.shape)
else:
    raise NotImplementedError
xs = example_xs
ys = example_ys


# Set up train and test sets
train_inputs = np.zeros((args.n_train_samples, args.dim))
train_outputs = np.zeros((args.n_train_samples, 2))
train_coords = np.zeros((args.n_train_samples, 2))

test_inputs = np.zeros((args.n_test_samples, args.dim))
test_outputs = np.zeros((args.n_test_samples, 2))
test_coords = np.zeros((args.n_test_samples, 2))

for n in range(args.n_train_samples):
    x = np.random.uniform(low=xs[0], high=xs[-1])
    y = np.random.uniform(low=ys[0], high=ys[-1])

    train_inputs[n, :] = encode_point(x, y, x_axis_sp, y_axis_sp).v
    train_coords[n, :] = np.array([x, y])
    train_outputs[n, :] = path_function(train_coords[n, :], path, xs, ys)


for n in range(args.n_test_samples):
    x = np.random.uniform(low=xs[0], high=xs[-1])
    y = np.random.uniform(low=ys[0], high=ys[-1])

    test_inputs[n, :] = encode_point(x, y, x_axis_sp, y_axis_sp).v
    test_coords[n, :] = np.array([x, y])
    test_outputs[n, :] = path_function(test_coords[n, :], path, xs, ys)


dataset_train = PathDataset(ssp_inputs=train_inputs, direction_outputs=train_outputs, coord_inputs=train_coords)
dataset_test = PathDataset(ssp_inputs=test_inputs, direction_outputs=test_outputs, coord_inputs=test_coords)

trainloader = torch.utils.data.DataLoader(
    dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0,
)

# For testing just do everything in one giant batch
testloader = torch.utils.data.DataLoader(
    dataset_test, batch_size=len(dataset_test), shuffle=False, num_workers=0,
)

model = FeedForward(input_size=train_inputs.shape[1], output_size=train_outputs.shape[1])

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
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

for e in range(args.epochs):
    print('Epoch: {0}'.format(e + 1))

    avg_loss = 0
    n_batches = 0
    for i, data in enumerate(trainloader):

        ssps, directions, coords = data

        if ssps.size()[0] != args.batch_size:
            continue  # Drop data, not enough for a batch
        optimizer.zero_grad()

        outputs = model(ssps)

        loss = criterion(outputs, directions)
        # print(loss.data.item())
        avg_loss += loss.data.item()
        n_batches += 1

        loss.backward()

        optimizer.step()

    if args.logdir != '':
        if n_batches > 0:
            avg_loss /= n_batches
            writer.add_scalar('avg_loss', avg_loss, e + 1)

        if args.weight_histogram and (e + 1) % 10 == 0:
            for name, param in model.named_parameters():
                writer.add_histogram('parameters/' + name, param.clone().cpu().data.numpy(), e + 1)

print("Testing")
with torch.no_grad():
    # Everything is in one batch, so this loop will only happen once
    for i, data in enumerate(testloader):
        ssps, directions, coords = data

        outputs = model(ssps)

        loss = criterion(outputs, directions)

        # print(loss.data.item())

    if args.logdir != '':
        fig_pred = plot_path_predictions(
            directions=outputs, coords=coords,
        )
        writer.add_figure('test set predictions', fig_pred)
        fig_truth = plot_path_predictions(
            directions=directions, coords=coords,
        )
        writer.add_figure('ground truth', fig_truth)
        writer.add_scalar('test_loss', loss.data.item())

# Close tensorboard writer
if args.logdir != '':
    writer.close()

    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))

    params = vars(args)
    with open(os.path.join(save_dir, "params.json"), "w") as f:
        json.dump(params, f)