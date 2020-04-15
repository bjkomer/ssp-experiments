import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import argparse
import json
from datetime import datetime
import os.path as osp
from ssp_navigation.utils.encodings import get_encoding_function, add_encoding_params
from ssp_navigation.utils.datasets import GenericDataset

from training import add_training_params, train

parser = argparse.ArgumentParser('Return the distance in 2D space between two encodings')

parser = add_encoding_params(parser)
parser = add_training_params(parser)

parser.add_argument('--logdir', type=str, default='',
                    help='Directory for saved model and tensorboard log')
parser.add_argument('--concatenate', type=int, default=0,
                    help='if 0, sum the two input points together, if 1, concatenate them into a larger vector instead')
parser.add_argument('--function', type=str, default='distance', choices=['distance', 'direction', 'displacement', 'centroid'])
parser.add_argument('--no-viz', action='store_true', help='do not create visualizations')
args = parser.parse_args()

# modify logdir based on the parameters
logdir = "output_function/{}/{}_concat{}_d{}_limit{}_seed{}_{}samples".format(args.function, args.spatial_encoding, args.concatenate, args.dim, args.limit, args.seed, args.n_samples)
# if default logdir given, add a subfolder based on the parameters
if args.logdir == '':
    args.logdir = logdir

encoding_func, repr_dim = get_encoding_function(args, limit_low=-args.limit, limit_high=args.limit)


if args.concatenate == 1:
    repr_dim *= 2

vectors = np.zeros((args.n_samples, repr_dim))


rng = np.random.RandomState(args.seed)
coords_a = rng.uniform(-args.limit, args.limit, size=(args.n_samples, 2))
coords_b = rng.uniform(-args.limit, args.limit, size=(args.n_samples, 2))

if args.function == 'distance':
    outputs = np.zeros((args.n_samples, 1))
    outputs[:, 0] = np.linalg.norm(coords_a - coords_b, axis=1)
elif args.function == 'centroid':
    outputs = np.zeros((args.n_samples, 2))
    outputs[:, :] = (coords_a + coords_b)/2.
elif args.function == 'direction':
    outputs = np.zeros((args.n_samples, 1))
    diff = coords_a - coords_b
    outputs[:, 0] = np.arctan2(diff[:, 1], diff[:, 0])
    # only use positive angles, add pi to negative ones
    # this makes the order of the two input coordinates not matter
    for n in range(args.n_samples):
        if outputs[n, 0] < 0:
            outputs[n, 0] += np.pi
elif args.function == 'displacement':
    outputs = np.zeros((args.n_samples, 2))
    outputs[:, :] = coords_a - coords_b
    raise NotImplementedError

encoded_coords_a = np.zeros((args.n_samples, args.dim))
encoded_coords_b = np.zeros((args.n_samples, args.dim))

for n in range(args.n_samples):
    encoded_coords_a[n, :] = encoding_func(coords_a[n, 0], coords_a[n, 1])
    encoded_coords_b[n, :] = encoding_func(coords_b[n, 0], coords_b[n, 1])

if args.concatenate == 1:
    vectors[:, :repr_dim // 2] = encoded_coords_a
    vectors[:, repr_dim // 2:] = encoded_coords_b
else:
    vectors = encoded_coords_a + encoded_coords_b

n_samples = vectors.shape[0]
n_train = int(args.train_fraction * n_samples)
n_test = n_samples - n_train
assert(n_train > 0 and n_test > 0)
train_vectors = vectors[:n_train]
train_outputs = outputs[:n_train]
test_vectors = vectors[n_train:]
test_outputs = outputs[n_train:]

dataset_train = GenericDataset(inputs=train_vectors, outputs=train_outputs)
dataset_test = GenericDataset(inputs=test_vectors, outputs=test_outputs)

trainloader = torch.utils.data.DataLoader(
    dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0,
)

# For testing just do everything in one giant batch
testloader = torch.utils.data.DataLoader(
    dataset_test, batch_size=len(dataset_test), shuffle=False, num_workers=0,
)

# training, testing, and saving results all happen in this function
train(args=args, trainloader=trainloader, testloader=testloader, input_size=repr_dim, output_size=outputs.shape[1])
