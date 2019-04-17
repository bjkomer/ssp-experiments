import numpy as np
import argparse
from spatial_semantic_pointers.utils import make_good_unitary, encode_point
from autoencoder_utils import AutoEncoder
import torch
import matplotlib.pyplot as plt
import os


parser = argparse.ArgumentParser('Run 2D supervised path integration experiment using pytorch')

parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--model', type=str, default='')
parser.add_argument('--res', type=int, default=256)
parser.add_argument('--limit', type=int, default=5)
parser.add_argument('--dim', type=int, default=512)
parser.add_argument('--hidden-size', type=int, default=128)
parser.add_argument('--folder', type=str, default='')

args = parser.parse_args()

rng = np.random.RandomState(seed=args.seed)

x_axis_sp = make_good_unitary(dim=args.dim, rng=rng)
y_axis_sp = make_good_unitary(dim=args.dim, rng=rng)

if args.folder == '':
    folder = 'images/autoencoder_rate_maps/dim{}_limit{}_seed{}'.format(args.dim, args.limit, args.seed)
else:
    folder = args.folder

if not os.path.exists(folder):
    os.makedirs(folder)

# TODO: get params from the json file saved alongside the model
model = AutoEncoder(input_dim=args.dim, hidden_dim=args.hidden_size)

model.load_state_dict(torch.load(args.model), strict=False)

model.eval()

xs = np.linspace(-args.limit, args.limit, args.res)
ys = np.linspace(-args.limit, args.limit, args.res)


activations = np.zeros((args.hidden_size, args.res, args.res))

with torch.no_grad():
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            activations[:, i, j] = model.forward_activations(
                torch.Tensor(encode_point(x, y, x_axis_sp, y_axis_sp).v).reshape(1, args.dim)
            )[1].detach().numpy()


for n in range(args.hidden_size):
    print("Neuron {} of {}".format(n + 1, args.hidden_size))

    fig, ax = plt.subplots()
    ax.imshow(activations[n, :, :])
    fig.savefig("{}/neuron_{}".format(folder, n))