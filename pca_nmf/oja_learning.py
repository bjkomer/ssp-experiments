import numpy as np
import matplotlib.pyplot as plt
import argparse
from utils import spatial_heatmap


parser = argparse.ArgumentParser()

parser.add_argument('--n-components', type=int, default=20)
parser.add_argument('--dataset', type=str,
                    default='data/random_walk_pc-gauss_0.75_0.0to2.2_512dim_10000steps.npz')
parser.add_argument('--spatial-encoding', type=str, default='pc-gauss',
                    choices=[
                        'ssp', 'random', '2d', '2d-normalized', 'one-hot', 'hex-trig',
                        'trig', 'random-trig', 'random-proj', 'learned', 'frozen-learned',
                        'pc-gauss', 'tile-coding', 'pc-dog'
                    ])
parser.add_argument('--res', type=int, default=128, help='resolution of the spatial heatmap')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n-epochs', type=int, default=1)

parser.add_argument('--saturation', type=float, default=30, help='grid cell activation saturation')

parser.add_argument('--non-negative', action='store_true')

# Encoding specific parameters
parser.add_argument('--pc-gauss-sigma', type=float, default=0.75)
parser.add_argument('--hex-freq-coef', type=float, default=2.5, help='constant to scale frequencies by')
parser.add_argument('--n-tiles', type=int, default=8, help='number of layers for tile coding')
parser.add_argument('--n-bins', type=int, default=8, help='number of bins for tile coding')
parser.add_argument('--ssp-scaling', type=float, default=1.0)
parser.add_argument('--dim', type=int, default=512)
parser.add_argument('--seed', type=int, default=13)

args = parser.parse_args()

data = np.load(args.dataset)

r = data['activations'][:, :args.dim]

if True:
    if False:
        for d in range(args.dim):
            r[:, d] -= np.mean(r[:, d])
    if True:
        for n in range(r.shape[0]):
            r[n, :] -= np.mean(r[n, :])

positions = data['positions']

n_steps = r.shape[0]


def activation(x):
    # args.saturation*.7 * np.arctan(100*x)
    return np.tanh(x)
    # return np.mean(x)
    # return x


rng = np.random.RandomState(seed=args.seed)
n_neurons = args.dim

# weights
# J = np.zeros((n_steps, n_neurons))
# J[0, :] = rng.uniform(0, 1, size=(n_neurons,))
# J[0, :] /= np.linalg.norm(J[0, :])

# J = np.zeros((n_neurons,))
J = rng.uniform(0, 1, size=(n_neurons,))
J /= np.linalg.norm(J)

# output
w = np.zeros((n_steps, 1))

# # inputs
# r = np.zeros((n_neurons,))

for e in range(args.n_epochs):
    for t in range(n_steps-1):
        print('\x1b[2K\r Step {} of {}'.format(t + 1, n_steps), end="\r")

        # w[t, 0] = activation(np.dot(J[t, :], r[t, :]))
        #
        # dJ = args.lr * (w[t]*r[t, :] - w[t, 0]**2 * J[t, :])
        # J[t + 1, :] = J[t, :] + dJ

        w[t, 0] = activation(np.dot(J, r[t, :]))

        # lr = 1 / (t + args.lr)
        lr = args.lr

        # dJ = lr * (w[t, 0]*r[t, :] - w[t, 0]**2 * J)
        # dJ = lr * (w[t, 0] * r[t, :] - (np.dot(w[t, 0], w[t, 0])) * J)
        dJ = lr * (np.dot(w[t, 0], r[t, :]) - (np.dot(w[t, 0], w[t, 0])) * J)
        J += dJ

        if args.non_negative:
            J[J < 0] = 0

    # w[t + 1, 0] = activation(np.dot(J[t + 1, :], r[t + 1, :]))
    w[t + 1, 0] = activation(np.dot(J, r[t + 1, :]))

# print(w[-1])

outputs = np.zeros((n_steps, 1))
for t in range(n_steps):
    # tmp = 0
    # for j in range(n_neurons):
    #     tmp += J[j] * r[t, j]
    # print(np.dot(J, r[t, :]))
    # print(tmp)
    # assert np.allclose(tmp, np.dot(J, r[t, :]))
    outputs[t, 0] = activation(np.dot(J, r[t, :]))

    # apply saturation
    outputs[outputs>args.saturation] = args.saturation

xs = np.linspace(0, 2.2, args.res)
ys = np.linspace(0, 2.2, args.res)
# heatmap = spatial_heatmap(w, positions, xs, ys)
heatmap = spatial_heatmap(outputs, positions, xs, ys)

print(heatmap)
print(np.max(heatmap))
print(np.min(heatmap))

plt.imshow(heatmap[0, :, :])
plt.show()
