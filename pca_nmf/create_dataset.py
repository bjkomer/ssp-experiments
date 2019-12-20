import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils import Environment, spatial_heatmap
from ssp_navigation.utils.encodings import get_encoding_function
from sklearn.decomposition import PCA
import argparse

parser = argparse.ArgumentParser()

# parser.add_argument('--n-components', type=int, default=20)
# parser.add_argument('--dataset', type=str,
#                     default='../path_integration/data/path_integration_raw_trajectories_100t_15s_seed13.npz')
parser.add_argument('--spatial-encoding', type=str, default='pc-gauss',
                    choices=[
                        'ssp', 'random', '2d', '2d-normalized', 'one-hot', 'hex-trig',
                        'trig', 'random-trig', 'random-proj', 'learned', 'frozen-learned',
                        'pc-gauss', 'tile-coding'
                    ])
parser.add_argument('--res', type=int, default=64, help='resolution of the spatial heatmap')

parser.add_argument('--limit-low', type=float, default=0.0)
parser.add_argument('--limit-high', type=float, default=2.2)

parser.add_argument('--n-steps', type=int, default=10000, help='number of steps in the random walk')

# Encoding specific parameters
parser.add_argument('--pc-gauss-sigma', type=float, default=0.75)
parser.add_argument('--hex-freq-coef', type=float, default=2.5, help='constant to scale frequencies by')
parser.add_argument('--n-tiles', type=int, default=8, help='number of layers for tile coding')
parser.add_argument('--n-bins', type=int, default=8, help='number of bins for tile coding')
parser.add_argument('--ssp-scaling', type=float, default=1.0)
parser.add_argument('--dim', type=int, default=512)
parser.add_argument('--seed', type=int, default=13)

args = parser.parse_args()

encoding_func, dim = get_encoding_function(args, limit_low=args.limit_low, limit_high=args.limit_high)

env = Environment(encoding_func=encoding_func, limit_low=args.limit_low, limit_high=args.limit_high)

# encoding specific string
encoding_specific = ''
if args.spatial_encoding == 'ssp':
    encoding_specific = args.ssp_scaling
elif args.spatial_encoding == 'frozen-learned':
    encoding_specific = args.frozen_model
elif args.spatial_encoding == 'pc-gauss' or args.spatial_encoding == 'pc-gauss-softmax':
    encoding_specific = args.pc_gauss_sigma
elif args.spatial_encoding == 'hex-trig':
    encoding_specific = args.hex_freq_coef
elif args.spatial_encoding == 'tile-coding':
    encoding_specific = '{}tiles_{}bins'.format(args.n_tiles, args.n_bins)

fname = 'data/random_walk_{}_{}_{}dim_{}steps.npz'.format(
    args.spatial_encoding,
    encoding_specific,
    args.dim,
    args.n_steps,
)

positions = np.zeros((args.n_steps, 2))
activations = np.zeros((args.n_steps, dim))

for n in range(args.n_steps):
    print('\x1b[2K\r Step {} of {}'.format(n + 1, args.n_steps), end="\r")
    activations[n, :], positions[n, :] = env.step()

np.savez(fname, positions=positions, activations=activations)
