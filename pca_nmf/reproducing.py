import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from utils import spatial_heatmap


parser = argparse.ArgumentParser()

parser.add_argument('--n-components', type=int, default=20)
# parser.add_argument('--dataset', type=str,
#                     default='data/random_walk_pc-gauss_0.75_0.0to2.2_512dim_10000steps.npz')
parser.add_argument('--spatial-encoding', type=str, default='pc-gauss',
                    choices=[
                        'ssp', 'random', '2d', '2d-normalized', 'one-hot', 'hex-trig',
                        'trig', 'random-trig', 'random-proj', 'learned', 'frozen-learned',
                        'pc-gauss', 'tile-coding', 'pc-dog'
                    ])
# parser.add_argument('--res', type=int, default=128, help='resolution of the spatial heatmap')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n-epochs', type=int, default=1)


parser.add_argument('--n-place-cells-dim', type=int, default=20)
parser.add_argument('--pc-size', type=float, default=0.75)
parser.add_argument('--n-grid-cells', type=int, default=100)
parser.add_argument('--res', type=int, default=35)
parser.add_argument('--limit-low', type=float, default=0)
parser.add_argument('--limit-high', type=float, default=10)
parser.add_argument('--saturation', type=float, default=30, help='grid cell activation saturation')
parser.add_argument('--duration', type=int, default=5e5)
parser.add_argument('--delta', type=float, default=0.1)
parser.add_argument('--max-weight', type=float, default=0.1)
parser.add_argument('--epsilon', type=float, default=1e7)
parser.add_argument('--activation-func', type=str, default='linear', choices=['linear', 'sigmoid'])
parser.add_argument('--arc', type=str, default='single', choices=['single', 'multiple'])

parser.add_argument('--non-negative', action='store_true')

parser.add_argument('--fname', type=str, default='rep_output.npz')

parser.add_argument('--gc-index', type=int, default=0, help='index of grid cell to visualize')

parser.add_argument('--no-display', action='store_true')

# Encoding specific parameters
# parser.add_argument('--pc-gauss-sigma', type=float, default=0.75)
# parser.add_argument('--hex-freq-coef', type=float, default=2.5, help='constant to scale frequencies by')
# parser.add_argument('--n-tiles', type=int, default=8, help='number of layers for tile coding')
# parser.add_argument('--n-bins', type=int, default=8, help='number of bins for tile coding')
# parser.add_argument('--ssp-scaling', type=float, default=1.0)
# parser.add_argument('--dim', type=int, default=512)
parser.add_argument('--seed', type=int, default=13)

args = parser.parse_args()

rng = np.random.RandomState(seed=args.seed)

eps = 1e-12

n_total_pc = args.n_place_cells_dim ** 2

lin_vel = args.limit_high / args.res
ang_vel = 2*np.pi

# pc_centers = '??'
pc_centers = np.zeros((n_total_pc, 2))
for i in range(args.n_place_cells_dim):
    for j in range(args.n_place_cells_dim):
        pc_centers[i * args.n_place_cells_dim + j, 0] = i * (args.limit_high / args.n_place_cells_dim)
        pc_centers[i * args.n_place_cells_dim + j, 1] = j * (args.limit_high / args.n_place_cells_dim)

sigma_x = args.pc_size
sigma_y = args.pc_size
theta = 2*np.pi*rng.uniform(1, len(pc_centers))  # Gives a "tilt" to the place cell... not used when sigma_x =sigma_y

a_std = np.cos(theta)**2/2/sigma_x**2 + np.sin(theta)**2/2/sigma_y**2
b_std = -np.sin(2*theta)/4/sigma_x**2 + np.sin(2*theta)/4/sigma_y**2
c_std = np.sin(theta)**2/2/sigma_x**2 + np.cos(theta)**2/2/sigma_y**2


sigma_x2 = 2 * sigma_x
sigma_y2 = 2 * sigma_y

theta2 = 2*np.pi*rng.uniform(1, len(pc_centers))
a_std2 = np.cos(theta2)**2/2/sigma_x2**2 + np.sin(theta2)**2/2/sigma_y2**2
b_std2 = -np.sin(2*theta2)/4/sigma_x2**2 + np.sin(2*theta2)/4/sigma_y2**2
c_std2 = np.sin(theta2)**2/2/sigma_x2**2 + np.cos(theta2)**2/2/sigma_y2**2

J = rng.uniform(0, 1, size=(args.n_grid_cells, n_total_pc))

# normFactor = repmat(sum(J,2),1,Struct.N1);
# norm_factor = np.sum(J, axis=1)
# print(J.shape)
# print(np.sum(J, axis=1).shape)
# norm_factor = np.sum(J, axis=1).reshape(args.n_grid_cells, 1)
norm_factor = np.tile(np.sum(J, axis=1)[:, np.newaxis], (1, n_total_pc))


J = args.max_weight * J / norm_factor
# inter layer weights initialized at zero
# W = np.zeros((args.n_grid_cells, 1))
W = np.zeros((1, args.n_grid_cells))
# the relative importance of the inter layer connections
rho = 0.5

h_avg = np.zeros((1, args.n_grid_cells))
# input data
r = np.zeros((1, n_total_pc))
# input from place cells to grid cells
h = np.zeros((1, args.n_grid_cells))

x = args.limit_high / 2
y = args.limit_high / 2

cur_dir = 0

next_x = x
next_y = y

# initialize derivatives
r_prev = 0
r_prev_prev = 0
r_gaussian = 0

dt = 1

# this is an assumption of the code
assert(args.limit_low == 0)

for t in range(int(args.duration)):
    # print(t)
    print('\x1b[2K\r Step {} of {}'.format(t + 1, int(args.duration)), end="\r")

    cur_dir += ang_vel * rng.randn()

    if cur_dir > 2*np.pi:
        cur_dir -= 2*np.pi
    elif cur_dir < 0:
        cur_dir += 2*np.pi

    next_x = x + lin_vel * np.cos(cur_dir)
    next_y = y + lin_vel * np.sin(cur_dir)

    # NOTE: this is assuming limit_low = 0
    if next_x > args.limit_high:
        next_x -= args.limit_high
    elif next_x < 0:
        next_x += args.limit_high
    if next_y > args.limit_high:
        next_y -= args.limit_high
    elif next_y < 0:
        next_y += args.limit_high

    x = next_x
    y = next_y

    # Calculating new place cells firing rate
    # since the boundary conditions are periodic, the diffs must reflect this
    # via min of modulo of the distances
    # NOTE: np.minimum is used instead of np.min because it is element-wise
    diff_x = np.minimum((x - pc_centers[:, 0]) % args.limit_high, (pc_centers[:, 0] - x) % args.limit_high)
    diff_y = np.minimum((y - pc_centers[:, 1]) % args.limit_high, (pc_centers[:, 1] - y) % args.limit_high)

    # Place cell activations at this location
    square_dists = (a_std *(diff_x)**2 +2*b_std*(diff_x)*(diff_y) + c_std*(diff_y)**2 )
    square_dists2 = (a_std2 *(diff_x)**2 +2*b_std2*(diff_x)*(diff_y) + c_std2*(diff_y)**2 )

    A = sigma_x**2 / sigma_x2**2

    # print("square_dists.shape", square_dists.shape)

    r[0, :] = args.saturation * np.exp(-square_dists) - args.saturation * A * np.exp(-square_dists2) + eps

    epsilon = 1/(t*args.delta + args.epsilon)

    if args.arc == 'single':
        # print("h before", h.shape)
        h = (J @ r.T).T#np.dot(J, r)  # TODO: make sure transposes are correct
        # print("h after", h.shape)
    elif args.arc == 'multiple':
        # print((J @ r.T).T.shape)
        # print("")
        # print("W.shape", W.shape)
        # print("h.shape", h.shape)
        # print((W @ (h + eps).T).shape)
        h = (1 - rho) * (J @ r.T).T + rho * (W @ (h + eps).T).T

    h_out = h

    if args.activation_func == 'sigmoid':
        slope = 100
        psi = args.saturation * 0.7 * np.arctan(slope * h_out)
    elif args.activation_func == 'linear':
        slope = 100
        psi = slope * h_out
    else:
        raise NotImplementedError

    # updating weights using the Oja rule
    # deltaJ = psi'*r -  eye(Struct.NmEC,Struct.NmEC).*(psi'*psi)*J;
    dJ = psi.T @ r - (psi.T @ psi) @ J
    J = J + epsilon * dJ

    if args.arc == 'multiple':
        # inter-output layer. weights need to be learned very slow. there's a positive feedback
        W = W - 0.001*epsilon*(psi.T @ psi)
        # no self connections
        W = np.ones_like(W) - np.dot(np.eye(len(W)), W)# NOTE: dot product

    if args.non_negative:
        J[J<0] = 0


    # Drawing
    if (not args.no_display) and (t % int(args.duration/100) == 0):
        # print(W)
        # print(J)
        # print(J.shape)
        # print(psi.shape)

        plt.imshow(J[args.gc_index,:].reshape(args.n_place_cells_dim, args.n_place_cells_dim))
        plt.show(block=False)
        plt.pause(0.001)


if not os.path.exists('output'):
    os.makedirs('output')

fname = 'output/{}'.format(args.fname)

np.savez(
    fname,
    J=J,
    W=W,
    pc_centers=pc_centers,
    a_std=a_std,
    b_std=b_std,
    c_std=c_std,
    a_std2=a_std2,
    b_std2=b_std2,
    c_std2=c_std2,
)