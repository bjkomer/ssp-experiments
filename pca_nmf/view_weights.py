import numpy as np
import matplotlib.pyplot as plt
import argparse
from utils import spatial_heatmap


parser = argparse.ArgumentParser()

parser.add_argument('--fname', type=str, default='output/rep_output.npz')
parser.add_argument('--gc-index', type=int, default=0, help='index of grid cell to visualize')
parser.add_argument('--img-res', type=int, default=256)
parser.add_argument('--limit-low', type=float, default=0)
parser.add_argument('--limit-high', type=float, default=10)
parser.add_argument('--n-place-cells-dim', type=int, default=20)
parser.add_argument('--n-grid-cells', type=int, default=100)
parser.add_argument('--pc-size', type=float, default=0.75)
parser.add_argument('--saturation', type=float, default=30, help='grid cell activation saturation')
parser.add_argument('--activation-func', type=str, default='linear', choices=['linear', 'sigmoid'])
parser.add_argument('--arc', type=str, default='single', choices=['single', 'multiple'])

parser.add_argument('--view-tesselation', action='store_true',
                    help='tesselate the output image to show the looping boundaries better')

args = parser.parse_args()

data = np.load(args.fname)
J = data['J']
W = data['W']
pc_centers = data['pc_centers']
a_std = data['a_std']
b_std = data['b_std']
c_std = data['c_std']
a_std2 = data['a_std2']
b_std2 = data['b_std2']
c_std2 = data['c_std2']
sigma_x = args.pc_size
sigma_x2 = sigma_x * 2

rho = 0.5

xs = np.linspace(args.limit_low, args.limit_high, args.img_res)
ys = np.linspace(args.limit_low, args.limit_high, args.img_res)

# activations for every grid cell for each pixel in the image
activations = np.zeros((args.n_grid_cells, args.img_res, args.img_res))

def get_activations(x, y, J, W):
    diff_x = np.minimum((x - pc_centers[:, 0]) % args.limit_high, (pc_centers[:, 0] - x) % args.limit_high)
    diff_y = np.minimum((y - pc_centers[:, 1]) % args.limit_high, (pc_centers[:, 1] - y) % args.limit_high)

    # Place cell activations at this location
    square_dists = (a_std *(diff_x)**2 +2*b_std*(diff_x)*(diff_y) + c_std*(diff_y)**2 )
    square_dists2 = (a_std2 *(diff_x)**2 +2*b_std2*(diff_x)*(diff_y) + c_std2*(diff_y)**2 )

    A = sigma_x**2 / sigma_x2**2

    # print("square_dists.shape", square_dists.shape)

    r = args.saturation * np.exp(-square_dists) - args.saturation * A * np.exp(-square_dists2)

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
        # TODO: need an initial h here
        # NOTE: might only make sense to have this in learning, and not in viewing
        h = (1 - rho) * (J @ r.T).T + rho * (W @ (h).T).T
    else:
        raise NotImplementedError

    # NOTE: the outputs look very saturated if the slope is high, maybe only useful for training?
    if args.activation_func == 'sigmoid':
        slope = 1#100
        psi = args.saturation * 0.7 * np.arctan(slope * h)
    elif args.activation_func == 'linear':
        slope = 1#100
        psi = slope * h
    else:
        raise NotImplementedError

    return psi

for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        activations[:, i, j] = get_activations(x=x, y=y, J=J, W=W)

for i in range(J.shape[0]):
    print(activations[i, :, :])
    print(np.min(activations[i, :, :]))
    print(np.max(activations[i, :, :]))
    if args.view_tesselation:
        plt.imshow(np.tile(activations[i, :, :], (3, 3)))
    else:
        plt.imshow(activations[i, :, :])
    plt.show()


# for i in range(J.shape[0]):
#     plt.imshow(J[i, :].reshape(args.n_place_cells_dim, args.n_place_cells_dim))
#     plt.show()

# plt.imshow(J[args.gc_index, :].reshape(args.n_place_cells_dim, args.n_place_cells_dim))
# plt.show()
