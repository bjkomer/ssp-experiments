# Modify the ground truth maze solving data using an energy function for the walls and resave with a different name

import numpy as np
import argparse
from path_utils import plot_path_predictions
import matplotlib.pyplot as plt


def gaussian_2d(x, y, meshgrid, sigma):
    X, Y = meshgrid
    return np.exp(-((X - x) ** 2 + (Y - y) ** 2) / sigma / sigma)


parser = argparse.ArgumentParser(
    'Generate random mazes, and their solutions for particular goal locations'
)

parser.add_argument('--fname', type=str, default='maze_datasets/maze_dataset_{}_style_{}mazes_{}goals_{}res_{}size_{}seed.npz')
parser.add_argument('--energy-sigma', type=float, default=0.25, help='std for the wall energy gaussian')
parser.add_argument('--energy-scale', type=float, default=0.75, help='scale for the wall energy gaussian')

args = parser.parse_args()

prefix, extension = args.fname.split('.')

new_name = prefix + '_modified' + extension

# Load the original data
data = np.load(args.fname)

coarse_mazes = data['coarse_mazes']
fine_mazes = data['fine_mazes']
solved_mazes = data['solved_mazes']
maze_sps = data['maze_sps']
x_axis_sp = data['x_axis_sp']
y_axis_sp = data['y_axis_sp']
goal_sps = data['goal_sps']
goals = data['goals']
xs = data['xs']
ys = data['ys']
meshgrid = np.meshgrid(xs, ys)

# saving a copy for plotting side by side
solved_mazes_original = solved_mazes.copy()

n_mazes = solved_mazes.shape[0]
n_goals = solved_mazes.shape[1]
res = solved_mazes.shape[2]

# will contain a sum of gaussians placed at all wall locations
wall_energy = np.zeros_like(fine_mazes)

# will contain directions corresponding to the gradient of the wall energy
# to be added to the solved maze to augment the directions chosen
wall_gradient = np.zeros_like(solved_mazes)

# Compute wall energy
for ni in range(n_mazes):
    for xi, x in enumerate(xs):
        for yi, y in enumerate(ys):
            if fine_mazes[ni, xi, yi]:
                wall_energy[ni, :, :] += args.energy_scale * gaussian_2d(x=x, y=y, meshgrid=meshgrid, sigma=args.energy_sigma)

# Compute wall gradient using wall energy
for ni in range(n_mazes):

    gradients = np.gradient(wall_energy[ni, :, :])

    x_grad = gradients[0]
    y_grad = gradients[1]

    for gi in range(n_goals):
        # Note all the flipping and transposing to get things right
        wall_gradient[ni, gi, :, :, 1] = -x_grad.T
        wall_gradient[ni, gi, :, :, 0] = -y_grad.T

# modify the maze solution with the gradient
solved_mazes = solved_mazes + wall_gradient

# re-normalize the desired directions
for ni in range(n_mazes):
    for gi in range(n_goals):
        for xi in range(res):
            for yi in range(res):
                norm = np.linalg.norm(solved_mazes[ni, gi, xi, yi, :])
                if norm != 0:
                    solved_mazes[ni, gi, xi, yi, :] /= np.linalg.norm(solved_mazes[ni, gi, xi, yi, :])

# Save the modified data
# 'solved_mazes' is the only thing changing
np.savez(
    new_name,
    coarse_mazes=coarse_mazes,
    fine_mazes=fine_mazes,
    solved_mazes=solved_mazes,
    maze_sps=maze_sps,
    x_axis_sp=x_axis_sp,
    y_axis_sp=y_axis_sp,
    goal_sps=goal_sps,
    goals=goals,
    xs=xs,
    ys=ys,
)

fig, ax = plt.subplots(1, 4)

# combined optimal path and wall gradient
directions_combined = solved_mazes[0, 0, :, :, :].reshape(res * res, 2)

# just the wall gradient
directions_gradient = wall_gradient[0, 0, :, :, :].reshape(res * res, 2)

# original solution
directions_original = solved_mazes_original[0, 0, :, :, :].reshape(res * res, 2)

# combined with walls zerod out for clarity
directions_combined_overlay = solved_mazes[0, 0, :, :, :].copy()
walls = np.where(fine_mazes[0, :, :] == 1)
directions_combined_overlay[walls[0], walls[1], :] = 0
directions_combined_overlay = directions_combined_overlay.reshape(res * res, 2)

coords = np.zeros((res * res, 2))
# NOTE: there should be a better way to do this than a loop
for xi, x in enumerate(xs):
    for yi, y in enumerate(ys):
        coords[xi * res + yi, 0] = x
        coords[xi * res + yi, 1] = y

# plt.imshow(wall_energy[0, :, :])
# plt.show()

# view a sample (original, gradient, combined, combined with wall overlay)
plot_path_predictions(
    directions=directions_original, coords=coords, name='', min_val=-1, max_val=1,
    type='quiver',
    # type='colour',
    dcell=xs[1] - xs[0],
    ax=ax[0], wall_overlay=None
)
plot_path_predictions(
    directions=directions_gradient, coords=coords, name='', min_val=-1, max_val=1,
    type='quiver',
    # type='colour',
    dcell=xs[1] - xs[0],
    ax=ax[1], wall_overlay=None
)
plot_path_predictions(
    directions=directions_combined, coords=coords, name='', min_val=-1, max_val=1,
    type='quiver',
    # type='colour',
    dcell=xs[1] - xs[0],
    ax=ax[2], wall_overlay=None
)
plot_path_predictions(
    directions=directions_combined_overlay, coords=coords, name='', min_val=-1, max_val=1,
    type='quiver',
    # type='colour',
    dcell=xs[1] - xs[0],
    ax=ax[3], wall_overlay=None
)

plt.show()
