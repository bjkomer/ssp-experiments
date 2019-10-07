import argparse
import numpy as np
import matplotlib.pyplot as plt

from ssp_navigation.utils.path import generate_maze_sp, solve_maze
from ssp_navigation.utils.misc import gaussian_2d
from spatial_semantic_pointers.utils import make_good_unitary, encode_point
from ssp_navigation.utils.path import plot_path_predictions, plot_path_predictions_image

parser = argparse.ArgumentParser(
    'Figures showing an optimal policy, before and after the energy function'
)

parser.add_argument('--maze-size', type=int, default=13, help='Size of the coarse maze structure')
parser.add_argument('--map-style', type=str, default='maze', choices=['blocks', 'maze', 'mixed'], help='Style of maze')
parser.add_argument('--res', type=int, default=64, help='resolution of the fine maze')
parser.add_argument('--seed', type=int, default=13)
# Parameters for modifying optimal path to not cut corners
parser.add_argument('--energy-sigma', type=float, default=0.25, help='std for the wall energy gaussian')
parser.add_argument('--energy-scale', type=float, default=0.75, help='scale for the wall energy gaussian')

args = parser.parse_args()


# hackily pasted and modified from the dataset generation script

print("Generating maze data")

rng = np.random.RandomState(seed=args.seed)

# these aren't used, only as an argument to the generate_maze_sp function
x_axis_sp = make_good_unitary(dim=4, rng=rng)
y_axis_sp = make_good_unitary(dim=4, rng=rng)

np.random.seed(args.seed)

limit_low = 0
limit_high = args.maze_size

xs_coarse = np.linspace(limit_low, limit_high, args.maze_size)
ys_coarse = np.linspace(limit_low, limit_high, args.maze_size)

xs = np.linspace(limit_low, limit_high, args.res)
ys = np.linspace(limit_low, limit_high, args.res)

coarse_mazes = np.zeros((args.maze_size, args.maze_size))
fine_mazes = np.zeros((args.res, args.res))
solved_mazes = np.zeros((args.res, args.res, 2))


goals = np.zeros((2, ))

# Generate a random maze
if args.map_style == 'mixed':
    map_style = np.random.choice(['blocks', 'maze'])
else:
    map_style = args.map_style
maze_ssp, coarse_maze, fine_maze = generate_maze_sp(
    size=args.maze_size,
    xs=xs,
    ys=ys,
    x_axis_sp=x_axis_sp,
    y_axis_sp=y_axis_sp,
    normalize=True,
    obstacle_ratio=.2,
    map_style=map_style,
)
coarse_mazes[:, :] = coarse_maze
fine_mazes[:, :] = fine_maze

# Get a list of possible goal locations to choose (will correspond to all free spaces in the coarse maze)
# free_spaces = np.argwhere(coarse_maze == 0)
# Get a list of possible goal locations to choose (will correspond to all free spaces in the fine maze)
free_spaces = np.argwhere(fine_maze == 0)
print(free_spaces.shape)
# downsample free spaces
free_spaces = free_spaces[(free_spaces[:, 0] % 4 == 0) & (free_spaces[:, 1] % 4 == 0)]
print(free_spaces.shape)

n_free_spaces = free_spaces.shape[0]

# no longer choosing indices based on any fine location, so that the likelihood of overlap between mazes increases
# which should decrease the chances of overfitting based on goal location

goal_index = np.random.randint(0, n_free_spaces)

# 2D coordinate of the goal
goal_index = free_spaces[goal_index, :]

goal_x = xs[goal_index[0]]
goal_y = ys[goal_index[1]]

# make sure the goal is placed in an open space
assert(fine_maze[goal_index[0], goal_index[1]] == 0)

# Compute the optimal path given this goal
# Full solve is set to true, so start_indices is ignored
print("Solving Maze")
solved_maze = solve_maze(fine_maze, start_indices=goal_index, goal_indices=goal_index, full_solve=True)

goals[0] = goal_x
goals[1] = goal_y

solved_mazes[:, :, :] = solved_maze

# Compute energy function to avoid walls

# will contain a sum of gaussians placed at all wall locations
wall_energy = np.zeros_like(fine_mazes)

# will contain directions corresponding to the gradient of the wall energy
# to be added to the solved maze to augment the directions chosen
wall_gradient = np.zeros_like(solved_mazes)

meshgrid = np.meshgrid(xs, ys)

locs = np.zeros((len(xs), len(ys), 2))

print("Computing Wall Energy")
# Compute wall energy
for xi, x in enumerate(xs):
    for yi, y in enumerate(ys):
        locs[xi, yi, 0] = x
        locs[xi, yi, 1] = y
        if fine_mazes[xi, yi]:
            wall_energy[:, :] += args.energy_scale * gaussian_2d(x=x, y=y, meshgrid=meshgrid,
                                                                     sigma=args.energy_sigma)

# Compute wall gradient using wall energy


gradients = np.gradient(wall_energy[:, :])

x_grad = gradients[0]
y_grad = gradients[1]

# Note all the flipping and transposing to get things right
wall_gradient[:, :, 1] = -x_grad.T
wall_gradient[:, :, 0] = -y_grad.T

# modify the maze solution with the gradient
combined_solved_mazes = solved_mazes + wall_gradient

print("Re-normalizing the desired directions")
# re-normalize the desired directions
for xi in range(args.res):
    for yi in range(args.res):
        norm = np.linalg.norm(solved_mazes[xi, yi, :])
        # If there is a wall, make the output be (0, 0)
        if fine_mazes[xi, yi]:
            solved_mazes[xi, yi, 0] = 0
            solved_mazes[xi, yi, 1] = 0
            combined_solved_mazes[xi, yi, 0] = 0
            combined_solved_mazes[xi, yi, 1] = 0
        elif norm != 0:
            solved_mazes[xi, yi, :] /= np.linalg.norm(solved_mazes[xi, yi, :])
            combined_solved_mazes[xi, yi, :] /= np.linalg.norm(combined_solved_mazes[xi, yi, :])

print("Generating Figures")

wall_overlay = (fine_mazes.reshape(args.res*args.res) != 0)

fig_truth, rmse = plot_path_predictions_image(
    directions_pred=solved_mazes.reshape(args.res*args.res, 2),
    directions_true=solved_mazes.reshape(args.res*args.res, 2),
    wall_overlay=wall_overlay
)

fig_truth, rmse = plot_path_predictions_image(
    directions_pred=combined_solved_mazes.reshape(args.res*args.res, 2),
    directions_true=combined_solved_mazes.reshape(args.res*args.res, 2),
    wall_overlay=wall_overlay
)

fig_truth_quiver = plot_path_predictions(
    directions=solved_mazes.reshape(args.res*args.res, 2), coords=locs.reshape(args.res*args.res, 2), dcell=xs[1] - xs[0]
)

fig_truth_quiver = plot_path_predictions(
    directions=combined_solved_mazes.reshape(args.res*args.res, 2), coords=locs.reshape(args.res*args.res, 2), dcell=xs[1] - xs[0]
)

plt.show()
