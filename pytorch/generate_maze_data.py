import argparse
from path_utils import plot_path_predictions, generate_maze_sp, solve_maze
import numpy as np
from spatial_semantic_pointers.utils import make_good_unitary

parser = argparse.ArgumentParser(
    'Generate random mazes, and their solutions for particular goal locations'
)

parser.add_argument('--maze-size', type=int, default=10, help='Size of the coarse maze structure')
parser.add_argument('--res', type=int, default=64, help='resolution of the fine maze')
parser.add_argument('--limit-low', type=float, default=-5, help='lowest coordinate value')
parser.add_argument('--limit-high', type=float, default=5, help='highest coordinate value')
parser.add_argument('--n-mazes', type=int, default=10, help='number of different maze configurations')
parser.add_argument('--n-goals', type=int, default=25, help='number of different goal locations for each maze')

args = parser.parse_args()

rng = np.random.RandomState(seed=args.seed)

x_axis_sp = make_good_unitary(dim=args.dim, rng=rng)
y_axis_sp = make_good_unitary(dim=args.dim, rng=rng)

np.random.seed(args.seed)

xs = np.linspace(args.limit_low, args.limit_high, args.res)
ys = np.linspace(args.limit_low, args.limit_high, args.res)

coarse_mazes = np.zeros((args.n_mazes, args.maze_size, args.maze_size))
fine_mazes = np.zeros((args.n_mazes, args.res, args.res))

for mi in range(args.n_mazes):
    # Generate a random maze
    maze_ssp, coarse_maze, fine_maze = generate_maze_sp(
        size=args.maze_size,
        xs=xs,
        ys=ys,
        x_axis_sp=x_axis_sp,
        y_axis_sp=y_axis_sp,
        normalize=True,
        obstacle_ratio=.2,
        map_style='blocks'
    )

    # Get a list of possible goal locations to choose (will correspond to all free spaces in the coarse maze)
    # free_spaces = np.argwhere(coarse_maze == 0)
    # Get a list of possible goal locations to choose (will correspond to all free spaces in the fine maze)
    free_spaces = np.argwhere(fine_maze == 0)
    print(free_spaces.shape)
    n_free_spaces = free_spaces.shape[0]
