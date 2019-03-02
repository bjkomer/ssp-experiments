import argparse
from path_utils import plot_path_predictions, generate_maze_sp, solve_maze
import numpy as np
from spatial_semantic_pointers.utils import make_good_unitary, encode_point
import os

parser = argparse.ArgumentParser(
    'Generate random mazes, and their solutions for particular goal locations'
)

parser.add_argument('--maze-size', type=int, default=10, help='Size of the coarse maze structure')
parser.add_argument('--res', type=int, default=64, help='resolution of the fine maze')
parser.add_argument('--limit-low', type=float, default=-5, help='lowest coordinate value')
parser.add_argument('--limit-high', type=float, default=5, help='highest coordinate value')
parser.add_argument('--n-mazes', type=int, default=10, help='number of different maze configurations')
parser.add_argument('--n-goals', type=int, default=25, help='number of different goal locations for each maze')
parser.add_argument('--save-dir', type=str, default='maze_datasets')
parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--dim', type=int, default=512)

args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

fname = 'maze_dataset_{}mazes_{}goals_{}res_{}seed.npz'.format(args.n_mazes, args.n_goals, args.res, args.seed)

rng = np.random.RandomState(seed=args.seed)

x_axis_sp = make_good_unitary(dim=args.dim, rng=rng)
y_axis_sp = make_good_unitary(dim=args.dim, rng=rng)

np.random.seed(args.seed)

xs = np.linspace(args.limit_low, args.limit_high, args.res)
ys = np.linspace(args.limit_low, args.limit_high, args.res)

coarse_mazes = np.zeros((args.n_mazes, args.maze_size, args.maze_size))
fine_mazes = np.zeros((args.n_mazes, args.res, args.res))
solved_mazes = np.zeros((args.n_mazes, args.n_goals, args.res, args.res, 2))

# locs = np.zeros((args.n_mazes, args.n_goals, 2))
goals = np.zeros((args.n_mazes, args.n_goals, 2))
# loc_sps = np.zeros((args.n_mazes, args.n_goals, args.dim))
goal_sps = np.zeros((args.n_mazes, args.n_goals, args.dim))

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

    # current_loc_indices = np.random.randint(low=0, high=n_free_spaces, size=args.n_goals)
    goal_indices = np.random.randint(low=0, high=n_free_spaces, size=args.n_goals)

    for n in range(args.n_goals):
        print("Generating Sample {} of {}".format(n+1, args.n_goals))
        # 2D coordinate of the goal
        goal_index = free_spaces[goal_indices[n], :]
        goal_x = xs[goal_index[0]]
        goal_y = ys[goal_index[1]]


        # Compute the optimal path given this goal
        # Full solve is set to true, so start_indices is ignored
        solved_maze = solve_maze(fine_maze, start_indices=goal_index, goal_indices=goal_index, full_solve=True)

        goals[mi, n, 0] = goal_x
        goals[mi, n, 1] = goal_y
        goal_sps[mi, n, :] = encode_point(goal_x, goal_y, x_axis_sp, y_axis_sp).v

        solved_mazes[mi, n, :, :, :] = solved_maze

np.savez(
    os.path.join(args.save_dir, fname),
    coarse_mazes=coarse_mazes,
    fine_mazes=fine_mazes,
    solved_mazes=solved_mazes,
    x_axis_sp=x_axis_sp.v,
    y_axis_sp=y_axis_sp.v,
    goal_sps=goal_sps,
    goals=goals,
)
