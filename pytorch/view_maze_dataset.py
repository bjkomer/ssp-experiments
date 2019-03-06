import sys
import numpy as np
import matplotlib.pyplot as plt
from path_utils import plot_path_predictions

fname = sys.argv[1]

data = np.load(fname)

# n_mazes by res by res
fine_mazes = data['fine_mazes']

# n_mazes by n_goals res by res by 2
solved_mazes = data['solved_mazes']

# n_mazes by n_goals by 2
goals = data['goals']

n_mazes = solved_mazes.shape[0]
n_goals = solved_mazes.shape[1]
res = solved_mazes.shape[2]

xs = data['xs']
ys = data['ys']

viz_locs = np.zeros((res*res, n_mazes, n_goals, 2))
# viz_goals = np.zeros((n_mazes, n_goals, 2))
viz_output_dirs = np.zeros((res*res, n_mazes, n_goals, 2))


# colour figures
fig, ax = plt.subplots(n_mazes, n_goals)

# quiver figures
fig_q, ax_q = plt.subplots(n_mazes, n_goals)

subsample = 1
for mi in range(n_mazes):
    for gi in range(n_goals):
        for xi in range(0, res, subsample):
            for yi in range(0, res, subsample):
                loc_x = xs[xi]
                loc_y = ys[yi]

                viz_locs[xi*res+yi, mi, gi, 0] = loc_x
                viz_locs[xi*res+yi, mi, gi, 1] = loc_y
                # viz_goals[mi, gi, :] = goals[mi, gi, :]

                viz_output_dirs[xi*res+yi, mi, gi, :] = solved_mazes[mi, gi, xi, yi, :]

        ax[mi, gi].set_xlim([xs[0], xs[-1]])
        ax[mi, gi].set_ylim([ys[0], ys[-1]])
        plot_path_predictions(
            directions=viz_output_dirs[:, mi, gi, :],
            coords=viz_locs[:, mi, gi, :], type='colour', ax=ax[mi, gi]
        )

        plot_path_predictions(
            directions=viz_output_dirs[:, mi, gi, :],
            coords=viz_locs[:, mi, gi, :], dcell=xs[1] - xs[0], ax=ax_q[mi, gi]
        )

plt.show()
