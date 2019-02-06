import numpy as np
import matplotlib.pyplot as plt


# Up, Down, Left, Right
U = 1
L = 2
D = 3
R = 4

example_path = np.array([
    [R, R, R, R, R, D],
    [U, U, 0, R, R, 0],
    [U, 0, 0, 0, 0, 0],
    [U, L, L, L, L, L],
    [U, 0, 0, 0, 0, 0],
    [U, L, L, L, L, L],
])

example_xs = np.linspace(-1, 1, example_path.shape[0])
example_ys = np.linspace(-1, 1, example_path.shape[1])


def dir_to_vec(direction):
    if direction == U:
        return np.array([0, 1])
    elif direction == L:
        return np.array([-1, 0])
    elif direction == R:
        return np.array([1, 0])
    elif direction == D:
        return np.array([0, -1])
    elif direction == 0:
        return np.array([0, 0])
    else:
        raise NotImplementedError


def path_function(coord, path, xs, ys):
    ind_x = (np.abs(xs - coord[0])).argmin()
    ind_y = (np.abs(ys - coord[0])).argmin()

    return dir_to_vec(path[ind_x, ind_y])


def plot_path_predictions(directions, coords, name='', min_val=-1, max_val=1):
    """
    plot direction predictions by colouring based on direction, and putting the dot at the coord
    both directions and coords are (n_samples, 2) vectors
    """
    fig, ax = plt.subplots()

    for n in range(directions.shape[0]):
        x = coords[n, 0]
        y = coords[n, 1]

        # Note: this clipping shouldn't be necessary
        xa = np.clip(directions[n, 0], min_val, max_val)
        ya = np.clip(directions[n, 1], min_val, max_val)

        r = float(((xa - min_val) / (max_val - min_val)))
        # g = float(((ya - min_val) / (max_val - min_val)))
        b = float(((ya - min_val) / (max_val - min_val)))

        # ax.scatter(x, y, color=(r, g, 0))
        ax.scatter(x, y, color=(r, 0, b))

    if name:
        fig.suptitle(name)

    return fig
