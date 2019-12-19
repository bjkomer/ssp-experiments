import numpy as np


def get_activations(data, encoding_func, encoding_dim):
    """
    :param data: dataset
    :param encoding_func: spatial encoding function
    :return: spatial activations for every location in the dataset
    """

    # shape is (n_trajectories, n_steps, pos)
    positions = data['positions']

    n_trajectories = positions.shape[0]
    n_steps = positions.shape[1]

    flat_pos = np.zeros((n_trajectories * n_steps, 2))

    activations = np.zeros((n_trajectories * n_steps, encoding_dim))

    for ti in range(n_trajectories):
        for si in range(n_steps):
            activations[ti*n_steps + si, :] = encoding_func(x=positions[ti, si, 0], y=positions[ti, si, 1])
            # guarantee the flattening is done correctly
            flat_pos[ti*n_steps + si, :] = positions[ti, si, :]

    return activations, flat_pos


def spatial_heatmap(activations, positions, xs, ys):

    # activations is (n_trajectories * n_steps, n_components)
    # positions is (n_trajectories * n_steps, pos)

    heatmap = np.zeros((activations.shape[1], len(xs), len(ys)))

    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):

            indices = np.where((positions[:, 0] > x - dx) & (positions[:, 0] < x + dx) & (positions[:, 1] > y - dy) & (positions[:, 1] < y + dy))[0]
            n_samples = len(indices)

            # print(indices)
            #
            # # print(indices.shape)
            # print(positions.shape)
            # print(activations.shape)
            # print(activations[indices, :].shape)
            # print(activations[indices, :].sum(axis=0).shape)

            heatmap[:, i, j] = activations[indices, :].sum(axis=0) / n_samples

    return heatmap
