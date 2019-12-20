import numpy as np
from ssp_navigation.utils.encodings import get_encoding_function


class Environment(object):

    def __init__(self, encoding_func, limit_low=0, limit_high=2.2, periodic_boundary=True, seed=13,
                 lin_vel_rayleigh_scale=0.13, rot_vel_std=330, dt=0.02):

        self.encoding_func = encoding_func
        self.limit_low = limit_low
        self.limit_high = limit_high
        self.side_len = limit_high - limit_low
        self.periodic_boundary = periodic_boundary
        self.lin_vel_rayleigh_scale = lin_vel_rayleigh_scale
        self.rot_vel_std = rot_vel_std
        self.dt = dt

        self.rng = np.random.RandomState(seed=seed)

        # initialize the agent to the center of the environment
        self.pos = np.array([(limit_high + limit_low) / 2, (limit_high + limit_low) / 2])
        # initialize the agent heading
        self.angle = 0

    def step(self):
        """
        Move the agent by one timestep, and return both the true position and the encoding activation
        """

        lin_vel = self.rng.rayleigh(scale=self.lin_vel_rayleigh_scale)
        ang_vel = self.rng.normal(loc=0, scale=self.rot_vel_std) * np.pi / 180

        self.pos[0] = self.pos[0] + lin_vel * np.cos(self.angle) * self.dt
        self.pos[1] = self.pos[1] + lin_vel * np.sin(self.angle) * self.dt

        self.angle = self.angle + ang_vel * self.dt

        if self.angle > np.pi:
            self.angle -= 2*np.pi
        elif self.angle < -np.pi:
            self.angle += 2*np.pi

        if self.periodic_boundary:
            if self.pos[0] > self.limit_high:
                self.pos[0] = self.pos[0] - self.side_len
            elif self.pos[0] < self.limit_low:
                self.pos[0] = self.pos[0] + self.side_len
            if self.pos[1] > self.limit_high:
                self.pos[1] = self.pos[1] - self.side_len
            elif self.pos[1] < self.limit_low:
                self.pos[1] = self.pos[1] + self.side_len
        else:
            # just smush into the walls for now
            if self.pos[0] > self.limit_high:
                self.pos[0] = self.limit_high
            elif self.pos[0] < self.limit_low:
                self.pos[0] = self.limit_low
            if self.pos[1] > self.limit_high:
                self.pos[1] = self.limit_high
            elif self.pos[1] < self.limit_low:
                self.pos[1] = self.limit_low

        activations = self.encoding_func(x=self.pos[0], y=self.pos[1])

        return activations, self.pos


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

            if n_samples > 0:
                heatmap[:, i, j] = activations[indices, :].sum(axis=0) / n_samples

    return heatmap
