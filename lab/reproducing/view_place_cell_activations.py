import numpy as np
import matplotlib.pyplot as plt

view_activations = True

data = np.load('data/path_integration_trajectories_200t_15s.npz')

positions = data['positions']

pc_centers = data['pc_centers']
hd_centers = data['hd_centers']
pc_activations = data['pc_activations']
hd_activations = data['hd_activations']

n_trajectories = positions.shape[0]
n_steps = positions.shape[1]
n_place_cells = pc_centers.shape[0]

# TODO: make sure the reshaping isn't misaligning things
flat_positions = positions.reshape((n_trajectories * n_steps, 2))
flat_activations = pc_activations.reshape((n_trajectories * n_steps, n_place_cells))

for pi in range(n_place_cells):
    # colour the points of the trajectories based on how much this particular place cell fired
    plt.scatter(flat_positions[:, 0], flat_positions[:, 1], c=flat_activations[:, pi], s=1)
    plt.xlim([0, 2.2])
    plt.ylim([0, 2.2])

    # also plot the place cell itself
    plt.scatter(pc_centers[pi, 0], pc_centers[pi, 1], marker='*')

    plt.show()
