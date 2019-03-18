import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

view_activations = True

data = np.load('data/path_integration_trajectories_200t_15s.npz')

angles = data['angles']

hd_centers = data['hd_centers']
hd_activations = data['hd_activations']

n_trajectories = angles.shape[0]
n_steps = angles.shape[1]
n_hd_cells = hd_centers.shape[0]

# TODO: make sure the reshaping isn't misaligning things
flat_angles = angles.reshape((n_trajectories * n_steps))
flat_activations = hd_activations.reshape((n_trajectories * n_steps, n_hd_cells))

for hi in range(n_hd_cells):

    # Get all angles where there is some activation
    active_angles = flat_angles[flat_activations[:, hi] > .5]

    print(active_angles.shape)

    # colour the points of the trajectories based on how much this particular place cell fired
    # plt.scatter(flat_positions[:, 0], flat_positions[:, 1], c=flat_activations[:, pi], s=1)

    sns.distplot(active_angles)
    # plt.hist(active_angles, bins=np.arange(-np.pi, np.pi, 36))

    # # also plot the place cell itself
    # plt.scatter(pc_centers[pi, 0], pc_centers[pi, 1], marker='*')

plt.show()
