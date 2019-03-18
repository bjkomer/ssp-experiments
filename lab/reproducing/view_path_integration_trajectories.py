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

# fig, ax = plt.subplots()
# ax.set_xlim([0, 2.2])
# ax.set_ylim([0, 2.2])

max_pc_activations = np.max(pc_activations, axis=1)

for n in range(n_trajectories):
    # colour the points based on time. Purple will be the start, and yellow will be the end
    plt.scatter(positions[n, :, 0], positions[n, :, 1], c=np.arange(n_steps))
    plt.xlim([0, 2.2])
    plt.ylim([0, 2.2])

    if view_activations:
        plt.scatter(pc_centers[:, 0], pc_centers[:, 1], c=max_pc_activations[n, :])


    plt.show()
