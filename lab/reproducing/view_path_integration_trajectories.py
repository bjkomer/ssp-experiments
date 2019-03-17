import numpy as np
import matplotlib.pyplot as plt

data = np.load('data/path_integration_trajectories_200.npz')

positions = data['positions']

n_trajectories = positions.shape[0]
n_steps = positions.shape[1]

# fig, ax = plt.subplots()
# ax.set_xlim([0, 2.2])
# ax.set_ylim([0, 2.2])


for n in range(n_trajectories):
    # colour the points based on time. Purple will be the start, and yellow will be the end
    plt.scatter(positions[n, :, 0], positions[n, :, 1], c=np.arange(n_steps))
    plt.xlim([0, 2.2])
    plt.ylim([0, 2.2])
    plt.show()
