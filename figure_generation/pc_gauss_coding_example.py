import numpy as np
import matplotlib.pyplot as plt

res = 128

# loc = [40, 50]
loc = [75, 50]

sigma = 20

n_gauss = 6

min_alpha = 0.05

seed = 13#21#20

rng = np.random.RandomState(seed=seed)

# force the centers to be more in the middle, since this is just an example
# centers = rng.uniform(0+sigma*1.5, res-sigma*1.5, size=(n_gauss, 2))

centers = np.array([
    [40, 40],
    [40, 80],
    [70, 50],  # [60, 60]
    [60, 30],
    [80, 40],
    [80, 80],
])

colours = [
    'r', 'b', 'g', 'orange', 'purple', 'cyan'
]

centers = centers + rng.normal(0, 5, size=(n_gauss, 2))

fig, ax = plt.subplots()
ax.set_xlim([0, res])
ax.set_ylim([0, res])

activations = np.ones((n_gauss,)) * min_alpha

# TODO: use real activations
activations[2] = .50
activations[3] = .20
activations[4] = .15

for i in range(n_gauss):
    ax.add_artist(plt.Circle((centers[i, 0], centers[i, 1]), sigma, color=colours[i], alpha=activations[i]))
    # Solid dot for center of the gaussian
    # plt.scatter(x=centers[i, 0], y=centers[i, 1], color=colours[i], alpha=activations[i])


plt.scatter(x=loc[0], y=loc[1], color='black')

plt.show()
