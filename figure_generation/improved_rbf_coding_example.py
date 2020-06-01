import numpy as np
import matplotlib.pyplot as plt
from ssp_navigation.utils.encodings import hilbert_2d
import seaborn as sns

dim = 16#64
limit_low = 0
limit_high = 128

# colours = sns.color_palette("hls", n_colors=dim)
# colours = sns.color_palette(n_colors=dim)
# colours = sns.color_palette("Paired", n_colors=dim)

colours = [
    sns.color_palette("muted", n_colors=dim//2),
    sns.color_palette("colorblind", n_colors=dim//2)
]

loc = [40, 50]

rng = np.random.RandomState(seed=13)
sigma = 30

pc_centers = hilbert_2d(limit_low=limit_low, limit_high=limit_high, n_samples=dim, rng=rng)

activations = np.zeros((dim,))
for i in range(dim):
    activations[i] = np.exp(-((pc_centers[i, 0] - loc[0]) ** 2 + (pc_centers[i, 1] - loc[1]) ** 2) / sigma / sigma)

fig, ax = plt.subplots(1, 2, figsize=(6, 4), tight_layout=True, gridspec_kw={'width_ratios': [16, 5]})
res = 128
ax[0].set_xlim([0, res])
ax[0].set_ylim([0, res])
print(np.min(activations))
print(np.max(activations))
for i in range(dim):
    ax[0].add_artist(plt.Circle((pc_centers[i, 0], pc_centers[i, 1]), sigma, color=colours[i//(dim//2)][int(i % (dim//2))], alpha=activations[i]))

# plt.scatter(x=loc[0], y=loc[1], color='black')
ax[0].add_artist(plt.Circle((loc[0], loc[1]), 2, color='black', alpha=1))
ax[0].set_aspect('equal')
# show the values of each gaussian

ax[1].set_axis_off()
ax[1].set_xlim([0, res])
ax[1].set_ylim([0, res])
side_len = (limit_high/dim)
for i in range(dim):
    ax[1].add_artist(plt.Rectangle((0, i*side_len), width=side_len*2, height=side_len, color=colours[i//(dim//2)][int(i % (dim//2))], alpha=1))
    ax[1].annotate('{:.3f}'.format(activations[i]), ((side_len*2)*1.2, i*side_len), color='black', weight='bold',
                fontsize=12, ha='left', va='bottom')

plt.show()
