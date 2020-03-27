import numpy as np
import matplotlib.pyplot as plt
from ssp_navigation.utils.encodings import hilbert_2d

limit_high = 1
limit_low = 0

seed = 13
rng = np.random.RandomState(seed=13)

dim = 256

fig, ax = plt.subplots(1, 4)

pc_centers_random = rng.uniform(low=limit_low, high=limit_high, size=(dim, 2))
pc_centers_hilbert = hilbert_2d(limit_low=limit_low, limit_high=limit_high, n_samples=dim, rng=rng)
side_len = int(np.sqrt(dim))
vx, vy = np.meshgrid(np.linspace(limit_low, limit_high, side_len), np.linspace(limit_low, limit_high, side_len))
pc_centers_tiled = np.vstack([vx.flatten(), vy.flatten()]).T

half_dim = dim / 2.
res_y = np.sqrt(half_dim * np.sqrt(3)/3.)
res_x = half_dim / res_y
# need these to be integers, err on the side of too many points and trim after
res_y = int(np.ceil(res_y))
res_x = int(np.ceil(res_x))

xs = np.linspace(limit_low, limit_high, res_x)
ys = np.linspace(limit_low, limit_high, res_y)

vx, vy = np.meshgrid(xs, ys)

# scale the sides of each hexagon
# in the y direction, hexagon centers are 3 units apart
# in the x direction, hexagon centers are sqrt(3) units apart
scale = (ys[1] - ys[0]) / 3.

pc_centers_a = np.vstack([vx.flatten(), vy.flatten()]).T
pc_centers_b = pc_centers_a + np.array([np.sqrt(3)/2., 3./2.]) * scale

# place the two offset grids together, and cut off excess points so that there are only 'dim' centers
pc_centers_hex_tiled = np.concatenate([pc_centers_a, pc_centers_b], axis=0)[:dim, :]

ax[0].scatter(pc_centers_random[:, 0], pc_centers_random[:, 1])
ax[0].set_title("Random Uniform")
ax[1].scatter(pc_centers_hilbert[:, 0], pc_centers_hilbert[:, 1])
ax[1].set_title("Hilbert Curve")
ax[2].scatter(pc_centers_tiled[:, 0], pc_centers_tiled[:, 1])
ax[2].set_title("Square Tiling")
ax[3].scatter(pc_centers_hex_tiled[:, 0], pc_centers_hex_tiled[:, 1])
ax[3].set_title("Hexagonal Tiling")
for i in range(4):
    ax[i].set_xlim([limit_low, limit_high])
    ax[i].set_ylim([limit_low, limit_high])
    ax[i].set_aspect('equal')

plt.show()
