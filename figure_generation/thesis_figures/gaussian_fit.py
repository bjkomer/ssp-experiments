import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from spatial_semantic_pointers.utils import make_good_unitary, get_axes, get_heatmap_vectors, power, encode_point_hex


def gaussian_1d(x, sigma, linspace):
    return np.exp(-((linspace - x) ** 2) / sigma / sigma)


def gaussian_2d(x, y, sigma, meshgrid):
    X, Y = meshgrid
    return np.exp(-((X - x) ** 2 + (Y - y) ** 2) / sigma / sigma)


# fname = '/home/bjkomer/ssp_navigation_sandbox/axis_vector_explorations/heatmap_output/256dim_25seeds.npz'
# data = np.load(fname)
# square_heatmaps = data['square_heatmaps']
# hex_heatmaps = data['hex_heatmaps']
#
# avg_square_heatmap = square_heatmaps.mean(axis=0)
# avg_hex_heatmap = hex_heatmaps.mean(axis=0)

dim = 512
res = 256
limit = 5
xs = np.linspace(-limit, limit, res)

Xh, Yh = get_axes(dim=dim, seed=13)

rng = np.random.RandomState(seed=13)
X = make_good_unitary(dim=dim, rng=rng)
Y = make_good_unitary(dim=dim, rng=rng)
Z = make_good_unitary(dim=dim, rng=rng)

fig, ax = plt.subplots(1, 3, figsize=(8, 4))

sigma_normal = 0.5
sigma_hex = 0.5
sigma_hex_c = 0.5

sim_hex = np.zeros((res, ))
sim_hex_c = np.zeros((res, ))  # this version has axes generated together and then converted to 2D
sim_normal = np.zeros((res, ))
gauss_hex = gaussian_1d(0, sigma_hex, xs)
gauss_hex_c = gaussian_1d(0, sigma_hex_c, xs)
gauss_normal = gaussian_1d(0, sigma_normal, xs)

for i, x, in enumerate(xs):
    sim_normal[i] = power(X, x).v[0]
    sim_hex_c[i] = power(Xh, x).v[0]
    sim_hex[i] = encode_point_hex(0, x, X, Y, Z).v[0]

ax[0].plot(xs, sim_normal)
ax[0].plot(xs, gauss_normal)
ax[1].plot(xs, sim_hex)
ax[1].plot(xs, gauss_hex)
ax[2].plot(xs, sim_hex_c)
ax[2].plot(xs, gauss_hex_c)

for i in range(len(ax)):
    ax[i].set_ylim([-.2, 1.2])

plt.show()
