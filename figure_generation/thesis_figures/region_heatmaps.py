import numpy as np
import matplotlib.pyplot as plt
from spatial_semantic_pointers.utils import make_good_unitary, get_axes, get_heatmap_vectors, \
    generate_circular_region_vector, generate_region_vector, circular_region
import imageio
from skimage.transform import resize

seed = 13
dim = 512#1024#512

res = 512#256
limit = 5
xs = np.linspace(-limit, limit, res)
ys = np.linspace(-limit, limit, res)

desired_circular = circular_region(xs, ys, radius=3, x_offset=0, y_offset=0)
desired_rectangular = np.zeros((res, res))
desired_complex = np.zeros((res, res))


# desired_rectangular[30:150, 60:120] = 1
desired_rectangular[30*2:150*2, 60*2:120*2] = 1

# only the 4th channel has the data, values from 0 to 255
desired_complex = resize(imageio.imread('assets/icons8-star-96.png')/255, (res, res))[:, :, 3]

# desired_complex[128-48:128+48, 128-48:128+48] = star_im[:, :, 3]


# print(star_im.shape)
# print(np.min(star_im[:, :, 3]))
# print(np.max(star_im[:, :, 3]))

rng = np.random.RandomState(seed=seed)

# X = make_good_unitary(dim=dim, rng=rng)
# Y = make_good_unitary(dim=dim, rng=rng)

X, Y = get_axes(dim=dim, n=3, seed=seed)

hmv = get_heatmap_vectors(xs, ys, X, Y)

circular_sp = generate_region_vector(desired_circular, xs, ys, X, Y, normalize=True)
rectangular_sp = generate_region_vector(desired_rectangular, xs, ys, X, Y, normalize=True)
complex_sp = generate_region_vector(desired_complex, xs, ys, X, Y, normalize=True)

fig, ax = plt.subplots(3, 3, figsize=(3, 8))

# TODO: change the axes to be the limits instead of the resolution
# TODO: change colour to be consistent with other figures
# TODO: add colourbar

kwargs = {
    'cmap': 'plasma',
    'origin': 'upper',
    'extent': (xs[0], xs[-1], ys[0], ys[-1]),
}

ax[0, 0].imshow(desired_circular, **kwargs)
ax[0, 1].imshow(desired_rectangular, **kwargs)
ax[0, 2].imshow(desired_complex, **kwargs)
ax[1, 0].imshow(np.tensordot(circular_sp.v, hmv, axes=([0], [2])), **kwargs)
ax[1, 1].imshow(np.tensordot(rectangular_sp.v, hmv, axes=([0], [2])), **kwargs)
ax[1, 2].imshow(np.tensordot(complex_sp.v, hmv, axes=([0], [2])), **kwargs)
ax[2, 0].imshow(np.tensordot(circular_sp.v, hmv, axes=([0], [2])), vmin=-1, vmax=1, **kwargs)
ax[2, 1].imshow(np.tensordot(rectangular_sp.v, hmv, axes=([0], [2])), vmin=-1, vmax=1, **kwargs)
ax[2, 2].imshow(np.tensordot(complex_sp.v, hmv, axes=([0], [2])), vmin=-1, vmax=1, **kwargs)


plt.show()
