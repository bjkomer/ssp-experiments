import numpy as np
from spatial_semantic_pointers.utils import make_good_unitary, get_heatmap_vectors_hex, encode_point_hex
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

dim = 1024#4096#1024#512
rng = np.random.RandomState(seed=13)
X = make_good_unitary(dim=dim, rng=rng)
Y = make_good_unitary(dim=dim, rng=rng)
Z = make_good_unitary(dim=dim, rng=rng)

limit = 50
res = 128#128

xs = np.linspace(-limit, limit, res)
ys = np.linspace(-limit, limit, res)

hmv = get_heatmap_vectors_hex(xs, ys, X, Y, Z)

occ_vec = np.zeros((dim, ))


img = mpimg.imread('example_layout.png')

ratio = img.shape[0] // res

for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        if img[i*ratio, j*ratio, 0] == 0:
            occ_vec += encode_point_hex(x, y, X, Y, Z).v

occ_sim = np.tensordot(occ_vec, hmv, axes=(0, 2))
occ_sim /=np.linalg.norm(occ_sim)

# plt.figure()
# plt.imshow(hmv[:, :, 0])
plt.figure()
plt.imshow(occ_sim)
# plt.imshow(occ_sim, vmin=.3, vmax=1)
plt.figure()
plt.imshow(img)
plt.figure()
plt.imshow(img[::ratio,::ratio,:])
plt.show()
