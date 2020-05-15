from encoders import grid_cell_encoder, band_cell_encoder, orthogonal_hex_dir, orthogonal_hex_dir_7dim
import numpy as np
import matplotlib.pyplot as plt
from spatial_semantic_pointers.utils import make_good_unitary, encode_point, get_heatmap_vectors, get_axes




dim = 13
# dim = 7

# X, Y = orthogonal_hex_dir(phis=(np.pi / 2., np.pi/8.), angles=(0, np.pi/3.))
# X, Y = orthogonal_hex_dir(phis=(np.pi / 2., np.pi/3.), angles=(0, np.pi/3.))
X, Y = orthogonal_hex_dir(phis=(np.pi / 2., np.pi/3., np.pi/4.), angles=(0, np.pi/3., np.pi/5.))
# X, Y = orthogonal_hex_dir(phis=(np.pi / 2.,), angles=(0,))
# X, Y = orthogonal_hex_dir(phis=(np.pi / 2.,), angles=(np.pi/3.,))
dim = len(X.v)

# X, Y, sv = orthogonal_hex_dir_7dim(phi=np.pi / 2., angle=0)

# encoder = grid_cell_encoder(dim=dim, phases=(0, 0, 0), toroid_index=0)
# encoder = grid_cell_encoder(dim=dim, phases=(np.pi/4., np.pi/2., 0), toroid_index=0)
# encoder = grid_cell_encoder(dim=dim, phases=(np.pi/4., np.pi/2., np.pi/3.), toroid_index=0)
# encoder = grid_cell_encoder(dim=dim, phases=(np.pi/8., np.pi/2., np.pi/3.), toroid_index=0)
# encoder = grid_cell_encoder(dim=dim, phases=(np.pi/8., np.pi/2., np.pi/3.), toroid_index=1)
# encoder = grid_cell_encoder(dim=dim, phases=(np.pi/6., np.pi/2., np.pi/3.), toroid_index=2)
# encoder = grid_cell_encoder(dim=dim, phases=(0, 0, 0), toroid_index=2)
# encoder = grid_cell_encoder(dim=dim, phases=(np.pi/2., np.pi/2., np.pi/2.), toroid_index=0)

# encoder = band_cell_encoder(dim=dim, phase=0, toroid_index=0, band_index=0)


limit = 10#5
res = 128
xs = np.linspace(-limit, limit, res)
ys = np.linspace(-limit, limit, res)

hmv = get_heatmap_vectors(xs, ys, X, Y)

# encoder = grid_cell_encoder(dim=dim, phases=(np.pi/1., np.pi/1., np.pi/1.), toroid_index=1)
# encoder = grid_cell_encoder(dim=dim, phases=(0, 0, 0), toroid_index=0)
# encoder = band_cell_encoder(dim=dim, phase=0, toroid_index=0, band_index=0)

encoder = grid_cell_encoder(dim=dim, phi=np.pi / 2., angle=np.pi/3., location=(0, 0), toroid_index=0)

plt.figure()
sim = np.tensordot(encoder, hmv, axes=([0], [2]))
plt.imshow(sim)

# encoder = grid_cell_encoder(dim=dim, phases=(np.pi/2., np.pi/2., np.pi/2.), toroid_index=1)
# encoder = grid_cell_encoder(dim=dim, phases=(np.pi/1., np.pi/1., np.pi/1.), toroid_index=1)
# encoder = grid_cell_encoder(dim=dim, phases=(np.pi/1., 0, 0), toroid_index=0)
# encoder = grid_cell_encoder(dim=dim, phases=(np.pi/1., np.pi/1., np.pi/1.), toroid_index=0)
# encoder = grid_cell_encoder(dim=dim, phases=(0, np.pi/1., np.pi/1.), toroid_index=0)
# encoder = grid_cell_encoder(dim=dim, phases=(0, np.pi/1., 0), toroid_index=0)
# encoder = band_cell_encoder(dim=dim, phase=1*np.pi/1., toroid_index=0, band_index=0)

encoder = grid_cell_encoder(dim=dim, phi=np.pi / 2., angle=np.pi/6., location=(1, 1), toroid_index=0)

plt.figure()
sim = np.tensordot(encoder, hmv, axes=([0], [2]))
plt.imshow(sim)

plt.show()
