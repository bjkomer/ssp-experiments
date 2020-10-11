import numpy as np
import matplotlib.pyplot as plt
from spatial_semantic_pointers.utils import get_heatmap_vectors, get_fixed_dim_sub_toriod_axes, make_good_unitary


n_rows = 4
n_cols = 4
fig, ax = plt.subplots(n_rows, n_cols, figsize=(4, 4), tight_layout=True)

dim = 256
dim = 1024
dim = 512
dim = 2048
# dim = 16
X, Y = get_fixed_dim_sub_toriod_axes(
        dim=dim,
        n_proj=3,
        scale_ratio=0,
        scale_start_index=0,
        rng=np.random.RandomState(seed=13),
        eps=0.001,
)
# X = make_good_unitary(dim=dim)
# Y = make_good_unitary(dim=dim)

res = 256#128
limit = 25#15#5
xs = np.linspace(-limit, limit, res)
ys = np.linspace(-limit, limit, res)
heatmap_vectors = get_heatmap_vectors(xs, ys, X, Y)
print(heatmap_vectors.shape)

for r in range(n_rows):
    for c in range(n_cols):

        ax[r, c].imshow(heatmap_vectors[:, :, r*n_cols+c])
        ax[r, c].set_axis_off()

# ax[-1, -1].imshow(heatmap_vectors[:, :, -1])


plt.show()
