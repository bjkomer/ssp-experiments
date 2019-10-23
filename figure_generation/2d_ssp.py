import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from spatial_semantic_pointers.utils import encode_point, make_good_unitary, power, get_heatmap_vectors

seed = 13
dim = 512
limit = 5
res = 512

vmin=-1
vmax=1
cmap='plasma'

xs = np.linspace(-limit, limit, res)
ys = np.linspace(-limit, limit, res)

x_axis_sp = make_good_unitary(dim)
y_axis_sp = make_good_unitary(dim)

heatmap_vectors = get_heatmap_vectors(xs, ys, x_axis_sp, y_axis_sp)


def plt_heatmap(vec, heatmap_vectors, name='', vmin=-1, vmax=1, cmap='plasma'):
    # vec has shape (dim) and heatmap_vectors have shape (xs, ys, dim) so the result will be (xs, ys)
    # the output is transposed and flipped so that it is displayed intuitively on the image plot
    
    print(vec.shape)
    print(heatmap_vectors.shape)
    
    vs = np.flip(np.tensordot(vec, heatmap_vectors, axes=([0], [2])).T, axis=0)
    #vs = np.tensordot(vec, heatmap_vectors, axes=([0], [2]))

    if cmap == 'diverging':
        cmap = sns.diverging_palette(150, 275, s=80, l=55, as_cmap=True)

    plt.imshow(vs, interpolation='none', extent=(xs[0], xs[-1], ys[0], ys[-1]), vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar()

    if name:
        plt.suptitle(name)

coord_sp = encode_point(3, -2, x_axis_sp, y_axis_sp).v

plt_heatmap(coord_sp, heatmap_vectors)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.savefig("ssp_2d.pdf", dpi=900, bbox_inches='tight')

plt.show()
