import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as plticker
import numpy as np
import seaborn as sns

from nengo.dists import UniformHypersphere
import nengo.spa as spa
import nengo_spa
from spatial_semantic_pointers.utils import make_good_unitary
from scipy.linalg import circulant


def random_unitary(n_samples=1000, dim=3, version=1, eps=0.001):
    points = np.zeros((n_samples, dim))
    good = np.zeros((n_samples, ))

    for i in range(n_samples):
        if version == 1:
            sp = nengo_spa.SemanticPointer(data=np.random.randn(dim))
            sp = sp.normalized()
            sp = sp.unitary()
        elif version == 0:
            sp = spa.SemanticPointer(dim)
            sp.make_unitary()
        elif version == 2:
            sp = make_good_unitary(dim=dim)
        else:
            raise NotImplementedError

        points[i, :] = sp.v
        pf = np.fft.fft(points[i, :])
        if dim % 2 == 0:
            if np.abs(pf[0] - 1) < eps and  np.abs(pf[dim // 2] - 1) < eps:
                good[i] = 1
        else:
            if np.abs(pf[0] - 1) < eps:
                good[i] = 1
    return points, good


def random_sphere_points(n_samples=1000, dim=4):
    surface = np.random.randn(n_samples, dim)

    for i in range(n_samples):
        surface[i, :] /= np.linalg.norm(surface[i, :])

    return surface


def get_coord_rot_matrix(dim=4):
    new_axes = np.zeros((dim, dim))
    if dim == 4:
        # this is the unit displacement vector of the hypersphere centers

        # manually choosing a set that works
        new_axes[0, :] = 1./np.sqrt(dim)

        new_axes[1, 0] = new_axes[0, 1]
        new_axes[1, 1] = -new_axes[0, 0]
        new_axes[1, :] = new_axes[1, :] / np.linalg.norm(new_axes[1, :])

        new_axes[2, 2] = new_axes[0, 1]
        new_axes[2, 3] = -new_axes[0, 0]
        new_axes[2, :] = new_axes[2, :]/ np.linalg.norm(new_axes[2, :])

        new_axes[3, :2] = 1. / np.sqrt(dim)
        new_axes[3, 2:] = -1. / np.sqrt(dim)
        new_axes[3, :] = new_axes[3, :] / np.linalg.norm(new_axes[3, :])
    elif dim == 5:
        # this is the unit displacement vector of the hypersphere centers

        # manually choosing a set that works
        # [1  1  1  1  1] <- the required normal for projection
        # [1 -1  0  0  0]
        # [0  0  1 -1  0]
        # [1  1 -1 -1  0]
        # [1  1  1  1 -4]
        new_axes[0, :] = 1. / np.sqrt(dim)

        new_axes[1, 0] = new_axes[0, 1]
        new_axes[1, 1] = -new_axes[0, 0]
        new_axes[1, :] = new_axes[1, :] / np.linalg.norm(new_axes[1, :])

        new_axes[2, 2] = new_axes[0, 1]
        new_axes[2, 3] = -new_axes[0, 0]
        new_axes[2, :] = new_axes[2, :] / np.linalg.norm(new_axes[2, :])

        new_axes[3, :2] = 1. / np.sqrt(dim)
        new_axes[3, 2:4] = -1. / np.sqrt(dim)
        new_axes[3, :] = new_axes[3, :] / np.linalg.norm(new_axes[3, :])

        new_axes[4, :4] = -1.
        new_axes[4, 4] = 4.
        new_axes[4, :] = new_axes[4, :] / np.linalg.norm(new_axes[4, :])

        # check to make sure all axes are orthogonal
        for i in range(5):
            for j in range(i + 1, 5):
                if i != j:
                    assert np.allclose(np.dot(new_axes[i, :], new_axes[j, :]), 0)

    # want the determinant to be +1
    assert(np.allclose(np.linalg.det(new_axes), 1))

    return new_axes

fig = plt.figure(tight_layout=True, figsize=(16, 4))
ax = []
plot_limit = 1
dims = [4, 5]
for i, dim in enumerate(dims):
    for j in range(2):  # two views of each
        ax.append(fig.add_subplot(1, len(dims)*2, i*2+1+j, projection='3d'))
        ax[i*2+j].set_xlim([-plot_limit, plot_limit])
        ax[i*2+j].set_ylim([-plot_limit, plot_limit])
        ax[i*2+j].set_zlim([-plot_limit, plot_limit])

        # show the axes
        ax[i*2+j].plot([0, plot_limit], [0, 0], [0, 0], color='black')
        ax[i*2+j].plot([0, 0], [0, plot_limit], [0, 0], color='black')
        ax[i*2+j].plot([0, 0], [0, 0], [0, plot_limit], color='black')
        if j == 0:
            # ax[i*2+j].view_init(elev=32, azim=-42)
            ax[i * 2 + j].view_init(elev=45, azim=-59)
        else:
            ax[i*2+j].view_init(elev=19, azim=-69)

        loc = plticker.MultipleLocator(base=0.5)  # this locator puts ticks at regular intervals
        ax[i*2+j].xaxis.set_major_locator(loc)
        ax[i*2+j].yaxis.set_major_locator(loc)
        ax[i*2+j].zaxis.set_major_locator(loc)
        ax[i*2+j].set_title("D = {}".format(dim))

n_unitary = 2000
n_surface = 2000
for i, dim in enumerate(dims):
    coord_rot = get_coord_rot_matrix(dim=dim)
    if dim == 5:
        n_unitary = 10000
    unitary_points, good = random_unitary(n_samples=n_unitary, dim=dim, version=1)

    # unitary_points = unitary_points @ coord_rot

    #generated on unitary sphere
    surface = np.zeros((n_surface, dim))
    surface[:, 1:] = random_sphere_points(n_samples=n_surface, dim=dim - 1) * np.sqrt((dim - 1) / dim)
    surface = surface @ coord_rot
    surface2 = surface.copy()
    displacement = np.ones((dim,)) * 1. / dim
    surface += displacement
    surface2 -= displacement
    ax[i*2].scatter(surface[:, 0], surface[:, 1], surface[:, 2], color='green', alpha=0.075)
    ax[i*2].scatter(surface2[:, 0], surface2[:, 1], surface2[:, 2], color='green', alpha=0.075)
    ax[i*2+1].scatter(surface[:, 0], surface[:, 1], surface[:, 2], color='green', alpha=0.075)
    ax[i*2+1].scatter(surface2[:, 0], surface2[:, 1], surface2[:, 2], color='green', alpha=0.075)

    ind_good = good == 1
    ind_bad = good == 0
    ax[i*2].scatter(unitary_points[ind_good, 0], unitary_points[ind_good, 1], unitary_points[ind_good, 2], color='purple', alpha=0.075)
    ax[i*2].scatter(unitary_points[ind_bad, 0], unitary_points[ind_bad, 1], unitary_points[ind_bad, 2], color='red', alpha=0.075)
    ax[i * 2 + 1].scatter(unitary_points[ind_good, 0], unitary_points[ind_good, 1], unitary_points[ind_good, 2],color='purple', alpha=0.075)
    ax[i * 2 + 1].scatter(unitary_points[ind_bad, 0], unitary_points[ind_bad, 1], unitary_points[ind_bad, 2], color='red',alpha=0.075)

    sphere_points = random_sphere_points(n_samples=7500, dim=dim)
    ax[i*2].scatter(sphere_points[:, 0], sphere_points[:, 1], sphere_points[:, 2], color='blue', alpha=0.01)
    ax[i * 2 + 1].scatter(sphere_points[:, 0], sphere_points[:, 1], sphere_points[:, 2], color='blue', alpha=0.01)


plt.show()
