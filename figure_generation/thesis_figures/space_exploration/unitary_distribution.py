import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns

from nengo.dists import UniformHypersphere
import nengo.spa as spa
import nengo_spa
from spatial_semantic_pointers.utils import make_good_unitary
from scipy.linalg import circulant


def random_sphere_points(n_samples=1000, dim=4):
    surface = np.random.randn(n_samples, dim)

    for i in range(n_samples):
        surface[i, :] /= np.linalg.norm(surface[i, :])

    return surface


def get_coord_rot_matrix(dim=4):
    original_axes = np.eye(dim)
    new_axes = np.zeros((dim, dim))
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

    print(np.linalg.det(new_axes))

    return new_axes

coord_rot = get_coord_rot_matrix(dim=4)

version = 1


n_samples = 25000#10000
dim = 4#3

points = np.zeros((n_samples, dim))

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

    points[i, :] = sp.v

    # print(np.linalg.det(circulant(points[i, :])))
    # print(np.sum(points[i, :]))
    # print(np.fft.fft(points[i, :])[[0, 2]])
    # print("")
    # assert np.allclose(np.sum(points[i, :]), np.linalg.det(circulant(points[i, :])))
# print(np.mean(np.sum(points, axis=1)))
# assert False

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# points = points @ coord_rot

# ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', alpha=0.01)
ax.scatter(points[:, 1], points[:, 2], points[:, 3], color='blue', alpha=0.01)

if False:
    surface = random_sphere_points(n_samples=1000, dim=4)
    surface = surface @ coord_rot
    ax.scatter(surface[:, 0], surface[:, 1], surface[:, 2], color='green', alpha=0.1)

if True:
    #generated on unitary sphere
    surface = np.zeros((1000, 4))
    surface[:, 1:] = random_sphere_points(n_samples=1000, dim=3) * np.sqrt((dim-1)/dim)
    surface = surface @ coord_rot
    surface2 = surface.copy()
    surface += np.array([[1/4., 1/4., 1/4., 1/4.]])
    surface2 -= np.array([[1 / 4., 1 / 4., 1 / 4., 1 / 4.]])
    # ax.scatter(surface[:, 0], surface[:, 1], surface[:, 2], color='green', alpha=0.1)
    # ax.scatter(surface2[:, 0], surface2[:, 1], surface2[:, 2], color='green', alpha=0.1)
    ax.scatter(surface[:, 1], surface[:, 2], surface[:, 3], color='green', alpha=0.1)
    ax.scatter(surface2[:, 1], surface2[:, 2], surface2[:, 3], color='green', alpha=0.1)

    print(np.linalg.norm(surface2, axis=1))
    print(np.sum(surface2, axis=1))

    fft_surface = np.fft.fft(surface, axis=1)

    # all_fft_norms = np.linalg.norm(fft_surface, axis=1)
    all_fft_norms = np.sqrt(fft_surface.imag ** 2 + fft_surface.real ** 2)
    # print(all_fft_norms.shape)
    print(all_fft_norms)

    # print(fft_surface)

    # print(fft_surface[14,:])
    #
    # fft_norms = np.sqrt(fft_surface[14,:].imag ** 2 + fft_surface[14,:].real ** 2)
    # print(fft_norms)
    # fft_unit = fft_surface[14,:] / fft_norms
    # vec = np.fft.ifft(fft_unit)
    # print("back and forth:", vec)
    # print("original", surface[14,:])

    # for i in range(1000):
    #     surface[i, :]

    # create a vector based on one of these non-unitary on the space, and see what happens if it is an axis vector
    X = surface[14,:]

    xs = np.linspace(-1, 1, 64)
    traj = np.zeros((64, 4))
    for i, x in enumerate(xs):
        traj[i, :] = np.fft.ifft(np.fft.fft(X)**x).real

    ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], color='red', alpha=0.9)

plt.show()
