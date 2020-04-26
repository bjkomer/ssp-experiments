import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.linalg import circulant
from spatial_semantic_pointers.utils import make_good_unitary
import nengo.spa as spa

# generate a pile of vectors
# create circulant matrices out of them
# only plot the points where the determinant is 1 or -1

def unitary_determinant_test(dim=2, n_samples=1000):
    for i in range(n_samples):
        if True:
            vec = make_good_unitary(dim=dim).v
            print(vec)
        else:
            sp = spa.SemanticPointer(dim)
            sp.make_unitary()
            vec = sp.v
        if not np.allclose(np.dot(circulant(vec)[0], circulant(vec)[1]), 0):
            print(np.dot(circulant(vec)[0], circulant(vec)[1]))
            print(np.linalg.det(circulant(vec)))
        # assert np.allclose(np.dot(circulant(vec)[0], circulant(vec)[1]), 0)
        assert np.allclose(np.abs(np.linalg.det(circulant(vec))), 1)
    return True


def unit_determinant_test(dim=3, n_samples=1000):
    vecs = np.random.randn(n_samples, dim)
    for i in range(n_samples):
        vecs[i, :] = vecs[i, :] / np.linalg.norm(vecs[i, :])
        print(vecs[i, :])
        print(np.linalg.det(circulant(vecs[i, :])))
        assert np.allclose(np.abs(np.linalg.det(circulant(vecs[i, :]))), 1)
    return True


def sum_to_one_3d(res):
    points = np.zeros((res**2, 3))
    for i, x in enumerate(np.linspace(0, 1, res)):
        for j, y in enumerate(np.linspace(0, 1 - x, res)):
            z = 1 - x - y
            points[i*res+j, :] = np.array([x, y, z])
    return points


def unitary_points(dim, n_samples, good_unitary=False):
    points = np.zeros((n_samples, dim))
    for i in range(n_samples):
        if good_unitary:
            points[i, :] = make_good_unitary(dim=dim).v
        else:
            sp = spa.SemanticPointer(dim)
            sp.make_unitary()
            points[i, :] = sp.v
    return points

# print(unit_determinant_test())

# print(unitary_determinant_test(dim=3))
# assert False

dim = 5
n_subspace = dim // 2
res = 32
n_samples = 50000

# print(np.linalg.det(circulant(np.array([1./dim, 1./dim, 1./dim]))))

limit = 1#1
xs = np.linspace(-limit, limit, res)
points = np.zeros((res**dim, dim))

points = np.random.uniform(-limit, limit, size=(n_samples, dim))
# points = unitary_points(dim, n_samples)
# points = sum_to_one_3d(32)
# n_samples = 32**2
dets = np.zeros((n_samples,))
f_norms_one = np.zeros((n_samples,))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
eps = 0.05
# eps = 0.0001

subspace_matches = np.zeros((n_samples, n_subspace))

for i in range(n_samples):
    mat = circulant(points[i, :])
    for j in range(n_subspace):
        subspace_matches[i, j] = np.dot(mat[0, :], mat[j+1, :])
    # dets[i] = np.linalg.det(mat)
    # if np.allclose(np.fft.fft(points[i, :]).real**2 + np.fft.fft(points[i, :]).imag**2, np.ones(dim), rtol=eps, atol=eps):
    #     f_norms_one[i] = 1
    # else:
    #     f_norms_one[i] = 0
    # if np.abs(np.abs(dets[i]) - 1) < eps:
    #     if dim == 2:
    #         ax.scatter(points[i, 0], points[i, 1], color='blue', alpha=0.3)
    #     else:
    #         ax.scatter(points[i, 0], points[i, 1], points[i, 2], color='blue', alpha=0.3)


inds = np.abs(np.abs(dets) - 1) < eps
# inds = f_norms_one > 0

colours = sns.color_palette("hls", dim)
for j in range(n_subspace):
    if j == 0:
        inds = np.abs(np.abs(subspace_matches[:, j])) < eps
        # ax.scatter(points[inds, 0], points[inds, 1], points[inds, 2], color=colours[j], alpha=0.1)
        ax.scatter(points[inds, 1], points[inds, 2], points[inds, 3], color=colours[j], alpha=0.1)

# print(points[inds, :])

# sto = sum_to_one_3d(16)
# ax.scatter(sto[:, 0], sto[:, 1], sto[:, 2], color='red', alpha=0.3)

# u = unitary_points(dim, 200)
# ax.scatter(u[:, 0], u[:, 1], u[:, 2], color='green', alpha=0.8)


plt.show()
