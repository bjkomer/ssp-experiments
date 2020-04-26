# Finding a set of 3 axis vectors that point in orthogonal directions along a 4D hypershere (3-sphere)
# Radon-Hurwitz numbers predict that a maximum of 3 orthogonal vector fields are possible to cover this shape
# therefore, if 3 orthogonal vectors are chosen, they are likely the correct ones? I don't know this for sure,
# but I know they will at least be some rotation of the correct ones
# treating these axes as hexagonal axes may be the best use of the space
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
from matplotlib.colors import hsv_to_rgb
from spatial_semantic_pointers.utils import encode_point, get_heatmap_vectors, make_good_unitary
from scipy.linalg import circulant


def orthogonal_axes(eps=0.1, vector_dist='epsilon'):
    dim = 5
    # get the orthogonal tangent axes to a 3-sphere (4D hypersphere)
    normal_vec = np.array([[1, 0, 0, 0]])

    # each other vector will have a component in one of the other 3 axis directions,
    # but also a smaller component in the normal direction

    if vector_dist == 'orthogonal':
        a = np.array([[0, 1, 0, 0]])
        b = np.array([[0, 0, 1, 0]])
        c = np.array([[0, 0, 0, 1]])
    elif vector_dist == 'midway':
        a = np.array([[1, 1, 0, 0]])
        b = np.array([[1, 0, 1, 0]])
        c = np.array([[1, 0, 0, 1]])
    elif vector_dist == 'epsilon':
        a = np.array([[1 - eps, eps, 0, 0]])
        b = np.array([[1 - eps, 0, eps, 0]])
        c = np.array([[1 - eps, 0, 0, eps]])
    else:
        raise NotImplementedError

    a = (a / np.linalg.norm(a)) * np.sqrt((dim-1)/dim)
    b = (b / np.linalg.norm(b)) * np.sqrt((dim-1)/dim)
    c = (c / np.linalg.norm(c)) * np.sqrt((dim-1)/dim)

    # a, b, and c are points along the hypersphere in orthogonal directions

    # need to transform them into the 5D space

    # need to offset them to have the correct center

    return a, b, c


def orthogonal_5d_axes(eps=0.1, vector_dist='epsilon'):
    # get the axes directly with no coordinate transform
    dim = 5
    # this is the normal that the lower dim hypersphere shift moves on, so it will be ignored
    # but also has to be orthogonal to everything
    proj_norm = np.ones((dim,))*1./dim
    # this is the normal that the surface coordinate system must be orthogonal to
    # can be calculated since the point [1, 0, 0, 0, 0] is the origin of the manifold
    # and the manifold center is known, it is the proj_norm
    unitary_surface_norm = np.array([1, 0, 0, 0, 0]) - proj_norm

    # check to make sure this is correct
    assert np.allclose(np.dot(proj_norm, unitary_surface_norm), 0)

    # now need to find any 3 vectors orthogonal to these first two
    # picking these manually to work
    x_dir = np.array([0,  1, -1,  0,  0])
    y_dir = np.array([0,  0,  0,  1, -1])
    z_dir = np.array([0,  1,  1, -1, -1])
    # normalize
    x_dir = x_dir / np.linalg.norm(x_dir)
    y_dir = y_dir / np.linalg.norm(y_dir)
    z_dir = z_dir / np.linalg.norm(z_dir)

    # now need to find points partially along the surface in these directions to use as axis vectors
    # the manifold origin to move out from is still [1, 0, 0, 0, 0]
    origin = np.array([1, 0, 0, 0, 0])
    X = origin + eps * x_dir
    Y = origin + eps * y_dir
    Z = origin + eps * z_dir


    # these axes are now slightly too big, need to normalize, but pulling in the right direction
    X -= proj_norm
    X = X / np.linalg.norm(X) * np.sqrt((dim-1)/dim)
    X += proj_norm

    Y -= proj_norm
    Y = Y / np.linalg.norm(Y) * np.sqrt((dim-1)/dim)
    Y += proj_norm

    Z -= proj_norm
    Z = Z / np.linalg.norm(Z) * np.sqrt((dim-1)/dim)
    Z += proj_norm

    return X, Y, Z


def get_coord_rot_matrix():
    dim = 5
    original_axes = np.eye(dim)
    new_axes = np.zeros((dim, dim))
    # this is the unit displacement vector of the hypersphere centers

    # manually choosing a set that works
    # [1  1  1  1  1] <- the required normal for projection
    # [1 -1  0  0  0]
    # [0  0  1 -1  0]
    # [1  1 -1 -1  0]
    # [1  1  1  1 -4]
    new_axes[0, :] = 1./np.sqrt(dim)

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
        for j in range(i+1, 5):
            if i != j:
                assert np.allclose(np.dot(new_axes[i, :], new_axes[j, :]), 0)

    print(np.linalg.det(new_axes))
    print("")
    # want the determinant to be +1
    # assert(np.allclose(np.linalg.det(new_axes), 1))

    return new_axes

dim = 5
if True:
    X, Y, Z = orthogonal_5d_axes(eps=1)

else:
    # old way, not as nice
    X = np.zeros((dim,))
    Y = np.zeros((dim,))
    Z = np.zeros((dim,))

    # TODO: this needs to be the correct rotation matrix
    rot_mat = get_coord_rot_matrix()

    a, b, c = orthogonal_axes(eps=0.1, vector_dist='epsilon')
    # leaving the first element as 0, that is the one being rotated away with the coordinate transform
    # is this correct??
    X[1:] = a
    Y[1:] = b
    Z[1:] = c
    # X[:-1] = a
    # Y[:-1] = b
    # Z[:-1] = c
    # print(a)
    # print(X)

    print(np.linalg.norm(X))
    print(np.linalg.norm(Y))
    print(np.linalg.norm(Z))
    print("")

    # rotate and translate
    X = (X @ rot_mat) + np.ones((dim,))*1./dim
    Y = (Y @ rot_mat) + np.ones((dim,))*1./dim
    Z = (Z @ rot_mat) + np.ones((dim,))*1./dim

# a test of the properties
# X = make_good_unitary(dim=5).v

print("norms")
print(np.linalg.norm(X))
print(np.linalg.norm(Y))
print(np.linalg.norm(Z))
print("")
print("sums")
print(np.sum(X))
print(np.sum(Y))
print(np.sum(Z))
print("")
print("circulant det")
print(np.linalg.det(circulant(X)))
print(np.linalg.det(circulant(Y)))
print(np.linalg.det(circulant(Z)))
print("")
print("fft norms")
print(np.fft.fft(X).real**2 + np.fft.fft(X).imag**2)
print(np.fft.fft(Y).real**2 + np.fft.fft(Y).imag**2)
print(np.fft.fft(Z).real**2 + np.fft.fft(Z).imag**2)

assert np.allclose(np.linalg.norm(X), 1)
assert np.allclose(np.linalg.norm(Y), 1)
assert np.allclose(np.linalg.norm(Z), 1)


print(np.fft.fft(X))
print(np.fft.fft(Y))
print(np.fft.fft(Z))
