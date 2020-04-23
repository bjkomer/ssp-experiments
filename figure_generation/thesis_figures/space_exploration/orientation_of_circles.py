import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.linalg import circulant
from spatial_semantic_pointers.utils import make_good_unitary
import nengo.spa as spa


def orthogonal_unitary(dim, index, phi):

    fv = np.zeros(dim, dtype='complex64')
    fv[:] = 1
    fv[index] = np.exp(1.j*phi)
    fv[-index] = np.conj(fv[index])

    assert np.allclose(np.abs(fv), 1)
    v = np.fft.ifft(fv)
    v = v.real
    assert np.allclose(np.fft.fft(v), fv)
    assert np.allclose(np.linalg.norm(v), 1)
    return v


def calc_circle_axes(dim):

    n_indices = (dim-1)//2

    # origin of the circle
    origin_points = np.zeros((n_indices, dim))
    # vector from origin of circle to origin of arc
    cross_vectors = np.zeros((n_indices, dim))
    # vector from origin of circle to 1/4 around the arc
    angle_vectors = np.zeros((n_indices, dim))

    u_crosses = np.zeros((n_indices, dim))
    u_angles = np.zeros((n_indices, dim))

    # the starting point on the circle is the same in all cases
    arc_origin = np.zeros((dim, ))
    arc_origin[0] = 1
    for index in range(n_indices):
        u_cross = orthogonal_unitary(dim, index + 1, np.pi)
        u_angle = orthogonal_unitary(dim, index + 1, np.pi/2.)

        # also keeping track of the axis vectors themselves, to see if they are all orthogonal
        u_crosses[index, :] = u_cross
        u_angles[index, :] = u_angle

        # midpoint of opposite ends of the circle
        origin_points[index, :] = (arc_origin + u_cross) / 2.
        cross_vectors[index, :] = arc_origin - origin_points[index, :]
        angle_vectors[index, :] = u_angle - origin_points[index, :]

        print(np.linalg.norm(cross_vectors[index, :]))
        # print(np.dot(cross_vectors[index, :], angle_vectors[index, :]))
        # assert np.allclose(np.dot(cross_vectors[index, :], angle_vectors[index, :]), 0)
        assert np.abs(np.dot(cross_vectors[index, :], angle_vectors[index, :])) < 0.0000001

    all_vectors = np.vstack([cross_vectors, angle_vectors])

    print(np.linalg.norm(origin_points, axis=1))
    print(origin_points)

    print(all_vectors.shape)

    for i in range(n_indices*2):
        for j in range(i+1, n_indices*2):
            # print(np.dot(all_vectors[i, :], all_vectors[j, :]))
            assert np.abs(np.dot(all_vectors[i, :], all_vectors[j, :])) < 0.0000001

    # for i in range(n_indices):
    #     for j in range(i+1, n_indices):
    #         print(np.dot(u_crosses[i, :], u_crosses[j, :]))
    #         assert np.abs(np.dot(u_crosses[i, :], u_crosses[j, :])) < 0.0000001
    #         print(np.dot(u_angles[i, :], u_angles[j, :]))
    #         assert np.abs(np.dot(u_angles[i, :], u_angles[j, :])) < 0.0000001

dim = 5
calc_circle_axes(dim=dim)
