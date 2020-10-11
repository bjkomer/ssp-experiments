import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.linalg import circulant
from spatial_semantic_pointers.utils import make_good_unitary
import nengo.spa as spa

# rotate the coordinate system to each sub-toroid is orhogonally aligned


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


def center_of_space(dim):
    fv_origin = np.zeros(dim, dtype='complex64')
    fv_origin[:] = 1
    fv_origin[1:(dim + 1) // 2] = 1
    fv_origin[-1:dim // 2:-1] = np.conj(fv_origin[1:(dim + 1) // 2])

    fv_pole = np.zeros(dim, dtype='complex64')
    fv_pole[:] = 1
    fv_pole[1:(dim + 1) // 2] = -1
    fv_pole[-1:dim // 2:-1] = np.conj(fv_pole[1:(dim + 1) // 2])

    f_origin = np.fft.ifft(fv_origin).real
    f_pole = np.fft.ifft(fv_pole).real

    print((f_origin + f_pole)/2)

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

    all_vectors = np.zeros((n_indices*2, dim))

    print("Radii")
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

        all_vectors[index*2, :] = cross_vectors[index, :]
        all_vectors[index*2+1, :] = angle_vectors[index, :]

        print(np.linalg.norm(cross_vectors[index, :]))
        # print(np.dot(cross_vectors[index, :], angle_vectors[index, :]))
        # assert np.allclose(np.dot(cross_vectors[index, :], angle_vectors[index, :]), 0)
        assert np.abs(np.dot(cross_vectors[index, :], angle_vectors[index, :])) < 0.0000001

    # all_vectors = np.vstack([cross_vectors, angle_vectors])
    print("")
    print("Origin offsets")
    print(np.linalg.norm(origin_points, axis=1))
    print(origin_points)

    print(all_vectors.shape)

    print("opposite points")
    print(u_crosses)
    print("angle points")
    print(u_angles)

    for i in range(n_indices*2):
        for j in range(i+1, n_indices*2):
            # print(np.dot(all_vectors[i, :], all_vectors[j, :]))
            assert np.abs(np.dot(all_vectors[i, :], all_vectors[j, :])) < 0.0000001

    return all_vectors

    # for i in range(n_indices):
    #     for j in range(i+1, n_indices):
    #         print(np.dot(u_crosses[i, :], u_crosses[j, :]))
    #         assert np.abs(np.dot(u_crosses[i, :], u_crosses[j, :])) < 0.0000001
    #         print(np.dot(u_angles[i, :], u_angles[j, :]))
    #         assert np.abs(np.dot(u_angles[i, :], u_angles[j, :])) < 0.0000001

dim = 14#8#7
all_vectors = calc_circle_axes(dim=dim)

print(all_vectors.shape)

rot_dim = all_vectors.shape[0]


transform_mat = np.linalg.lstsq(np.eye(rot_dim), all_vectors)

print(transform_mat)

print(transform_mat[0])
print(transform_mat[0].shape)


# print(all_vectors.T @ transform_mat[0])
print("all vectors transformed into the space")
print(np.round(all_vectors @ transform_mat[0].T, 2))
print("")
print("one-hot vectors transformed into the space")
print(np.round(np.eye(dim) @ transform_mat[0].T, 2))
print("")
origin = np.zeros((dim,))
origin[0] = 1
print(np.round(origin @ transform_mat[0].T, 2))
rot_origin = origin @ transform_mat[0].T
origin_back = rot_origin @ np.linalg.pinv(transform_mat[0]).T
print(np.round(origin_back, 2))
diff = origin - origin_back
print(np.round(diff,2))
print("center of the space")
center_of_space(dim)


origin_offset = origin - diff
rot_origin_offset = origin_offset @ transform_mat[0].T
origin_back_offset = rot_origin @ np.linalg.pinv(transform_mat[0]).T

print("origin_offset", np.round(origin_offset, 2))
print("rot_origin_offset",  np.round(rot_origin_offset, 2))
print("origin_back_offset", np.round(origin_back_offset, 2))
