# defining a transform that rotates the stored SSP
# rotation needs to be defined about a point (using origin for this, but any should be possible)
# exploring in 5D for simplicity
import numpy as np
import matplotlib.pyplot as plt
from spatial_semantic_pointers.utils import make_good_unitary


def unitary(phi_a, phi_b, sign=1):

    fv = np.zeros(5, dtype='complex64')
    fv[0] = sign
    fv[1] = np.exp(1.j*phi_a)
    fv[2] = np.exp(1.j*phi_b)
    fv[3] = np.conj(fv[2])
    fv[4] = np.conj(fv[1])

    assert np.allclose(np.abs(fv), 1)
    v = np.fft.ifft(fv)
    # assert np.allclose(v.imag, 0, atol=1e-5)
    v = v.real
    assert np.allclose(np.fft.fft(v), fv)
    assert np.allclose(np.linalg.norm(v), 1)
    return v


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

def get_rot_matrix(X, Y):
    dim = X.shape[0]
    # xs = np.linspace(-dim, dim, dim+1)
    res = (dim+1)*2
    xs = np.linspace(-dim, dim, res)
    # points along the curve in the original coordinate frame
    points_A = np.zeros((res**2, dim))
    # points along the curve in the new coordinate frame
    points_B = np.zeros((res**2, dim))


    origin = np.zeros((dim,))
    origin[0] = 1

    origin = np.ones((dim,))* 1./dim

    # for i, x in enumerate(xs):
    #     points_X[i] = np.fft.ifft(np.fft.fft(X) ** x).real - origin
    #     points_Y[i] = np.fft.ifft(np.fft.fft(Y) ** x).real - origin

    for i, x in enumerate(xs):
        for j, y in enumerate(xs):
            points_A[i*res+j] = np.fft.ifft(np.fft.fft(X) ** x * np.fft.fft(Y) ** y).real - origin
            points_B[i*res+j] = np.fft.ifft(np.fft.fft(X) ** (-y) * np.fft.fft(Y) ** x).real - origin

    # transformation from X to Y
    transform_mat = np.linalg.lstsq(points_A, points_B)

    # transform_mat = transform_mat[0] / transform_mat[3]

    print(transform_mat[0])
    print(transform_mat[3])

    return transform_mat[0] / transform_mat[3], origin

    # return transform_mat[0], origin

    # apply scaling to the axes based on the singular values. Both should be the same
    x_axis = transform_mat[0][0, :] / transform_mat[3][0]
    y_axis = transform_mat[0][1, :] / transform_mat[3][1]


def sim_2d(point, X, Y, xs, ys):
    sim = np.zeros((len(xs), len(ys)))

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            sim[i, j] = np.dot(point, np.fft.ifft(np.fft.fft(X) ** x * np.fft.fft(Y) ** y).real)

    return sim


# X = unitary(np.pi/2., 0)
# Y = unitary(0, np.pi/2.)

rng = np.random.RandomState(seed=13)
dim = 128
X = make_good_unitary(dim=dim, rng=rng).v
Y = make_good_unitary(dim=dim, rng=rng).v

# dim = 32
# X = orthogonal_unitary(dim=dim, index=dim//4, phi=np.pi/2.)
# Y = orthogonal_unitary(dim=dim, index=dim//4+1, phi=np.pi/2.)

pos = [.5, .25]
pos = [2, 1]

# representation of the point as an SSP
point = np.fft.ifft(np.fft.fft(X) ** pos[0] * np.fft.fft(Y) ** pos[1]).real

limit = 5
xs = np.linspace(-limit, limit, 64)
ys = np.linspace(-limit, limit, 64)

fig, ax = plt.subplots(1, 3)

rot_mat, origin = get_rot_matrix(X, Y)
print(rot_mat.shape)
print(point.shape)

# point_rot = rot_mat @ point
# point_rot = ((point - origin) @ rot_mat) + origin
point_rot = (rot_mat @ (point - origin)) + origin

ax[0].imshow(sim_2d(point, X, Y, xs, ys))
ax[1].imshow(sim_2d(point_rot, X, Y, xs, ys))
ax[2].imshow(sim_2d(point_rot, Y, X, xs, ys))





plt.show()
