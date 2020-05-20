# checking whether the axes are a circle or an ellipse. My intuitions say ellipse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as plticker
import seaborn as sns
import numpy as np
import nengo.spa as spa
import nengo_spa
from spatial_semantic_pointers.utils import make_good_unitary, power, encode_point_hex, encode_point, get_fixed_dim_grid_axes


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


def orthogonal_hex_dir_7dim(phi=np.pi / 2., angle=0):
    dim = 7
    xf = np.zeros((dim,), dtype='Complex64')
    xf[0] = 1
    xf[1] = np.exp(1.j * phi)
    xf[2] = 1
    xf[3] = 1
    xf[4] = np.conj(xf[3])
    xf[5] = np.conj(xf[2])
    xf[6] = np.conj(xf[1])

    yf = np.zeros((dim,), dtype='Complex64')
    yf[0] = 1
    yf[1] = 1
    yf[2] = np.exp(1.j * phi)
    yf[3] = 1
    yf[4] = np.conj(yf[3])
    yf[5] = np.conj(yf[2])
    yf[6] = np.conj(yf[1])

    zf = np.zeros((dim,), dtype='Complex64')
    zf[0] = 1
    zf[1] = 1
    zf[2] = 1
    zf[3] = np.exp(1.j * phi)
    zf[4] = np.conj(zf[3])
    zf[5] = np.conj(zf[2])
    zf[6] = np.conj(zf[1])

    Xh = np.fft.ifft(xf).real
    Yh = np.fft.ifft(yf).real
    Zh = np.fft.ifft(zf).real

    # checks to make sure everything worked correctly
    assert np.allclose(np.abs(xf), 1)
    assert np.allclose(np.abs(yf), 1)
    assert np.allclose(np.fft.fft(Xh), xf)
    assert np.allclose(np.fft.fft(Yh), yf)
    assert np.allclose(np.linalg.norm(Xh), 1)
    assert np.allclose(np.linalg.norm(Yh), 1)

    axis_sps = [
        spa.SemanticPointer(data=Xh),
        spa.SemanticPointer(data=Yh),
        spa.SemanticPointer(data=Zh),
    ]

    n = 3
    points_nd = np.eye(n) * np.sqrt(n)
    # points in 2D that will correspond to each axis, plus one at zero
    points_2d = np.zeros((n, 2))
    thetas = np.linspace(0, 2 * np.pi, n + 1)[:-1] + angle
    # TODO: will want a scaling here, or along the high dim axes
    for i, theta in enumerate(thetas):
        points_2d[i, 0] = np.cos(theta)
        points_2d[i, 1] = np.sin(theta)

    transform_mat = np.linalg.lstsq(points_2d, points_nd)

    x_axis = transform_mat[0][0, :] / transform_mat[3][0]
    y_axis = transform_mat[0][1, :] / transform_mat[3][1]

    X = power(axis_sps[0], x_axis[0])
    Y = power(axis_sps[0], y_axis[0])
    for i in range(1, n):
        X *= power(axis_sps[i], x_axis[i])
        Y *= power(axis_sps[i], y_axis[i])

    sv = transform_mat[3][0]
    return X, Y, sv, transform_mat[0]


def random_unitary(n_samples=1000, dim=3, version=2):
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
        else:
            raise NotImplementedError

        points[i, :] = sp.v
    return points


dim = 512
rng = np.random.RandomState(seed=1411)
# X = make_good_unitary(dim=dim, rng=rng)
# Y = make_good_unitary(dim=dim, rng=rng)

X = spa.SemanticPointer(data=orthogonal_unitary(dim=dim, index=9, phi=np.pi/4.))
Y = spa.SemanticPointer(data=orthogonal_unitary(dim=dim, index=7, phi=np.pi/4.))
# Y = X
dim = 512
# X, Y = get_fixed_dim_grid_axes(dim=dim, seed=1151)
n_samples = 1000


mat = np.zeros((n_samples, dim))
pts = rng.uniform(-5, 5, size=(n_samples, 4))
# pts_a = rng.uniform(-5, -1, size=(n_samples, 2))
# pts_b = rng.uniform(1, 5, size=(n_samples, 2))
for k in range(n_samples):
    A = encode_point(pts[k, 0], pts[k, 1], X, Y)
    B = encode_point(pts[k, 2], pts[k, 3], X, Y)
    # A = encode_point(pts_a[k, 0], pts_a[k, 1], X, Y)
    # B = encode_point(pts_b[k, 0], pts_b[k, 1], X, Y)
    mat[k, :] = A.v - B.v

print("matrix rank", np.linalg.matrix_rank(mat))

