import numpy as np
import nengo_spa as spa
from spatial_semantic_pointers.utils import encode_point, power


def orthogonal_hex_dir_7dim(phi=np.pi / 2., angle=0, multi_phi=False):
    if multi_phi:
        phi_x = phi[0]
        phi_y = phi[1]
        phi_z = phi[2]
    else:
        phi_x = phi
        phi_y = phi
        phi_z = phi

    dim = 7
    xf = np.zeros((dim,), dtype='Complex64')
    xf[0] = 1
    xf[1] = np.exp(1.j * phi_x)
    xf[2] = 1
    xf[3] = 1
    xf[4] = np.conj(xf[3])
    xf[5] = np.conj(xf[2])
    xf[6] = np.conj(xf[1])

    yf = np.zeros((dim,), dtype='Complex64')
    yf[0] = 1
    yf[1] = 1
    yf[2] = np.exp(1.j * phi_y)
    yf[3] = 1
    yf[4] = np.conj(yf[3])
    yf[5] = np.conj(yf[2])
    yf[6] = np.conj(yf[1])

    zf = np.zeros((dim,), dtype='Complex64')
    zf[0] = 1
    zf[1] = 1
    zf[2] = 1
    zf[3] = np.exp(1.j * phi_z)
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
    return X, Y, sv


def orthogonal_hex_dir(phis=(np.pi / 2., np.pi/10.), angles=(0, np.pi/3.)):
    n_scales = len(phis)
    dim = 6*n_scales + 1

    xf = np.zeros((dim,), dtype='Complex64')
    xf[0] = 1

    yf = np.zeros((dim,), dtype='Complex64')
    yf[0] = 1

    for i in range(n_scales):
        phi_xs, phi_ys = get_sub_phi(phis[i], angles[i])
        xf[1 + i * 3:1 + (i + 1) * 3] = phi_xs
        yf[1 + i * 3:1 + (i + 1) * 3] = phi_ys

    xf[-1:dim // 2:-1] = np.conj(xf[1:(dim + 1) // 2])
    yf[-1:dim // 2:-1] = np.conj(yf[1:(dim + 1) // 2])

    X = np.fft.ifft(xf).real
    Y = np.fft.ifft(yf).real

    return spa.SemanticPointer(data=X), spa.SemanticPointer(data=Y)


def get_sub_phi(phi, angle, multi_phi=False):
    X, Y, sv = orthogonal_hex_dir_7dim(phi=phi, angle=angle, multi_phi=multi_phi)

    xf = np.fft.fft(X.v)
    yf = np.fft.fft(Y.v)

    # xf = np.fft.fft(X.v)**(1./sv)
    # yf = np.fft.fft(Y.v)**(1./sv)

    # xf = np.fft.fft(X.v)**(sv)
    # yf = np.fft.fft(Y.v)**(sv)

    return xf[1:4], yf[1:4]


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


def unitary_from_phi(phis):
    # phis is a numpy array of phi values

    dim = len(phis)*2+1

    fv = np.zeros(dim, dtype='complex64')
    fv[0] = 1
    fv[1:(dim + 1) // 2] = np.exp(1.j*phis)  # np.cos(phis) + 1j * np.sin(phis)
    fv[-1:dim // 2:-1] = np.conj(fv[1:(dim + 1) // 2])

    assert np.allclose(np.abs(fv), 1)
    v = np.fft.ifft(fv)
    v = v.real
    assert np.allclose(np.fft.fft(v), fv)
    assert np.allclose(np.linalg.norm(v), 1)
    return v
