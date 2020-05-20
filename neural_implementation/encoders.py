from spatial_semantic_pointers.utils import encode_point, power
import nengo_spa as spa
import numpy as np

# 3 directions 120 degrees apart
vec_dirs = [0, 2 * np.pi / 3, 4 * np.pi / 3]


def to_ssp(v, X, Y):

    return encode_point(v[0], v[1], X, Y).v


def to_bound_ssp(v, item, X, Y):

    return (item * encode_point(v[0], v[1], X, Y)).v


def to_hex_region_ssp(v, X, Y, spacing=4):

    ret = np.zeros((len(X.v),))
    ret[:] = encode_point(v[0], v[1], X, Y).v
    for i in range(3):
        ret += encode_point(v[0] + spacing * np.cos(vec_dirs[i]), v[1] + spacing * np.sin(vec_dirs[i]), X, Y).v
        ret += encode_point(v[0] - spacing * np.cos(vec_dirs[i]), v[1] - spacing * np.sin(vec_dirs[i]), X, Y).v

    return ret


def to_band_region_ssp(v, angle, X, Y):

    ret = np.zeros((len(X.v),))
    ret[:] = encode_point(v[0], v[1], X, Y).v
    for dx in np.linspace(20./63., 20, 64):
        ret += encode_point(v[0] + dx * np.cos(angle), v[1] + dx * np.sin(angle), X, Y).v
        ret += encode_point(v[0] - dx * np.cos(angle), v[1] - dx * np.sin(angle), X, Y).v

    return ret


def grid_cell_encoder_old(dim, phases=(0, 0, 0), toroid_index=0):
    # vector from the center of all other rings, to the specific point on this set of rings
    n_toroids = (dim - 1)//2

    origin = np.zeros(dim, dtype='Complex64')
    origin[:] = 1

    phi_xs, phi_ys = get_sub_phi(phi=phases, angle=0, multi_phi=True)

    # modify the toroid of interest to point to the correct location
    for i in range(3):
        # origin[1+toroid_index*3+i] = phases[i]
        origin[1 + toroid_index * 3 + i] = phi_xs[i]*phi_ys[i]

    # set all conjugates
    origin[-1:dim // 2:-1] = np.conj(origin[1:(dim + 1) // 2])

    pole = np.zeros(dim, dtype='Complex64')
    pole[:] = -1
    pole[0] = 1

    # fix the nyquist frequency if required
    if dim % 2 == 0:
        pole[dim // 2] = 1

    # modify the toroid of interest to point to the correct location
    for i in range(3):
        # pole[1+toroid_index*3+i] = phases[i]
        pole[1 + toroid_index * 3 + i] = phi_xs[i] * phi_ys[i]

    # set all conjugates
    pole[-1:dim // 2:-1] = np.conj(pole[1:(dim + 1) // 2])

    # midpoint of the origin and the pole
    return (np.fft.ifft(origin).real + np.fft.ifft(pole).real) / 2.

    # # midpoint of the origin and the pole
    # # encoder = (origin + pole) / 2.
    # encoder = ((origin + pole) / 2.) #+ np.ones((dim,)) * 1./dim
    # # encoder = origin
    # # print('origin and pole')
    # # print(origin)
    # # print(pole)
    # #
    # # print('encoder')
    # # print(encoder)
    #
    # # encoder = np.zeros((dim, ))
    # # for n in range(n_toroids):
    # #     if n != toroid_index:
    # #         encoder +=
    #
    # return np.fft.ifft(encoder).real

def grid_cell_encoder_other_old(dim, phases=(0, 0, 0), toroid_index=0):
    # vector from the center of all other rings, to the specific point on this set of rings
    n_toroids = (dim - 1)//2

    encoder = np.zeros((dim,))

    n_offsets = 40
    offsets = np.linspace(0, 1, n_offsets+1)[:-1]

    for offset in offsets:

        origin = np.zeros(dim, dtype='Complex64')
        origin[:] = np.exp(1.j*2*np.pi*offset)
        origin[0] = 1

        phi_xs, phi_ys = get_sub_phi(phi=phases, angle=0, multi_phi=True)

        # modify the toroid of interest to point to the correct location
        for i in range(3):
            # origin[1+toroid_index*3+i] = phases[i]
            origin[1 + toroid_index * 3 + i] = phi_xs[i]*phi_ys[i]

        # set all conjugates
        origin[-1:dim // 2:-1] = np.conj(origin[1:(dim + 1) // 2])

        pole = np.zeros(dim, dtype='Complex64')
        pole[:] = np.exp(1.j * (2 * np.pi * offset + np.pi))
        pole[0] = 1

        # fix the nyquist frequency if required
        if dim % 2 == 0:
            pole[dim // 2] = 1

        # modify the toroid of interest to point to the correct location
        for i in range(3):
            # pole[1+toroid_index*3+i] = phases[i]
            pole[1 + toroid_index * 3 + i] = phi_xs[i] * phi_ys[i]

        # set all conjugates
        pole[-1:dim // 2:-1] = np.conj(pole[1:(dim + 1) // 2])

        # midpoint of the origin and the pole
        encoder += (np.fft.ifft(origin).real + np.fft.ifft(pole).real) / 2.

    encoder /= n_offsets

    return encoder


def grid_cell_encoder(dim, phi, angle, location=(0, 0), toroid_index=0):
    # vector from the center of all other rings, to the specific point on this set of rings
    n_toroids = (dim - 1)//2

    origin = np.zeros(dim, dtype='Complex64')
    origin[:] = 1

    phi_xs, phi_ys = get_sub_phi_for_loc(location=location, phi=phi, angle=angle)

    # modify the toroid of interest to point to the correct location
    for i in range(3):
        # origin[1+toroid_index*3+i] = phases[i]
        origin[1 + toroid_index * 3 + i] = phi_xs[i]*phi_ys[i]

    # set all conjugates
    origin[-1:dim // 2:-1] = np.conj(origin[1:(dim + 1) // 2])

    pole = np.zeros(dim, dtype='Complex64')
    pole[:] = -1
    pole[0] = 1

    # fix the nyquist frequency if required
    if dim % 2 == 0:
        pole[dim // 2] = 1

    # modify the toroid of interest to point to the correct location
    for i in range(3):
        # pole[1+toroid_index*3+i] = phases[i]
        pole[1 + toroid_index * 3 + i] = phi_xs[i] * phi_ys[i]

    # set all conjugates
    pole[-1:dim // 2:-1] = np.conj(pole[1:(dim + 1) // 2])

    # midpoint of the origin and the pole
    return (np.fft.ifft(origin).real + np.fft.ifft(pole).real) / 2.


def band_cell_encoder(dim, phase=0, toroid_index=0, band_index=0):
    # vector from the center of all other rings, to one specific ring on this set of rings

    origin = np.zeros(dim, dtype='complex64')
    origin[:] = 1

    phi_xs, phi_ys = get_sub_phi(phi=phase, angle=0)

    # modify the toroid of interest to point to the correct location
    origin[1 + toroid_index * 3 + band_index] = phi_xs[band_index]*phi_ys[band_index]#phase

    # set all conjugates
    origin[-1:dim // 2:-1] = np.conj(origin[1:(dim + 1) // 2])

    pole = np.zeros(dim, dtype='complex64')
    pole[:] = -1

    # fix the nyquist frequency if required
    if dim % 2 == 0:
        pole[dim // 2] = 1

    # modify the toroid of interest to point to the correct location
    pole[1 + toroid_index * 3 + band_index] = phi_xs[band_index]*phi_ys[band_index]#phase

    # set all conjugates
    pole[-1:dim // 2:-1] = np.conj(pole[1:(dim + 1) // 2])

    # midpoint of the origin and the pole
    encoder = (origin + pole) / 2.

    return np.fft.ifft(encoder).real


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


def get_sub_phi_for_loc(location=(0, 0), phi=2*np.pi, angle=0):
    X, Y, sv = orthogonal_hex_dir_7dim(phi=phi, angle=angle, multi_phi=False)

    xf = np.fft.fft(X.v)**location[0]
    yf = np.fft.fft(Y.v)**location[1]

    return xf[1:4], yf[1:4]
