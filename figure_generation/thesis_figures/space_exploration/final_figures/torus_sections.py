import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as plticker
import seaborn as sns
import numpy as np
import nengo.spa as spa
import nengo_spa
from spatial_semantic_pointers.utils import make_good_unitary, power, encode_point_hex


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


def random_unitary(n_samples=1000, dim=3, version=1):
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


# style = 'surface'
style = 'lines'

fig = plt.figure(tight_layout=True, figsize=(16, 4))
ax = []
plot_limit = 1
if style == 'lines':
    loc = plticker.MultipleLocator(base=.5)  # this locator puts ticks at regular intervals
# elif style == 'surface':
#     loc = plticker.MultipleLocator(base=.2)  # this locator puts ticks at regular intervals
for i in range(4):
    ax.append(fig.add_subplot(1, 4, i+1, projection='3d'))

    if style == 'lines':
        ax[i].xaxis.set_major_locator(loc)
        ax[i].yaxis.set_major_locator(loc)
        ax[i].zaxis.set_major_locator(loc)
    elif style == 'surface':
        if i < 2:
            ax[i].set_xlim([.2, 1])
            ax[i].set_ylim([-.55, .25])
            ax[i].set_zlim([-.2, .6])
            ax[i].xaxis.set_major_locator(plticker.MultipleLocator(base=.25))
            ax[i].yaxis.set_major_locator(plticker.MultipleLocator(base=.25))
            ax[i].zaxis.set_major_locator(plticker.MultipleLocator(base=.25))
        else:
            ax[i].set_xlim([.2, 1])
            ax[i].set_ylim([-.4, .4])
            ax[i].set_zlim([-.4, .4])
            # ax[i].xaxis.set_major_locator(plticker.MultipleLocator(base=.1))
            ax[i].xaxis.set_major_locator(plticker.MultipleLocator(base=.25))
            ax[i].yaxis.set_major_locator(plticker.MultipleLocator(base=.25))
            ax[i].zaxis.set_major_locator(plticker.MultipleLocator(base=.25))

dim = 5

phi = np.pi / 2.
# phi = np.pi / 4.

X = orthogonal_unitary(dim=dim, phi=phi, index=1)
Y = orthogonal_unitary(dim=dim, phi=phi, index=2)

res = 32#64#32#16
limit = 0.5#8#0.5
xs = np.linspace(-limit, limit, res)
ys = np.linspace(-limit, limit, res)
zs = np.linspace(-limit, limit, res)

if style == 'surface':
    for n in range(4):
        ux = np.zeros((res*res, dim))

        if n < 2:
            xs = np.linspace(0, 1, res)
            ys = np.linspace(0, 1, res)
        else:
            xs = np.linspace(-0.5, 0.5, res)
            ys = np.linspace(-0.5, 0.5, res)

        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                ux[i*res+j] = np.fft.ifft((np.fft.fft(X) ** x) * (np.fft.fft(Y) ** y)).real

        # mark the initial axis in a different colour
        x_ring = np.zeros((res, dim))
        for i, x in enumerate(xs):
            x_ring[i, :] = np.fft.ifft(np.fft.fft(X) ** x).real
        y_ring = np.zeros((res, dim))
        for j, y in enumerate(ys):
            y_ring[j, :] = np.fft.ifft(np.fft.fft(Y) ** y).real

        ax[n].scatter(ux[:, 0], ux[:, 1], ux[:, 2], color='green', alpha=0.15)
        ax[n].scatter(x_ring[:, 0], x_ring[:, 1], x_ring[:, 2], color='blue', alpha=0.95)
        ax[n].scatter(y_ring[:, 0], y_ring[:, 1], y_ring[:, 2], color='red', alpha=0.95)

    # ax[0].view_init(elev=34, azim=-19)
    ax[0].view_init(elev=19, azim=-30)
    # ax[1].view_init(elev=12, azim=-130)
    ax[1].view_init(elev=9, azim=-139)
    ax[2].view_init(elev=169, azim=-4)
    # ax[2].view_init(elev=176, azim=-6)
    # ax[3].view_init(elev=-141, azim=-108)
    ax[3].view_init(elev=-157, azim=-82)
else:

    res = 128
    xs = np.linspace(0, 4, res)
    ys = np.linspace(0, 4, res)
    # mark the initial axis in a different colour
    x_ring = np.zeros((res, dim))
    for i, x in enumerate(xs):
        x_ring[i, :] = np.fft.ifft(np.fft.fft(X) ** x).real
    y_ring = np.zeros((res, dim))
    for j, y in enumerate(ys):
        y_ring[j, :] = np.fft.ifft(np.fft.fft(Y) ** y).real

    for n in range(4):
        ax[n].scatter(x_ring[:, 0], x_ring[:, 1], x_ring[:, 2], color='blue', alpha=0.75)
        ax[n].scatter(y_ring[:, 0], y_ring[:, 1], y_ring[:, 2], color='red', alpha=0.75)


    ax[0].view_init(elev=19, azim=-30)
    ax[1].view_init(elev=9, azim=-139)
    ax[2].view_init(elev=169, azim=-4)
    ax[3].view_init(elev=-157, azim=-82)

plt.show()
