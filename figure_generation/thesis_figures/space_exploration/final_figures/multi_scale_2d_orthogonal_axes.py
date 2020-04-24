import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as plticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import numpy as np
import nengo.spa as spa
import nengo_spa
from spatial_semantic_pointers.utils import make_good_unitary, power, get_heatmap_vectors
from matplotlib.gridspec import GridSpec


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

def unitary_5d(dim, phi_a, phi_b):

    fv = np.zeros(dim, dtype='complex64')
    fv[:] = 1
    fv[1] = np.exp(1.j * phi_a)
    fv[2] = np.exp(1.j * phi_b)
    fv[3] = np.conj(fv[2])
    fv[4] = np.conj(fv[1])

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
    return X, Y, sv



limit = 10
res = 256
xs = np.linspace(-limit, limit, res)
ys = np.linspace(-limit, limit, res)

dim = 5

gs = GridSpec(1, 41, left=0.02, right=0.96)#, left=0.05, right=0.48, wspace=0.05)
fig = plt.figure(figsize=(41/3., 10/3.))#, tight_layout=True)
# fig, ax = plt.subplots(1, 4, tight_layout=True, figsize=(16, 4))
ax = [
    fig.add_subplot(gs[:10]),
    fig.add_subplot(gs[10:20]),
    fig.add_subplot(gs[20:30]),
    fig.add_subplot(gs[30:40]),
    fig.add_subplot(gs[40:41]),
]
fontsize = 18


if dim == 5:

    phi_axs = [np.pi / 2., np.pi / 4., np.pi / 4., np.pi / 4.]
    phi_bxs = [        0.,         0., np.pi / 4., np.pi / 2.]
    phi_ays = [        0.,         0.,-np.pi / 4., -np.pi / 6]
    phi_bys = [np.pi / 2., np.pi / 4., np.pi / 4., np.pi / 4]
    titles = [
        'X = [\u03C0/2, 0], Y = [0, \u03C0/2]',
        'X = [\u03C0/4, 0], Y = [0, \u03C0/4]',
        'X = [\u03C0/4, \u03C0/4], Y = [-\u03C0/4, \u03C0/4]',
        'X = [\u03C0/4, \u03C0/2], Y = [-\u03C0/6, \u03C0/4]',
    ]
    fontsize = 14
    angles = [0, 0, np.pi/4., np.pi/6.]
    loc = plticker.MultipleLocator(base=5)  # this locator puts ticks at regular intervals
    for i in range(4):
        X = unitary_5d(dim=dim, phi_a=phi_axs[i], phi_b=phi_bxs[i])
        Y = unitary_5d(dim=dim, phi_a=phi_ays[i], phi_b=phi_bys[i])
        hmv = get_heatmap_vectors(xs, ys, X, Y)
        im = ax[i].imshow(hmv[:, :, 0], origin='lower', interpolation='none', extent=(xs[0], xs[-1], ys[0], ys[-1]), vmin=None, vmax=1)
        ax[i].set_title(titles[i], fontsize=fontsize)
        ax[i].xaxis.set_major_locator(loc)
        ax[i].yaxis.set_major_locator(loc)

    fig.colorbar(im, cax=ax[-1])
elif dim == 7:
    titles = [
        '\u03C6 = \u03C0/2, \u03B8 = 0',
        '\u03C6 = \u03C0/4, \u03B8 = 0',
        '\u03C6 = \u03C0/2, \u03B8 = \u03C0/4',
        '\u03C6 = 3\u03C0/4, \u03B8 = \u03C0/6',
    ]
    phis = [np.pi/2., np.pi/4., np.pi/2, 3*np.pi/4.]
    angles = [0, 0, np.pi/4., np.pi/6.]
    loc = plticker.MultipleLocator(base=5)  # this locator puts ticks at regular intervals
    for i in range(4):
        X, Y, sv = orthogonal_hex_dir_7dim(phi=phis[i], angle=angles[i])
        hmv = get_heatmap_vectors(xs*sv, ys*sv, X, Y)
        im = ax[i].imshow(hmv[:, :, 0], origin='lower', interpolation='none', extent=(xs[0], xs[-1], ys[0], ys[-1]), vmin=None, vmax=1)
        ax[i].set_title(titles[i], fontsize=fontsize)
        ax[i].xaxis.set_major_locator(loc)
        ax[i].yaxis.set_major_locator(loc)

    fig.colorbar(im, cax=ax[-1])

plt.show()
