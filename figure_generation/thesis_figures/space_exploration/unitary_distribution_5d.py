import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns

from nengo.dists import UniformHypersphere
import nengo.spa as spa
import nengo_spa
from spatial_semantic_pointers.utils import make_good_unitary

version = 2
n_samples = 25000#10000
dim = 5#3

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

    points[i, :] = sp.v

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def orthogonal_dir_unitary(dim=5, phi=np.pi/2.):
    xf = np.zeros((dim,), dtype='Complex64')
    xf[0] = 1
    xf[1] = np.exp(1.j*phi)
    xf[2] = 1
    xf[3] = 1
    xf[4] = np.exp(-1.j*phi)

    yf = np.zeros((dim,), dtype='Complex64')
    yf[0] = 1
    yf[1] = 1
    yf[2] = np.exp(1.j * phi)
    yf[3] = np.exp(-1.j * phi)
    yf[4] = 1

    X = np.fft.ifft(xf).real
    Y = np.fft.ifft(yf).real

    # checks to make sure everything worked correctly
    assert np.allclose(np.abs(xf), 1)
    assert np.allclose(np.abs(yf), 1)
    assert np.allclose(np.fft.fft(X), xf)
    assert np.allclose(np.fft.fft(Y), yf)
    assert np.allclose(np.linalg.norm(X), 1)
    assert np.allclose(np.linalg.norm(Y), 1)

    return X, Y


def orthogonal_dir_unitary_angle(dim=5, phi=np.pi/2., angle=0):
    xf = np.zeros((dim,), dtype='Complex64')
    xf[0] = 1
    xf[1] = np.exp(1.j * phi * np.cos(angle))
    xf[2] = np.exp(1.j * phi * np.sin(angle))
    xf[3] = np.conj(xf[2])
    xf[4] = np.conj(xf[1])

    yf = np.zeros((dim,), dtype='Complex64')
    yf[0] = 1
    yf[1] = np.exp(1.j * phi * np.cos(angle + np.pi/2.))
    yf[2] = np.exp(1.j * phi * np.sin(angle + np.pi/2.))
    yf[3] = np.conj(yf[2])
    yf[4] = np.conj(yf[1])

    X = np.fft.ifft(xf).real
    Y = np.fft.ifft(yf).real

    # checks to make sure everything worked correctly
    assert np.allclose(np.abs(xf), 1)
    assert np.allclose(np.abs(yf), 1)
    assert np.allclose(np.fft.fft(X), xf)
    assert np.allclose(np.fft.fft(Y), yf)
    assert np.allclose(np.linalg.norm(X), 1)
    assert np.allclose(np.linalg.norm(Y), 1)

    return X, Y


freq = 2.
limit = freq/2.#*2
res = 32
xs = np.linspace(0, limit, res)
ys = np.linspace(0, limit, res)
phi = np.pi/freq
angle = np.pi*0.0#np.pi*.25#0.
# X, Y = orthogonal_dir_unitary(dim=5, phi=phi)
X, Y = orthogonal_dir_unitary_angle(dim=5, phi=phi, angle=angle)

surface_points = np.zeros(((res-2)*(res-2), dim))
for i, x in enumerate(xs[1:-1]):
    for j, y in enumerate(ys[1:-1]):
        surface_points[i*(res-2)+j, :] = np.fft.ifft(np.fft.fft(X)**x*np.fft.fft(Y)**y).real


# ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', alpha=0.01)
ax.scatter(surface_points[:, 0], surface_points[:, 1], surface_points[:, 2], color='green', alpha=0.5)

# mark the initial axis in a different colour
x_ring = np.zeros((res, dim))
for i, x in enumerate(xs):
    x_ring[i, :] = np.fft.ifft(np.fft.fft(X)**x).real
y_ring = np.zeros((res, dim))
for j, y in enumerate(ys):
    y_ring[j, :] = np.fft.ifft(np.fft.fft(Y)**y).real
ax.scatter(x_ring[:, 0], x_ring[:, 1], x_ring[:, 2], color='blue', alpha=0.75)
ax.scatter(y_ring[:, 0], y_ring[:, 1], y_ring[:, 2], color='red', alpha=0.75)

plt.show()
