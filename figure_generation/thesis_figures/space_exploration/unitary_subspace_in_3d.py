# visualization of the space of unitary vectors in 3D (hard to visualize more than that)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np


def unitary(phi, sign=1):

    fv = np.zeros(3, dtype='complex64')
    fv[0] = sign
    fv[1] = np.exp(1.j*phi)
    fv[2] = np.conj(fv[1])

    assert np.allclose(np.abs(fv), 1)
    v = np.fft.ifft(fv)
    # assert np.allclose(v.imag, 0, atol=1e-5)
    v = v.real
    assert np.allclose(np.fft.fft(v), fv)
    assert np.allclose(np.linalg.norm(v), 1)
    return v


def random_sphere_points(n_samples=1000):
    xyz = np.random.uniform(-1, 1, size=(n_samples, 3))

    for i in range(n_samples):
        xyz[i, :] /= np.linalg.norm(xyz[i, :])

    # xyz = xyz / np.linalg.norm(xyz, axis=1) # doesnt work for some reason

    return xyz


n_samples = 6
eps = 0.001
phis = np.linspace(0+eps, np.pi-eps, n_samples)

# phis = [np.pi/12.]
phis = [np.pi/2.]

palette = sns.color_palette("hls", n_samples)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

res = 32#64
# xs = np.linspace(-5, 5, res)
xs = np.linspace(0, 2, res)

#signs = [1, -1]
signs = [1]

if True:
    xyz = random_sphere_points(n_samples=1000)
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color='blue', alpha=0.01)

if True:
    ax.plot([0, 1], [0, 0], [0, 0], color='green')
    ax.plot([0, 0], [0, 1], [0, 0], color='green')
    ax.plot([0, 0], [0, 0], [0, 1], color='green')

for i, phi in enumerate(phis):
    for sign in signs:
        u = unitary(phi, sign)
        v = unitary(np.pi/3., sign)
        for x in xs:
            for y in xs:
                ux = np.fft.ifft(np.fft.fft(u) ** x * np.fft.fft(v) ** y).real
                ax.scatter(ux[0], ux[1], ux[2], color=palette[i])
        # for x in xs:
        #     ux = np.fft.ifft(np.fft.fft(u) ** x).real
        #     ax.scatter(ux[0], ux[1], ux[2], color=palette[i])
        #     # optional imaginary part, only nonzero if sign is -1
        #     # ux = np.fft.ifft(np.fft.fft(u) ** x).imag
        #     # ax.scatter(ux[0], ux[1], ux[2], color=palette[2])
        #     # print(np.linalg.norm(ux))
        # # ax.scatter(u[0], u[1], u[2], color=palette[i])



plt.show()
