# visualization of the space of unitary vectors in 3D (hard to visualize more than that)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
from matplotlib.colors import hsv_to_rgb


def unitary_5d(phi_a, phi_b):

    dim = 5
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


def random_sphere_points(n_samples=1000):
    xyz = np.random.uniform(-1, 1, size=(n_samples, 5))

    for i in range(n_samples):
        xyz[i, :] /= np.linalg.norm(xyz[i, :])

    # xyz = xyz / np.linalg.norm(xyz, axis=1) # doesnt work for some reason

    return xyz


fig = plt.figure()
dim = 5
ax = [
    fig.add_subplot(221, projection='3d'),
    fig.add_subplot(222, projection='3d'),
    fig.add_subplot(223, projection='3d'),
    fig.add_subplot(224, projection='3d'),
]
plot_limit = 1
for i in range(len(ax)):
    ax[i].set_xlim([-plot_limit, plot_limit])
    ax[i].set_ylim([-plot_limit, plot_limit])
    ax[i].set_zlim([-plot_limit, plot_limit])
    ax[i].plot([0, plot_limit], [0, 0], [0, 0], color='black')
    ax[i].plot([0, 0], [0, plot_limit], [0, 0], color='black')
    ax[i].plot([0, 0], [0, 0], [0, plot_limit], color='black')

res = 64#256#32#256#64

# xs = np.linspace(0, 4, res)
xs = np.linspace(0, 4, res)


palette = sns.color_palette("hls", 2)


# phi_as = [0, np.pi*.4999]
# phi_bs = [np.pi*.4999, 0]

phi_as = [np.pi*.4999, np.pi*.4999]
phi_bs = [np.pi*.4999*.5, -np.pi*.4999]


for k in range(2):
    u = unitary_5d(phi_as[k], phi_bs[k])
    for x in xs:
        ux = np.fft.ifft(np.fft.fft(u) ** x).real
        for i in range(len(ax)):
            ax[i].scatter(ux[(0+i)%dim], ux[(1+i)%dim], ux[(2+i)%dim], color=palette[k])
            if i == len(ax) - 1:
                print(ux)

plt.show()
