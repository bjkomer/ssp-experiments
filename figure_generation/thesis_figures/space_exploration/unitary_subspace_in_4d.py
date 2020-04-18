# visualization of the space of unitary vectors in 3D (hard to visualize more than that)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
from matplotlib.colors import hsv_to_rgb


def unitary(phi_a, phi_b, sign=1):

    fv = np.zeros(4, dtype='complex64')
    fv[0] = sign
    fv[1] = np.exp(1.j*phi_a)
    fv[2] = 1
    fv[3] = np.conj(fv[1])

    assert np.allclose(np.abs(fv), 1)
    v = np.fft.ifft(fv)
    # assert np.allclose(v.imag, 0, atol=1e-5)
    v = v.real
    assert np.allclose(np.fft.fft(v), fv)
    assert np.allclose(np.linalg.norm(v), 1)
    return v


def random_sphere_points(n_samples=1000):
    surface = np.random.uniform(-1, 1, size=(n_samples, 4))

    for i in range(n_samples):
        surface[i, :] /= np.linalg.norm(surface[i, :])

    # xyz = xyz / np.linalg.norm(xyz, axis=1) # doesnt work for some reason

    return surface

def unitary_from_sphere(n_samples=1000, dim=4):
    surface = np.random.uniform(-1, 1, size=(n_samples, dim-1))

    for i in range(n_samples):
        # normalize, and then scale to the correct radius
        surface[i, :] /= np.linalg.norm(surface[i, :]) * np.sqrt((dim-1)/dim)

    # transform into the full space


    # offset by the center point

    return surface



n_samples = 6
eps = 0.001
phis = np.linspace(0+eps, np.pi-eps, n_samples)

# phis = [np.pi/12.]
# phis = [np.pi/2.]

palette = sns.color_palette("hls", n_samples)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

res = 256#32#256#64
# xs = np.linspace(-5, 5, res)
xs = np.linspace(0, 24, res)
xs = np.linspace(0, 24, res)

#signs = [1, -1]
signs = [1]

phis = [np.pi/3., np.pi/4.]

palette = sns.color_palette("hls", len(phis)**2)

if True:
    surface = random_sphere_points(n_samples=1000)
    ax.scatter(surface[:, 0], surface[:, 1], surface[:, 2], color='blue', alpha=0.1)
#
# if True:
#     ax.plot([0, 1], [0, 0], [0, 0], color='green')
#     ax.plot([0, 0], [0, 1], [0, 0], color='green')
#     ax.plot([0, 0], [0, 0], [0, 1], color='green')

ssp_manifold = 1
# ssp_manifold = 2

if ssp_manifold == 1:
    for i, phi_a in enumerate(phis):
        for j, phi_b in enumerate(phis):
    # for i, phi_a in enumerate([np.pi/2.]):
    #     for j, phi_b in enumerate([np.pi/3.]):
            for sign in signs:
                u = unitary(phi_a, phi_b, sign)
                # v = unitary(np.pi/3., sign)
                # for x in xs:
                #     for y in xs:
                #         ux = np.fft.ifft(np.fft.fft(u) ** x * np.fft.fft(v) ** y).real
                #         ax.scatter(ux[0], ux[1], ux[2], color=palette[i])
                for x in xs:
                    ux = np.fft.ifft(np.fft.fft(u) ** x).real
                    ax.scatter(ux[0], ux[1], ux[2], color=palette[i*len(phis) + j])
                    # print(np.linalg.norm(ux))
                #     # optional imaginary part, only nonzero if sign is -1
                #     # ux = np.fft.ifft(np.fft.fft(u) ** x).imag
                #     # ax.scatter(ux[0], ux[1], ux[2], color=palette[2])
                #     # print(np.linalg.norm(ux))
                # ax.scatter(u[0], u[1], u[2], color=palette[i])
else:
    palette = sns.color_palette("hls", len(xs))
    u = unitary(np.pi/3., np.pi/4., 1)
    v = unitary(np.pi/4., np.pi/3., 1)
    for i, x in enumerate(xs):
        for j, y in enumerate(xs):
            ux = np.fft.ifft(np.fft.fft(u) ** x * np.fft.fft(v) ** y).real
            # ax.scatter(ux[0], ux[1], ux[2], color=palette[i])
            ax.scatter(ux[0], ux[1], ux[2], color=hsv_to_rgb([x/xs[-1], 1, 1]))

plt.show()
