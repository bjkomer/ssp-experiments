# visualization of the space of unitary vectors in 3D (hard to visualize more than that)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as plticker
import seaborn as sns
import numpy as np
import nengo.spa as spa
import nengo_spa
from spatial_semantic_pointers.utils import make_good_unitary


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


# def sum_to_one_3d(res):
#     points = np.zeros((res**2, 3))
#     for i, x in enumerate(np.linspace(0, 2, res)):
#         for j, y in enumerate(np.linspace(0, 2 - x, res)):
#             z = 1 - x - y
#             points[i*res+j, :] = np.array([x, y, z])
#     return points

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


def sum_to_one_3d(res, limit_low=-.5, limit_high=1.5, sign=1):
    points = np.zeros((res**2, 3))
    for i, x in enumerate(np.linspace(limit_low, limit_high, res)):
        for j, y in enumerate(np.linspace(limit_low, limit_high, res)):
            z = sign - x - y
            points[i*res+j, :] = np.array([x, y, z])
    return points


def cone_constraint_sample(n_samples=2500000, limit=2, dim=3, eps=0.001):
    points = np.random.uniform(-limit, limit, size=(n_samples, dim))
    ind = np.abs(points[:, 0]*points[:, 1] + points[:, 1]*points[:, 2] + points[:, 2]*points[:, 0]) < eps
    return points[ind]


def cone_constraint(res=32, limit_low=-1.5, limit_high=1.5, dim=3, eps=0.001):
    points = np.zeros((res**2, 3))
    for i, x in enumerate(np.linspace(limit_low, limit_high, res)):
        for j, y in enumerate(np.linspace(limit_low, limit_high, res)):
            # x*y + y*z + z*x = 0
            # z * (x + y) = -x*y
            # z = -x*y/(x+y)
            if np.abs(x + y) < eps:
                z = 0
            else:
                z = -x * y / (x + y)
            points[i * res + j, :] = np.array([x, y, z])
    return points


def random_sphere_points(n_samples=1000):
    xyz = np.random.randn(n_samples, 3)

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

fig = plt.figure(tight_layout=True, figsize=(16, 4))
ax = []
plot_limit = 1
for i in range(4):
    ax.append(fig.add_subplot(1, 4, i+1, projection='3d'))
    ax[i].set_xlim([-plot_limit, plot_limit])
    ax[i].set_ylim([-plot_limit, plot_limit])
    ax[i].set_zlim([-plot_limit, plot_limit])

    # show the axes
    ax[i].plot([0, plot_limit], [0, 0], [0, 0], color='black')
    ax[i].plot([0, 0], [0, plot_limit], [0, 0], color='black')
    ax[i].plot([0, 0], [0, 0], [0, plot_limit], color='black')
    ax[i].view_init(elev=32, azim=-42)

    loc = plticker.MultipleLocator(base=0.5)  # this locator puts ticks at regular intervals
    ax[i].xaxis.set_major_locator(loc)
    ax[i].yaxis.set_major_locator(loc)
    ax[i].zaxis.set_major_locator(loc)

res = 32#64
# xs = np.linspace(-5, 5, res)
xs = np.linspace(0, 2, res)

#signs = [1, -1]
signs = [1]


sphere_points = random_sphere_points(n_samples=5000)

ax[0].scatter(sphere_points[:, 0], sphere_points[:, 1], sphere_points[:, 2], color='blue', alpha=0.025)
ax[3].scatter(sphere_points[:, 0], sphere_points[:, 1], sphere_points[:, 2], color='blue', alpha=0.005)


sto_pos = sum_to_one_3d(res=32, limit_low=-.5, limit_high=1.5)
sto_neg = sum_to_one_3d(res=32, limit_low=-1.5, limit_high=0.5, sign=-1)
# ind = (sto_pos[:, 0] < plot_limit) & (sto_pos[:, 0] > -plot_limit) & (sto_pos[:, 1] < plot_limit) & (sto_pos[:, 1] > -plot_limit) & (sto_pos[:, 2] < plot_limit) & (sto_pos[:, 2] > -plot_limit)
ax[1].scatter(sto_pos[:, 0], sto_pos[:, 1], sto_pos[:, 2], color='green', alpha=0.1)
ax[1].scatter(sto_neg[:, 0], sto_neg[:, 1], sto_neg[:, 2], color='green', alpha=0.1)
ax[3].scatter(sto_pos[:, 0], sto_pos[:, 1], sto_pos[:, 2], color='green', alpha=0.025)
ax[3].scatter(sto_neg[:, 0], sto_neg[:, 1], sto_neg[:, 2], color='green', alpha=0.025)


# xyz = cone_constraint(n_samples=50000, eps=0.01)
# xyz = cone_constraint(res=64)
# ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color='red', alpha=0.1)
cone_points = cone_constraint_sample(n_samples=5000000)
ax[2].scatter(cone_points[:, 0], cone_points[:, 1], cone_points[:, 2], color='purple', alpha=0.1)
ax[3].scatter(cone_points[:, 0], cone_points[:, 1], cone_points[:, 2], color='purple', alpha=0.02)


unitary_points = random_unitary(n_samples=2000)
ax[3].scatter(unitary_points[:, 0], unitary_points[:, 1], unitary_points[:, 2], color='red', alpha=0.025)
#
# for i, phi in enumerate(phis):
#     for sign in signs:
#         u = unitary(phi, sign)
#         v = unitary(np.pi/3., sign)
#         for x in xs:
#             ux = np.fft.ifft(np.fft.fft(u) ** x).real
#             ax.scatter(ux[0], ux[1], ux[2], color=palette[i])
#             # optional imaginary part, only nonzero if sign is -1
#             # ux = np.fft.ifft(np.fft.fft(u) ** x).imag
#             # ax.scatter(ux[0], ux[1], ux[2], color=palette[2])
#             # print(np.linalg.norm(ux))
#         # ax.scatter(u[0], u[1], u[2], color=palette[i])



plt.show()
