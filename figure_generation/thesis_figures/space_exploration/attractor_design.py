# experimenting with making an attractor cleanup memory on the whole space
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as plticker
import seaborn as sns
import numpy as np
import nengo.spa as spa
import nengo_spa
from spatial_semantic_pointers.utils import make_good_unitary

dim = 4

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


# unitaries = [
#     unitary(np.pi/3., +1),
#     unitary(-np.pi/8., +1),
#     unitary(np.pi/3., -1),
#     unitary(np.pi/8., -1),
# ]

# titles = [
#     ''
# ]
# palette = sns.color_palette("hls", len(unitaries))

# for shading the space
unitary_points = random_unitary(n_samples=2000, dim=dim)

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

    if i == 2:
        ax[i].view_init(elev=21, azim=-77)
    else:
        ax[i].view_init(elev=33, azim=16)

    loc = plticker.MultipleLocator(base=0.5)  # this locator puts ticks at regular intervals
    ax[i].xaxis.set_major_locator(loc)
    ax[i].yaxis.set_major_locator(loc)
    ax[i].zaxis.set_major_locator(loc)

    ax[i].scatter(unitary_points[:, 0], unitary_points[:, 1], unitary_points[:, 2], color='grey', alpha=0.015)

res = 64
xs = np.linspace(0, 4, res)

starts = np.array([
    1,
    2,
    3,
    4,
])

ends = np.array([
    2,
    4,
    6,
    8,
])

mids = (starts + ends)/2

u = unitary(np.pi/3., +1)
rng = np.random.RandomState(seed=13)
u = make_good_unitary(dim=dim, rng=rng).v
for i in range(4):

    u_start = np.fft.ifft(np.fft.fft(u) ** starts[i]).real
    u_end = np.fft.ifft(np.fft.fft(u) ** ends[i]).real
    u_mid = np.fft.ifft(np.fft.fft(u) ** mids[i]).real
    # get the middle point in the representation space, and then simply divide by its frequency space norm
    calc_mid = (u_start + u_end)/2
    calc_mid_f = np.fft.fft(calc_mid)
    calc_mid_f = calc_mid_f / (calc_mid_f.real**2 + calc_mid_f.imag**2)
    calc_mid = np.fft.ifft(calc_mid_f).real
    # calc_mid = calc_mid / np.linalg.norm(calc_mid)

    calc_mid2 = (u_start + u_end)/2
    calc_mid2 = calc_mid2 - np.ones((dim,))*1./dim
    calc_mid2 = (calc_mid2 / np.linalg.norm(calc_mid2)) * np.sqrt((dim-1)/dim)
    calc_mid2 = calc_mid2 + np.ones((dim,))*1./dim

    calc_mid3 = calc_mid2.copy()
    calc_mid3_f = np.fft.fft(calc_mid3)
    calc_mid3_f = calc_mid3_f / (calc_mid3_f.real ** 2 + calc_mid3_f.imag ** 2)
    calc_mid3 = np.fft.ifft(calc_mid3_f).real

    ax[i].scatter(u_start[0], u_start[1], u_start[2], color='red')
    ax[i].scatter(u_end[0], u_end[1], u_end[2], color='blue')
    # ax[i].scatter(u_mid[0], u_mid[1], u_mid[2], color='orange', alpha=.5)
    ax[i].scatter(calc_mid[0], calc_mid[1], calc_mid[2], color='green', alpha=0.7)
    ax[i].scatter(calc_mid2[0], calc_mid2[1], calc_mid2[2], color='purple', alpha=0.7)
    ax[i].scatter(calc_mid2[0], calc_mid2[1], calc_mid2[2], color='black', alpha=0.7)


plt.show()
