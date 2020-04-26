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


unitaries = [
    unitary(np.pi/3., +1),
    unitary(-np.pi/8., +1),
    unitary(np.pi/3., -1),
    unitary(np.pi/8., -1),
]

titles = [
    ''
]

palette = sns.color_palette("hls", len(unitaries))

# for shading the space
unitary_points = random_unitary(n_samples=2000)

fig = plt.figure(tight_layout=True, figsize=(16, 4))
ax = []
plot_limit = 1
for i in range(len(unitaries)):
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


for i, u in enumerate(unitaries):
    if i == 3:
        xs = np.linspace(0, 16, res*2)
    for x in xs:
        ux = np.fft.ifft(np.fft.fft(u) ** x).real
        ax[i].scatter(ux[0], ux[1], ux[2], color=palette[i])


plt.show()
