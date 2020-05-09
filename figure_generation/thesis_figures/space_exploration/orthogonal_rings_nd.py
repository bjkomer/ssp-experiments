# visualization of the space of unitary vectors in 3D (hard to visualize more than that)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
from matplotlib.colors import hsv_to_rgb


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


fig = plt.figure()
dim = 35
ax = fig.add_subplot(111, projection='3d')

res = 64#256#32#256#64

# xs = np.linspace(0, 4, res)
xs = np.linspace(0, 4, res)

n_rings = dim // 2

palette = sns.color_palette("hls", n_rings)

for i in range(n_rings):

    u = orthogonal_unitary(dim, i+1, np.pi/2.)
    for x in xs:
        ux = np.fft.ifft(np.fft.fft(u) ** x).real
        ax.scatter(ux[0], ux[1], ux[2], color=palette[i])


plt.show()
