import numpy as np
import matplotlib.pyplot as plt
from spatial_semantic_pointers.utils import get_centering_transformation, make_good_unitary, get_heatmap_vectors, \
    make_fixed_dim_periodic_axis, get_fixed_dim_sub_toriod_axes


def make_pole_crossing_axis(dim):
    xf = np.ones((dim, ), dtype='complex64')
    xf[1:(dim + 1) // 2] = -1
    xf[-1:dim // 2:-1] = -1  # complex conjugate is also -1

    assert np.allclose(np.abs(xf), 1)
    x = np.fft.ifft(xf).real
    assert np.allclose(np.fft.fft(x), xf)
    assert np.allclose(np.linalg.norm(x), 1)
    return x


dim = 128#128
# X = make_good_unitary(dim=dim).v
# X = make_fixed_dim_periodic_axis(dim=dim)
X = make_pole_crossing_axis(dim=dim)
# X, Y = get_fixed_dim_sub_toriod_axes(dim=dim)
# X = X.v
mat, offset = get_centering_transformation(dim=dim)

limit = 5#50
res = 512 #128
xs = np.linspace(-limit, limit, res)

ref_x = 0
ref_vec = np.fft.ifft(np.fft.fft(X)**ref_x).real
ref_vec_offset = ref_vec @ mat.T
ref_vec_offset /= np.linalg.norm(ref_vec_offset)

sim = np.zeros((res, 2))
for i, x in enumerate(xs):
    vec = np.fft.ifft(np.fft.fft(X)**x).real
    vec_offset = vec @ mat.T
    vec_offset /= np.linalg.norm(vec_offset)

    sim[i, 0] = np.dot(vec, ref_vec)
    sim[i, 1] = np.dot(vec_offset, ref_vec_offset)

plt.plot(xs, sim[:, 0], color='blue')
plt.plot(xs, sim[:, 1], color='orange')
plt.legend(['normal', 'recentered'])

plt.show()
