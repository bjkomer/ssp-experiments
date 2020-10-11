from jax import grad, vmap, jacfwd, jacrev
from jax.ops import index, index_update
import jax.numpy as np
import numpy
import matplotlib.pyplot as plt


def encode_2d_point_even(pos, phis):
    # dim = len(phis)*2 + 2
    dim = phis.shape[1] + 2
    xf = np.zeros((dim,), dtype='complex64')
    xf = index_update(xf, index[0], 1)
    xf = index_update(xf, index[1:-1], np.exp(1.j*phis[0, :]))
    xf = index_update(xf, index[-1], 1)

    yf = np.zeros((dim,), dtype='complex64')
    yf = index_update(yf, index[0], 1)
    yf = index_update(yf, index[1:-1], np.exp(1.j*phis[1, :]))
    yf = index_update(yf, index[-1], 1)
    # jax version of irfft assumes there is a nyquist frequency
    # they have not implemented it for odd dimensions
    ret = np.fft.irfft(xf**pos[0]*yf**pos[1])
    return ret

# grad_encode_point_x = grad(encode_2d_point_even, argnums=0)
# grad_encode_point_phis = grad(encode_2d_point_even, argnums=1)
grad_encode_point_x = jacrev(encode_2d_point_even, argnums=0)
grad_encode_point_phis = jacrev(encode_2d_point_even, argnums=1)

#print(encode_point(2., np.array([np.pi/2., np.pi/3.])))
#print(grad_encode_point_x(2., np.array([np.pi/2., np.pi/3.])))
#print(grad_encode_point_phis(2., np.array([np.pi/2., np.pi/3.])))

phi_vars = np.array([
    [np.pi/2., 0.],
    [0., np.pi/3.],
])
# phi_vars = np.array([np.pi/2.])
loc = (1.0, 0.5)

print(encode_2d_point_even(loc, phi_vars))
print("")
print(grad_encode_point_x(loc, phi_vars))
print("")
print(grad_encode_point_phis(loc, phi_vars))
print("")


# trying it with a batch dimension
batch_encode_point = vmap(encode_2d_point_even, (0, None))
grad_encode_point_x = vmap(jacrev(encode_2d_point_even, argnums=0), (0, None))
grad_encode_point_phis = vmap(jacrev(encode_2d_point_even, argnums=1), (0, None))

loc = numpy.array([
    [0.0, 0.5],
    [1.2, -1.3],
    [1.4, 0.0],
    [1.5, 0.4],
])

phi_vars = numpy.array([
    [np.pi/2., 0.],
    [0., np.pi/3.],
])

print(batch_encode_point(loc, phi_vars))
print("")
print(grad_encode_point_x(loc, phi_vars))
print("")
print(grad_encode_point_phis(loc, phi_vars))
print("")


a = batch_encode_point(loc, phi_vars)
print(a.shape)
b = grad_encode_point_x(loc, phi_vars)
print(b.shape)
c = grad_encode_point_phis(loc, phi_vars)
print(c.shape)