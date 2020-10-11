from jax import grad, vmap, jacfwd, jacrev
from jax.ops import index, index_update
import jax.numpy as np
import numpy
import matplotlib.pyplot as plt


def old_encode_point(x, phis):
    dim = len(phis) + 2
    uf = np.zeros((dim,), dtype='complex64')
    #uf[0] = 1
    #uf[1:-1] = np.exp(1.j*phis)
    #uf[-1] = 1
    uf = index_update(uf, index[0], 1)
    uf = index_update(uf, index[1:-1], np.exp(1.j*phis))
    uf = index_update(uf, index[-1], 1)
    # jax version of irfft assumes there is a nyquist frequency
    # they have not implemented it for odd dimensions
    ret = np.fft.irfft(uf**x)
    return ret

def encode_point(x, phis):
    dim = len(phis)*2+1
    uf = np.zeros((dim,), dtype='complex64')
    uf = index_update(uf, index[0], 1)
    uf = index_update(uf, index[1:(dim+1)//2], np.exp(1.j*phis))
    uf = index_update(uf, index[-1:dim//2:-1], np.conj(np.exp(1.j*phis)))
    # this version uses full ifft with complex to allow odd dim
    ret = np.fft.ifft(uf**x).real
    return ret

grad_encode_point_x = grad(encode_point, argnums=0)
grad_encode_point_phis = grad(encode_point, argnums=1)
# grad_encode_point_x = jacrev(encode_point, argnums=0)
# grad_encode_point_phis = jacrev(encode_point, argnums=1)

#print(encode_point(2., np.array([np.pi/2., np.pi/3.])))
#print(grad_encode_point_x(2., np.array([np.pi/2., np.pi/3.])))
#print(grad_encode_point_phis(2., np.array([np.pi/2., np.pi/3.])))

phi_vars = np.array([np.pi/2., np.pi/3.])
# phi_vars = np.array([np.pi/2.])
loc = 1.0

print(encode_point(loc, phi_vars))
print("")
print(grad_encode_point_x(loc, phi_vars))
print("")
print(grad_encode_point_phis(loc, phi_vars))
print("")

assert False

# trying it with a batch dimension
batch_encode_point = vmap(encode_point, (0, None))
grad_encode_point_x = vmap(jacrev(encode_point, argnums=0), (0, None))
grad_encode_point_phis = vmap(jacrev(encode_point, argnums=1), (0, None))

loc = numpy.array([
    [0.0],
    [1.2],
    [1.4],
    [1.5],
])

# phi_vars = numpy.array([
#     [np.pi/2., np.pi/3.],
#     [np.pi/2., np.pi/3.],
#     [np.pi/8., 1.0],
#     [-np.pi/7., np.pi/4.],
# ])
# the phis are not batched, same parameters used for every input, though I guess in theory they could be different
phi_vars = numpy.array([np.pi/2., np.pi/3.])
# phi_vars = numpy.array([
#     [np.pi/2., np.pi/3.],
#     [np.pi / 2., np.pi / 3.],
#     [np.pi / 2., np.pi / 3.],
#     [np.pi / 2., np.pi / 3.],
# ])

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