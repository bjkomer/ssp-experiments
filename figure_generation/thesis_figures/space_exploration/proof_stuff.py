import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.linalg import circulant
from spatial_semantic_pointers.utils import make_good_unitary
import nengo.spa as spa
import sys

if len(sys.argv) > 1:
    dim = int(sys.argv[1])
else:
    dim = 5

# the main origin of the space
origin = np.zeros((dim, ))
origin[0] = 1

# the opposite pole
opposite_f = np.zeros((dim, ), dtype='Complex64')
opposite_f[:] = -1
opposite_f[0] = 1
if dim % 2 == 0:
    opposite_f[dim // 2] = 1
opposite = np.fft.ifft(opposite_f).real
assert np.allclose(np.linalg.norm(opposite), 1)
print("opposite", opposite)
cross_vec = origin - opposite
print("cross_vec", cross_vec)

diameter = np.linalg.norm(cross_vec)
print('diameter', diameter)
print('theoretical diameter', np.sqrt((dim-1)/dim)*2)

# one-hot rotation directions
n = (dim-1)//2

ohr = np.zeros((n, dim))

# need to pick a phi, this should do, or maybe all the way?

for p, phi in enumerate([np.pi, np.pi/2.]):

    if p == 0:
        print("phi = pi")
    else:
        print("phi = pi/2")
    for i in range(n):
        u = np.zeros((dim, ), dtype='Complex64')
        u[:] = 1
        u[(i+1)] = np.exp(1.j*phi)
        u[-(i+1)] = np.conj(u[i])

        ohr[i] = np.fft.ifft(u).real

    print(ohr)
    print("")


# look at what each one-hot vector looks like
# these all live in that one great circle path, doesn't help me too much
# all_one_hot = np.eye(dim)
#
# for i in range(1, dim):
#     print(np.fft.fft(all_one_hot[i]))

print(np.exp(1.j*np.pi/2.)*np.exp(-1.j*np.pi/2.))
print(np.exp(1.j*np.pi/3.)+np.exp(-1.j*np.pi/3.))
print(np.exp(1.j*np.pi/4.)+np.exp(-1.j*np.pi/4.))


for i in range(n):
    for j in range(i+1, n):
        print(np.round(np.dot(ohr[i]-origin, ohr[j]-origin), 3))