import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import orthogonal_hex_dir, get_sub_phi, orthogonal_unitary, unitary_from_phi

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

res = 5#64
# phi = np.pi/2.
xs = np.linspace(0, 1, res)[:-1]
print(xs)

dim = 3#7


# check to see how the centers relate
n_rings = (dim-1)//2
centers = np.zeros((n_rings, dim))
for i in range(n_rings):
    vec_a = orthogonal_unitary(dim=dim, index=i + 1, phi=0)
    vec_b = orthogonal_unitary(dim=dim, index=i + 1, phi=np.pi)
    centers[i] = (vec_a + vec_b)/2

print(centers)
mean_center = centers.mean(axis=0)
print(mean_center)
print(mean_center.mean())
print(mean_center.sum())
print(1./dim)
# assert False


phi_const = np.pi/2.
for i, x in enumerate(xs):
    phi_move = 2*np.pi*x
    # vec_a = orthogonal_unitary(dim=dim, index=1, phi=2 * np.pi * x)
    # vec_b = orthogonal_unitary(dim=dim, index=1, phi=2 * np.pi * x + np.pi)
    # vec_a = unitary_from_phi(phis=np.array([phi_const, phi_move]))
    # vec_b = unitary_from_phi(phis=np.array([phi_const, phi_move + np.pi]))
    vec_a = unitary_from_phi(phis=np.array([phi_move, phi_move]))
    vec_b = unitary_from_phi(phis=np.array([phi_move, phi_move + np.pi]))
    vec_mid = (vec_a + vec_b) / 2
    ax.scatter(vec_mid[0], vec_mid[1], vec_mid[2])

plt.show()
