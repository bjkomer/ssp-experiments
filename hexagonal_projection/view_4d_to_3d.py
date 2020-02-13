import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from utils import get_simplex_coordinates


n = 4

points_nd = np.eye(n)  # * np.sqrt(n)
# want the 4 points on a tetrahedron
# points_3d = np.zeros((n, 3))
points_3d = np.array([
    [1, -1, -1],
    [-1, 1, -1],
    [-1, -1, 1],
    [1, 1, 1],
])

transform_mat = np.linalg.lstsq(points_3d, points_nd)

x_axis = transform_mat[0][0, :]
y_axis = transform_mat[0][1, :]
z_axis = transform_mat[0][2, :]

default_xyz_axes = np.vstack([x_axis, y_axis, z_axis]).T

default_xyz_axes = get_simplex_coordinates(3)


def xyzw_to_xyz_v(coords, xyz_axes=default_xyz_axes):
    """
    Projects a 3D hexagonal coordinate into the
    corresponding 2D coordinate

    coords is a (n, 3) matrix
    xy_axes is a (3, 2) matrix
    """

    return np.dot(coords, xyz_axes)


# xyzw = np.random.uniform(-10, 10, size=(300, 4))

xyzw = np.zeros((40, 4))
xyzw[:10, 0] = np.arange(10)
xyzw[10:20, 1] = np.arange(10)
xyzw[20:30, 2] = np.arange(10)
xyzw[30:, 3] = np.arange(10)

xyz = xyzw_to_xyz_v(xyzw)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(xyz[:10, 0], xyz[:10, 1], xyz[:10, 2], color='red')
ax.scatter(xyz[10:20, 0], xyz[10:20, 1], xyz[10:20, 2], color='green')
ax.scatter(xyz[20:30, 0], xyz[20:30, 1], xyz[20:30, 2], color='blue')
ax.scatter(xyz[30:, 0], xyz[30:, 1], xyz[30:, 2], color='orange')

# ax.scatter(default_xyz_axes[:, 0], default_xyz_axes[:, 1], default_xyz_axes[:, 2], color='purple')
# ax.scatter(0, 0, 0, color='yellow')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_aspect('equal')

plt.show()
