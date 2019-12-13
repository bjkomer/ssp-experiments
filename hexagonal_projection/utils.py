import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from spatial_semantic_pointers.utils import power, encode_point
import nengo.spa as spa

# from spatial-cognition/utils.py


def rotate_vector(vec, rot_axis, theta):
    axis = rot_axis.copy()
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.0)
    b, c, d = -axis*np.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.dot(np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                            [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                            [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]]),
                  vec)


grid_angle = 0 #15

normal = np.array([1, 1, 1])
default_x_axis = np.array([1, -1, 0])
default_y_axis = np.array([-1, -1, 2])
normal = normal / np.linalg.norm(normal)
default_x_axis = default_x_axis / np.linalg.norm(default_x_axis)
default_y_axis = default_y_axis / np.linalg.norm(default_y_axis)

default_x_axis = rotate_vector(default_x_axis, normal, grid_angle * np.pi / 180)
default_y_axis = rotate_vector(default_y_axis, normal, grid_angle * np.pi / 180)

# Used in the vectorized functions
default_xy_axes = np.vstack([default_x_axis, default_y_axis]).T


def xyz_to_xy(coord, x_axis=default_x_axis, y_axis=default_y_axis):
    """
    Projects a 3D hexagonal coordinate into the
    corresponding 2D coordinate
    """
    x = np.dot(x_axis, coord)
    y = np.dot(y_axis, coord)

    return np.array([x, y])


def xyz_to_xy_v(coords, xy_axes=default_xy_axes):
    """
    Projects a 3D hexagonal coordinate into the
    corresponding 2D coordinate

    coords is a (n, 3) matrix
    xy_axes is a (3, 2) matrix
    """

    return np.dot(coords, xy_axes)


def xy_to_xyz(coord, x_axis=default_x_axis, y_axis=default_y_axis):
    """
    Converts a 2D coordinate into the corresponding
    3D coordinate in the hexagonal representation
    """
    return x_axis*coord[0]+y_axis*coord[1]


def xy_to_xyz_v(coords, xy_axes=default_xy_axes):
    """
    Converts a 2D coordinate into the corresponding
    3D coordinate in the hexagonal representation
    coord is a (n, 2) matrix
    xy_axes is a (3, 2) matrix
    """

    return np.dot(coords, xy_axes.T)


def encode_point_3d(x, y, z, x_axis_sp, y_axis_sp, z_axis_sp):

    return power(x_axis_sp, x) * power(y_axis_sp, y) * power(z_axis_sp, z)


def get_projected_heatmap_vectors(xs, ys, x_axis_sp, y_axis_sp, z_axis_sp):
    if x_axis_sp.__class__.__name__ == 'SemanticPointer':
        dim = len(x_axis_sp.v)
    else:
        dim = len(x_axis_sp)
        x_axis_sp = spa.SemanticPointer(data=x_axis_sp)
        y_axis_sp = spa.SemanticPointer(data=y_axis_sp)
        z_axis_sp = spa.SemanticPointer(data=z_axis_sp)

    vectors = np.zeros((len(xs), len(ys), dim))

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            # xyz = xy_to_xyz([x, y])
            xyz = xy_to_xyz_v(np.array([[x, y]]))[0, :]
            p = encode_point_3d(
                x=xyz[0],
                y=xyz[1],
                z=xyz[2],
                x_axis_sp=x_axis_sp,
                y_axis_sp=y_axis_sp,
                z_axis_sp=z_axis_sp,
            )
            vectors[i, j, :] = p.v

    return vectors


if __name__ == '__main__':
    # run some tests

    xyz = np.random.uniform(-10, 10, size=(300, 3))
    # xyz = np.zeros((30, 3))
    # xyz[:10, 0] = np.arange(10)
    # xyz[10:20, 1] = np.arange(10)
    # xyz[20:, 2] = np.arange(10)

    xy = xyz_to_xy_v(xyz)



    if False:
        plt.scatter(xy[:, 0], xy[:, 1])
        # plt.xlim([-8, 8])
        # plt.ylim([-8, 8])
        plt.axes().set_aspect('equal', 'datalim')
        plt.show()
    elif False:
        xyz_back = xy_to_xyz_v(xy)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(xyz_back[:, 0], xyz_back[:, 1], xyz_back[:, 2])
        # ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()
    elif True:
        xy = np.random.uniform(-10, 10, size=(300, 2))
        xyz_back = xy_to_xyz_v(xy)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(xyz_back[:, 0], xyz_back[:, 1], xyz_back[:, 2])
        # ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()


