import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as plticker
import numpy as np
import seaborn as sns

rot = 2

# palette = sns.color_palette("hls", 7)
palette = sns.color_palette()

normal = np.array([1, 1, 1])
normal = normal / np.linalg.norm(normal)
default_x_axis = np.array([1, -1, 0])
default_y_axis = np.array([-1, -1, 2])
default_x_axis = default_x_axis / np.linalg.norm(default_x_axis)
default_y_axis = default_y_axis / np.linalg.norm(default_y_axis)

# Used in the vectorized functions
default_xy_axes = np.vstack([default_x_axis, default_y_axis]).T


print(np.dot(normal, default_x_axis))
print(np.dot(normal, default_y_axis))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
loc = plticker.MultipleLocator(base=.5)
ax.xaxis.set_major_locator(loc)
ax.yaxis.set_major_locator(loc)
ax.zaxis.set_major_locator(loc)
if rot == 1:
    ax.view_init(elev=0, azim=-45)
elif rot == 0:
    ax.view_init(elev=26, azim=-32)
elif rot == 2:
    ax.view_init(elev=26, azim=-32)


ax.plot([0, default_x_axis[0]], [0, default_x_axis[1]], [0, default_x_axis[2]], color=palette[0])
ax.plot([0, default_y_axis[0]], [0, default_y_axis[1]], [0, default_y_axis[2]], color=palette[1])
ax.plot([0, normal[0]], [0, normal[1]], [0, normal[2]], color=palette[2])

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

# this feature was removed from matplotlib because it was buggy
#ax.set_aspect('equal')

#print(np.dot(default_x_axis, default_y_axis))

# list with 2 elements of (res, res)
#meshgrid = np.meshgrid(xs, ys)

res = 32
xs = np.linspace(-1, 1, res)
ys = np.linspace(-1, 1, res)

points = np.zeros((2, res * res))
for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        points[0, i*res + j] = x
        points[1, i*res + j] = y

print(default_xy_axes.shape)

points_3d = np.tensordot(points, default_xy_axes, axes=[0, 1])

print(points_3d.shape)

if rot == 0:
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], alpha=.1, color=palette[0])

scaling = 1.22472

ax.plot([0, scaling], [0, 0], [0, 0], color=palette[3])

ax.plot([0, scaling], [0, 0], [0, 0], color=palette[4])
ax.plot([0, 0], [0, scaling], [0, 0], color=palette[5])
ax.plot([0, 0], [0, 0], [0, scaling], color=palette[6])

# connection in the normal direction
if rot == 1:
    ax.plot([default_y_axis[0], 0], [default_y_axis[1], 0], [default_y_axis[2], scaling], color=palette[2])
    ax.plot([0, -default_y_axis[0]], [0, -default_y_axis[1]], [0, -default_y_axis[2]], color=palette[1], linestyle='--')
elif rot == 2:
    # draw a cube
    for op in [[[0, 0], [1, 1]], [[1, 1], [0, 0]], [[1, 1], [1, 1]]]:
        ax.plot([0, 1], op[0], op[1], color=palette[4])
        ax.plot(op[0], [0, 1], op[1], color=palette[5])
        ax.plot(op[0], op[1], [0, 1], color=palette[6])

    # draw the normal all the way across
    ax.plot([0, 1], [0, 1], [0, 1], color=palette[2])

    # draw the base vector
    ax.plot([0, 1], [0, 1], [0, 0], color=palette[7])

ax.set_axis_off()

plt.show()
