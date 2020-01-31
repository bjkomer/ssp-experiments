import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1, projection='3d')


normal = np.array([1, 1, 1])
default_x_axis = np.array([1, -1, 0])
default_y_axis = np.array([-1, -1, 2])
normal = normal / np.linalg.norm(normal)
default_x_axis = default_x_axis / np.linalg.norm(default_x_axis)
default_y_axis = default_y_axis / np.linalg.norm(default_y_axis)

l_plane = 1.5

c1 = normal + l_plane * default_x_axis + l_plane * default_y_axis
c2 = normal + l_plane * default_x_axis - l_plane * default_y_axis
c3 = normal - l_plane * default_x_axis - l_plane * default_y_axis
c4 = normal - l_plane * default_x_axis + l_plane * default_y_axis

x = np.array([c1[0], c2[0], c3[0], c4[0]])
y = np.array([c1[1], c2[1], c3[1], c4[1]])
z = np.array([c1[2], c2[2], c3[2], c4[2]])

verts = [list(zip(x, y, z))]

# xx, yy = np.meshgrid(xx, yy)
pc = Poly3DCollection(verts, facecolors='C0', alpha=0.5, linewidths=1)
pc.set_alpha(0.5)
pc.set_facecolor('C0')
ax.add_collection3d(pc)

palette = sns.color_palette()

# plot the axes lines
l = 3
ax.plot([0, l], [0, 0], [0, 0], color=palette[0], linewidth=3)
ax.plot([0, 0], [0, l], [0, 0], color=palette[1], linewidth=3)
ax.plot([0, 0], [0, 0], [0, l], color=palette[2], linewidth=3)

ax.plot([-l, 0], [0, 0], [0, 0], color=palette[0], linestyle=':', linewidth=3)
ax.plot([0, 0], [-l, 0], [0, 0], color=palette[1], linestyle=':', linewidth=3)
ax.plot([0, 0], [0, 0], [-l, 0], color=palette[2], linestyle=':', linewidth=3)

ax.set_xlim(0, 3)
ax.set_ylim(0, 3)
ax.set_zlim(0, 3)

ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.zaxis.set_major_locator(MaxNLocator(integer=True))

ax.set_aspect('equal')

fig.tight_layout()

fig = plt.figure()

A = np.vstack([default_x_axis, default_y_axis]).T

y_axis = np.array([[1, 0, 0]]) @ A
x_axis = np.array([[0, 1, 0]]) @ A
z_axis = np.array([[0, 0, 1]]) @ A

ax_2d = fig.add_subplot(1, 1, 1)



# draw the axes

alpha = 0.9
ax_2d.plot([0, l*x_axis[0, 0]], [0, l*x_axis[0, 1]], color=palette[0], linewidth=3, alpha=alpha)
ax_2d.plot([0, l*y_axis[0, 0]], [0, l*y_axis[0, 1]], color=palette[1], linewidth=3, alpha=alpha)
ax_2d.plot([0, l*z_axis[0, 0]], [0, l*z_axis[0, 1]], color=palette[2], linewidth=3, alpha=alpha)

ax_2d.plot([0, -l*x_axis[0, 0]], [0, -l*x_axis[0, 1]], color=palette[0], linestyle=':', linewidth=3, alpha=alpha)
ax_2d.plot([0, -l*y_axis[0, 0]], [0, -l*y_axis[0, 1]], color=palette[1], linestyle=':', linewidth=3, alpha=alpha)
ax_2d.plot([0, -l*z_axis[0, 0]], [0, -l*z_axis[0, 1]], color=palette[2], linestyle=':', linewidth=3, alpha=alpha)

# draw some hexagons

# convenience values
X = np.array([x_axis[0, 0], x_axis[0, 1]])
Y = np.array([y_axis[0, 0], y_axis[0, 1]])
Z = np.array([z_axis[0, 0], z_axis[0, 1]])
Xx = X[0]
Xy = X[1]
Yx = Y[0]
Yy = Y[1]
Zx = Z[0]
Zy = Z[1]

alpha_hex = 1.0

ax_2d.plot(
    [X[0], -Y[0], Z[0], -X[0], Y[0], -Z[0], X[0]],
    [X[1], -Y[1], Z[1], -X[1], Y[1], -Z[1], X[1]],
    color='grey',
    alpha=alpha_hex,
)

offset_x = [
    -Y[0] + Z[0],
    +Y[0] - Z[0],
    -X[0] + Z[0],
    +X[0] - Z[0],
    -X[0] + Y[0],
    +X[0] - Y[0],
]
offset_y = [
    -Y[1] + Z[1],
    +Y[1] - Z[1],
    -X[1] + Z[1],
    +X[1] - Z[1],
    -X[1] + Y[1],
    +X[1] - Y[1],
]

for i in range(len(offset_x)):
    ax_2d.plot(
        [X[0] + offset_x[i], -Y[0] + offset_x[i], Z[0] + offset_x[i], -X[0] + offset_x[i], Y[0] + offset_x[i], -Z[0] + offset_x[i], X[0] + offset_x[i]],
        [X[1] + offset_y[i], -Y[1] + offset_y[i], Z[1] + offset_y[i], -X[1] + offset_y[i], Y[1] + offset_y[i], -Z[1] + offset_y[i], X[1] + offset_y[i]],
        color='grey',
        alpha=alpha_hex,
    )

ax_2d.set_aspect('equal')

# set the x-spine
ax_2d.spines['left'].set_position('zero')

# turn off the right spine/ticks
ax_2d.spines['right'].set_color('none')
ax_2d.yaxis.tick_left()

# set the y-spine
ax_2d.spines['bottom'].set_position('zero')

# turn off the top spine/ticks
ax_2d.spines['top'].set_color('none')
ax_2d.xaxis.tick_bottom()

ax_2d.set_xlim([-2.5, 2.5])
ax_2d.set_ylim([-2.5, 2.5])

fig.tight_layout()

plt.show()
