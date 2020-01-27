import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import seaborn as sns
import numpy as np

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

# plot the axes lines
l = 3
ax.plot([0, l], [0, 0], [0, 0])
ax.plot([0, 0], [0, l], [0, 0])
ax.plot([0, 0], [0, 0], [0, l])

ax.set_xlim(0, 3)
ax.set_ylim(0, 3)
ax.set_zlim(0, 3)

plt.show()
