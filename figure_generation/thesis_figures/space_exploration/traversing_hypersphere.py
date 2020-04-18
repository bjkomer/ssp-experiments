import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
from matplotlib.colors import hsv_to_rgb
from spatial_semantic_pointers.utils import make_good_unitary
from scipy.linalg import circulant

dim = 5

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n_samples = 2000
points = np.zeros((n_samples, dim))
# points[0, 0] = 1
points[0, 0] = 0
points[0, 1] = 1

# unitary = -make_good_unitary(dim=dim).v
unitary = make_good_unitary(dim=dim).v
rot_mat = circulant(unitary)

print(unitary)
print(rot_mat)


print("det rot mat: {}".format(np.linalg.det(rot_mat)))

for i in range(1, n_samples):
    points[i, :] = points[i-1, :] @ rot_mat


ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', alpha=0.1)

plt.show()
