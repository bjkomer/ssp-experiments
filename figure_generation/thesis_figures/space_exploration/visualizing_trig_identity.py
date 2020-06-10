import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
from matplotlib.colors import hsv_to_rgb

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n_samples = 40
offsets = np.linspace(0, 2*np.pi, n_samples)

n_proj = 3#3
angs = np.linspace(0, 1*np.pi, n_proj+1)[:-1]

points = np.zeros((n_samples, n_proj))

for i in range(n_samples):
    ang_set = angs + offsets[i]
    # points[i, :] = np.cos(ang_set)**2
    points[i, :] = np.cos(ang_set)

ax.scatter(points[:, 0], points[:, 1], points[:, 2])

ax.scatter(0, 0, 0)

print(np.linalg.norm(points, axis=1))

plt.show()
