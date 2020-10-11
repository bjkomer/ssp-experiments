import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
from matplotlib.colors import hsv_to_rgb


def trig_indentity_check(n_samples=40):
    offsets = np.linspace(0, 2 * np.pi, n_samples)
    for n_proj in range(2, 20):
        angs = np.linspace(0, 1 * np.pi, n_proj + 1)[:-1]
        total = 0
        for i in range(n_samples):
            ang_set = angs + offsets[i]
            # print(np.sum(np.cos(ang_set)**2))
            assert(np.allclose(np.sum(np.cos(ang_set)**2), n_proj/2))
    print("Check Success")

trig_indentity_check()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n_proj = 3#3
n_proj = 3
angs = np.linspace(0, 1*np.pi, n_proj+1)[:-1]

n_samples = 40
offsets = np.linspace(0, 2*np.pi, n_samples)
# offsets = np.linspace(0, np.pi/n_proj, n_samples)


points = np.zeros((n_samples, n_proj))

for i in range(n_samples):
    ang_set = angs + offsets[i]
    # points[i, :] = np.cos(ang_set)**2
    points[i, :] = np.cos(ang_set)

ax.scatter(points[:, 0], points[:, 1], points[:, 2])

ax.scatter(0, 0, 0)

print(np.linalg.norm(points, axis=1))

plt.show()
