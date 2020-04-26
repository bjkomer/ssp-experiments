import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns

from nengo.dists import UniformHypersphere


n_samples = 10000
dim = 3

# points = UniformHypersphere(surface=True).sample(n=n_samples, d=dim)

# points = np.random.uniform(-1, 1, size=(n_samples, dim))
# for i in range(n_samples):
    # points[i, :] /= np.linalg.norm(points[i, :])
    # points[i, :] /= np.sqrt(np.sum(points[i, :] ** 2, axis=None, keepdims=False))

points = np.random.randn(n_samples, dim)
points /= np.sqrt(np.sum(points ** 2, axis=1, keepdims=True))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', alpha=0.1)

plt.show()
