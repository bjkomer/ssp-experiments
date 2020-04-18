import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns

from nengo.dists import UniformHypersphere
import nengo.spa as spa
import nengo_spa
from spatial_semantic_pointers.utils import make_good_unitary

version = 2


n_samples = 25000#10000
dim = 7#3

points = np.zeros((n_samples, dim))

for i in range(n_samples):
    if version == 1:
        sp = nengo_spa.SemanticPointer(data=np.random.randn(dim))
        sp = sp.normalized()
        sp = sp.unitary()
    elif version == 0:
        sp = nengo_spa.SemanticPointer(dim)
        sp.make_unitary()
    elif version == 2:
        sp = make_good_unitary(dim=dim)

    points[i, :] = sp.v


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', alpha=0.01)
# ax.scatter(points[:, 0], points[:, 1], points[:, 3], color='blue', alpha=0.01)

plt.show()
