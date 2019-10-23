import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from spatial_semantic_pointers.utils import make_good_unitary, power

dim = 512
res = 512
seed = 13
limit = 5

xs = np.linspace(-limit, limit, res)

rng = np.random.RandomState(seed=seed)

X = make_good_unitary(dim=dim, rng=rng)

hmv = np.zeros((res, dim))

locs = [
    0,
    2.2,
    -3.4,
]
locs = [
    0,
    3.2,
]
n_locs = len(locs)

vecs = np.zeros((n_locs, dim))

for i, loc in enumerate(locs):
    vecs[i, :] = power(X, loc).v

sims = np.zeros((res, n_locs))

for i, x in enumerate(xs):
    hmv[i, :] = power(X, x).v
    for j in range(n_locs):
        sims[i, j] = np.dot(vecs[j], hmv[i, :])

plt.plot(xs, sims, linewidth=2.5)

plt.legend(
    ["k = {}".format(k) for k in locs],
    fontsize=16
)

plt.xlabel("Exponent", fontsize=18)
plt.ylabel("Dot Product", fontsize=18)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.savefig("ssp_1d.pdf", dpi=900, bbox_inches='tight')

plt.show()

