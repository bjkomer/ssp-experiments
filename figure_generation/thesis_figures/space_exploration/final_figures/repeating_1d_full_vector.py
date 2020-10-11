import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import numpy as np
import nengo.spa as spa
import nengo_spa
from spatial_semantic_pointers.utils import make_good_unitary, power, get_heatmap_vectors, make_fixed_dim_periodic_axis



fix, ax = plt.subplots(1, 2, tight_layout=True, figsize=(8, 2))

fontsize = 16

dims = [7, 16, 33, 64]
rng = np.random.RandomState(seed=13)

dim = 32
scales = [4, 8]
limits = [10, 10]
res = 256

sim = np.zeros((res, dim))

origin = np.zeros((dim, ))
origin[0] = 1

for k, scale in enumerate(scales):
    X = make_fixed_dim_periodic_axis(dim=dim, period=scale)
    xs = np.linspace(-limits[k], limits[k], res)
    for i, x in enumerate(xs):
        sim[i, :] = np.fft.ifft(np.fft.fft(X)**x).real
    ax[k].plot(xs, sim)
    ax[k].set_title('Period = {}'.format(scale), fontsize=fontsize)
    # loc = plticker.MultipleLocator(base=5)  # this locator puts ticks at regular intervals
    # ax[i].xaxis.set_major_locator(loc)

plt.show()
