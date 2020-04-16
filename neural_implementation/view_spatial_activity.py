import numpy as np
import matplotlib.pyplot as plt
import sys

limit_low = -5
limit_high = 5
res = 32#128

# using res + 1 as endpoints, so there will be res bins between
xs = np.linspace(limit_low, limit_high, res + 1)
ys = np.linspace(limit_low, limit_high, res + 1)


diff = xs[1] - xs[0]

fname = sys.argv[1]
data = np.load(fname)

# data = np.load('output_70s.npz')
# data = np.load('/media/ctnuser/53f2c4b3-4b3b-4768-ba69-f0a3da30c237/ctnuser/data/neural_implementation_output/output_grid_cell_70s.npz')
# data = np.load('output_band_70s.npz')

spikes = data['spikes']
pos = data['pos_2d']

n_samples = spikes.shape[0]
n_neurons = spikes.shape[1]

# spatial firing for each neuron
img = np.zeros((n_neurons, res, res))

for i, x in enumerate(xs[:-1]):
    for j, y in enumerate(ys[:-1]):
        ind = np.where((pos[:, 0] >= x) & (pos[:, 0] < x + diff) & (pos[:, 1] >= y) & (pos[:, 1] < y + diff))
        # print(ind[0])
        # print(len(ind[0]))
        if len(ind[0]) > 0:
            img[:, i, j] = np.mean(spikes[ind[0], :], axis=0)

# print(pos)
# print(pos.shape)
# print(spikes.shape)
#
# plt.plot(pos[:, 0], pos[:, 1])
# plt.show()



# for i in range(n_neurons):
#     plt.imshow(img[i, :, :])
#     plt.show()


fig, ax = plt.subplots(10, 10)

for i in range(100):
    ax[i // 10, i % 10].imshow(img[i, 1:-1, 1:-1])

plt.show()
