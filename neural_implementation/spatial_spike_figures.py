import numpy as np
import matplotlib.pyplot as plt
import sys

limit_low = -5
limit_high = 5
res = 128

# using res + 1 as endpoints, so there will be res bins between
xs = np.linspace(limit_low, limit_high, res + 1)
ys = np.linspace(limit_low, limit_high, res + 1)


diff = xs[1] - xs[0]

if len(sys.argv) > 1:
    fname = sys.argv[1]
else:
    fname = '/media/ctnuser/53f2c4b3-4b3b-4768-ba69-f0a3da30c237/ctnuser/data/neural_implementation_output/output_orth_hex_toroid_d25_mixed_50hz_hilbert_traj_100s.npz'
data = np.load(fname)

spikes = data['spikes']
pos = data['pos_2d']
metadata = data['metadata']

n_samples = spikes.shape[0]
n_neurons = spikes.shape[1]

# spatial firing for each neuron
img = np.zeros((n_neurons, res, res))

traj_img = np.zeros((res, res))

for i, x in enumerate(xs[:-1]):
    for j, y in enumerate(ys[:-1]):
        ind = np.where((pos[:, 0] >= x) & (pos[:, 0] < x + diff) & (pos[:, 1] >= y) & (pos[:, 1] < y + diff))
        if len(ind[0]) > 0:
            img[:, i, j] = np.mean(spikes[ind[0], :], axis=0)
            traj_img[i, j] = 1

fig = plt.figure()
plt.imshow(traj_img)

# indices for the neurons to show
# select 3 grid cell, 3 band cell, 3 place cell

# mix indices in the metadata
PLACE = 0
GRID = 1
BAND = 2

grids_a = np.where((metadata[:, 2] == GRID) & (metadata[:, 0] == 0))[0][0]
grids_b = np.where((metadata[:, 2] == GRID) & (metadata[:, 0] == 1))[0][0]
grids_c = np.where((metadata[:, 2] == GRID) & (metadata[:, 0] == 2))[0][0]

bands_a = np.where((metadata[:, 2] == BAND) & (metadata[:, 1] == 0) & (metadata[:, 0] == 1))[0][0]
bands_b = np.where((metadata[:, 2] == BAND) & (metadata[:, 1] == 1) & (metadata[:, 0] == 1))[0][0]
bands_c = np.where((metadata[:, 2] == BAND) & (metadata[:, 1] == 2) & (metadata[:, 0] == 1))[0][0]

places_abc = np.where(metadata[:, 2] == PLACE)[0]


inds = [
    [grids_a, grids_b, grids_c],
    [bands_a, bands_b, bands_c],
    [places_abc[0], places_abc[4], places_abc[9]],
]

fig, ax = plt.subplots(3, 3, tight_layout=True)

# clipping spike values to max at 1 for spiking
# a value of 0 means no spike
img = np.clip(img, 0., 1.)

# set the paths to grey
base_img = np.ones((res-2, res-2, 3))
base_img[:, :, 0] -= traj_img[1:-1, 1:-1]*.5
base_img[:, :, 1] -= traj_img[1:-1, 1:-1]*.5
base_img[:, :, 2] -= traj_img[1:-1, 1:-1]*.5

for i in range(3):
    for j in range(3):
        spike_img = base_img.copy()
        spike_img[:, :, 0] += img[inds[i][j], 1:-1, 1:-1] * .5
        spike_img[:, :, 1] -= img[inds[i][j], 1:-1, 1:-1] * .5
        spike_img[:, :, 2] -= img[inds[i][j], 1:-1, 1:-1] * .5
        ax[i, j].imshow(spike_img)
        ax[i, j].set_axis_off()

plt.show()
