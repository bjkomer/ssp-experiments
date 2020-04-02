import numpy as np
import matplotlib.pyplot as plt
import sys

fname = sys.argv[1]


limit_high = 30
limit_low = -30
# limit_high = 5
# limit_low = -5
res = 32#128
res = 40

# using res + 1 as endpoints, so there will be res bins between
xs = np.linspace(limit_low, limit_high, res + 1)
ys = np.linspace(limit_low, limit_high, res + 1)


diff = xs[1] - xs[0]

data = np.load(fname)

if 'spikes' in data:

    spikes = data['spikes']
    bound_spikes = data['bound_spikes']
    pos = data['pos_2d']

    n_samples = spikes.shape[0]
    n_neurons = spikes.shape[1]

    # spatial firing for each neuron
    img = np.zeros((n_neurons, res, res))
    bound_img = np.zeros((n_neurons, res, res))

    for i, x in enumerate(xs[:-1]):
        for j, y in enumerate(ys[:-1]):
            ind = np.where((pos[:, 0] >= x) & (pos[:, 0] < x + diff) & (pos[:, 1] >= y) & (pos[:, 1] < y + diff))
            if len(ind[0]) > 0:
                img[:, i, j] = np.mean(spikes[ind[0], :], axis=0)
                bound_img[:, i, j] = np.mean(bound_spikes[ind[0], :], axis=0)

    # print(np.min(img))
    # print(np.max(img))
else:
    img = data['img']
    bound_img = data['bound_img']

    n_neurons = img.shape[0]
    res = img.shape[1]

max_val = np.max(img)

# adding border around the edges for interpretability
border_val = np.max(img) / 2.
img[:, 0, :] = border_val
img[:, :, 0] = border_val
img[:, -1, :] = border_val
img[:, :, -1] = border_val
# border_val = np.max(bound_img) / 2.
bound_img[:, 0, :] = border_val
bound_img[:, :, 0] = border_val
bound_img[:, -1, :] = border_val
bound_img[:, :, -1] = border_val


# print(pos)
# print(pos.shape)
# print(spikes.shape)
#
# plt.plot(pos[:, 0], pos[:, 1])
# plt.show()

view_n_neurons = min(n_neurons, 625)
# view_n_neurons = 144#625
side_len = int(np.ceil(np.sqrt(view_n_neurons)))

full_img = np.zeros((side_len*res, side_len*res))
bound_full_img = np.zeros((side_len*res, side_len*res))

for i in range(0, view_n_neurons):
    ix = i % side_len
    iy = i // side_len
    full_img[ix * res:(ix + 1) * res, iy * res:(iy + 1) * res] = img[i, :, :]
    bound_full_img[ix * res:(ix + 1) * res, iy * res:(iy + 1) * res] = bound_img[i, :, :]

if True:
    plt.figure(figsize=(8, 8))
    plt.imshow(full_img, vmin=0, vmax=max_val)
    plt.figure(figsize=(8, 8))
    plt.imshow(bound_full_img, vmin=0, vmax=max_val)
    plt.show()
else:
    plt.figure(figsize=(8, 8))
    plt.imshow(full_img[:res*5,:res*5], vmin=0, vmax=max_val)
    plt.figure(figsize=(8, 8))
    plt.imshow(bound_full_img[:res*5,:res*5], vmin=0, vmax=max_val)
    plt.show()

# fig, ax = plt.subplots(side_len, side_len)
# plt.axis('off')
#
# for i in range(0, view_n_neurons):
#     ix = i % side_len
#     iy = i // side_len
#     # plt.imshow(img[i, :, :])
#     # plt.imshow(bound_img[i, :, :])
#     ax[ix, iy].imshow(bound_img[i, :, :])
# fig.tight_layout()
# plt.show()
