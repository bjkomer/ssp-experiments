import numpy as np
import matplotlib.pyplot as plt

data = np.load('output/rate_maps.npz')

rate_maps_pred=data['rate_maps_pred']
rate_maps_truth=data['rate_maps_truth']

vmin=None
vmax=None

# vmin=0
# vmax=1

for ni in range(rate_maps_pred.shape[0]):
    print("Neuron {} of {}".format(ni + 1, rate_maps_pred.shape[0]))
    plt.imshow(rate_maps_pred[ni, :, :], vmin=vmin, vmax=vmax)
    # plt.imshow(rate_maps_truth[ni, :, :], vmin=vmin, vmax=vmax)
    plt.show()
