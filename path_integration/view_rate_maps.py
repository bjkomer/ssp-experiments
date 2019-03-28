import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


data = np.load('output/rate_maps.npz')

rate_maps_pred=data['rate_maps_pred']
rate_maps_truth=data['rate_maps_truth']

vmin=None
vmax=None

# vmin=0
# vmax=1

for ni in range(rate_maps_pred.shape[0]):
    print("Neuron {} of {}".format(ni + 1, rate_maps_pred.shape[0]))
    corr = signal.correlate2d(
        rate_maps_pred[ni, :, :],
        rate_maps_pred[ni, :, :],
        mode='full',
        boundary='fill',
        fillvalue=0,
    )
    plt.imshow(corr)
    # plt.imshow(rate_maps_pred[ni, :, :], vmin=vmin, vmax=vmax)
    # plt.imshow(rate_maps_truth[ni, :, :], vmin=vmin, vmax=vmax)
    plt.show()
