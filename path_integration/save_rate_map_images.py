import numpy as np
import sys
import matplotlib.pyplot as plt
import os

fname = sys.argv[1]

data = np.load(fname)

# get the filename
p = fname.split('/')[-1]
# remove the .npz, use as subfolder in images
folder = 'images/' + p.split('.')[0]

truth_folder = folder + '/truth'
pred_folder = folder + '/pred'

if not os.path.exists(truth_folder):
    os.makedirs(truth_folder)
if not os.path.exists(pred_folder):
    os.makedirs(pred_folder)

rate_maps_pred = data['rate_maps_pred']
rate_maps_truth = data['rate_maps_truth']

n_neurons = rate_maps_pred.shape[0]

for ni in range(n_neurons):
    print("Neuron {} of {}".format(ni + 1, n_neurons))

    fig, ax = plt.subplots()
    ax.imshow(rate_maps_pred[ni, :, :])
    fig.savefig("{}/neuron_{}".format(pred_folder, ni + 1))

    fig, ax = plt.subplots()
    ax.imshow(rate_maps_truth[ni, :, :])
    fig.savefig("{}/neuron_{}".format(truth_folder, ni + 1))
