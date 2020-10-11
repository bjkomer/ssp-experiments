import matplotlib.pyplot as plt
import numpy as np
import sys

# plot_type = 'scan'
plot_type = 'final'

if len(sys.argv) > 1:
    fname = sys.argv[1]
else:
    fname = 'output/encoder_data_encoders.npz'

data = np.load(fname)

enc_item_initial = data['enc_item_initial']
enc_item_final = data['enc_item_final']
enc_loc_initial = data['enc_loc_initial']
enc_loc_final = data['enc_loc_final']
heatmap_vectors = data['heatmap_vectors']
item_locs = data['item_locs']

print(enc_item_initial.shape)
print(enc_item_final.shape)
print(enc_loc_initial.shape)
print(enc_loc_final.shape)
print(heatmap_vectors.shape)
print(item_locs.shape)

n_neurons = enc_loc_initial.shape[0]
dim = enc_loc_initial.shape[1]

# for n in range(n_neurons):
#     # sim_initial = np.tensordot(enc_item_initial[n, :], heatmap_vectors, axes=([0], [2]))
#     # sim_final = np.tensordot(enc_item_final[n, :], heatmap_vectors, axes=([0], [2]))
#     sim_initial = np.tensordot(enc_loc_initial[n, :], heatmap_vectors, axes=([0], [2]))
#     sim_final = np.tensordot(enc_loc_final[n, :], heatmap_vectors, axes=([0], [2]))
#
#     plt.figure()
#     plt.imshow(sim_initial)
#     plt.figure()
#     plt.imshow(sim_final)
#     plt.show()

if plot_type == 'final':
    # vmin=.25
    # vmax=1
    # # neuron_indices = [0, 82, 94, 98, 162, 175, 197, 205, 287, 316, 331, 374, 385]
    # # neuron_indices = [35, 111, 230, 351, 352]
    # # neuron_indices = [35, 111, 351, 352]
    # neuron_indices = [0, 162, 205, 374]
    # fig, ax = plt.subplots(2, len(neuron_indices))
    # for i, n in enumerate(neuron_indices):
    #     # sim_initial = np.tensordot(enc_item_initial[n, :], heatmap_vectors, axes=([0], [2]))
    #     # sim_final = np.tensordot(enc_item_final[n, :], heatmap_vectors, axes=([0], [2]))
    #     sim_initial = np.tensordot(enc_loc_initial[n, :]/np.linalg.norm(enc_loc_initial[n, :]), heatmap_vectors, axes=([0], [2]))
    #     sim_final = np.tensordot(enc_loc_final[n, :]/np.linalg.norm(enc_loc_final[n, :]), heatmap_vectors, axes=([0], [2]))
    #
    #     ax[0, i].imshow(sim_initial.T, vmin=vmin, vmax=vmax, origin='lower')
    #     ax[1, i].imshow(sim_final.T, vmin=vmin, vmax=vmax, origin='lower')


    vmin=.5
    vmax=1
    # non_pc_neuron_indices = [35, 111, 351, 352]
    # pc_neuron_indices = [0, 162, 205, 374]
    non_pc_neuron_indices = [35, 28, 351]
    pc_neuron_indices = [162, 205, 374]
    # fig, ax = plt.subplots(4, 4)
    neuron_indices = non_pc_neuron_indices + pc_neuron_indices
    fig, ax = plt.subplots(2, len(neuron_indices), tight_layout=True, figsize=(8, 3))
    for i, n in enumerate(neuron_indices):
        sim_initial = np.tensordot(enc_loc_initial[n, :]/np.linalg.norm(enc_loc_initial[n, :]), heatmap_vectors, axes=([0], [2]))
        sim_final = np.tensordot(enc_loc_final[n, :]/np.linalg.norm(enc_loc_final[n, :]), heatmap_vectors, axes=([0], [2]))

        ax[0, i].imshow(sim_initial.T, vmin=vmin, vmax=vmax, origin='lower')
        ax[1, i].imshow(sim_final.T, vmin=vmin, vmax=vmax, origin='lower')

        ax[0, i].set_axis_off()
        ax[1, i].set_axis_off()




    # figure of the item locations
    fig, ax = plt.subplots()
    ax.scatter(item_locs[:, 0], item_locs[:, 1])
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.set_aspect('equal')

    plt.show()

if plot_type == 'scan':
    vmin=.25#0
    vmax=1
    for k in range(40):
        neuron_indices = np.arange(10)+k*10
        fig, ax = plt.subplots(2, len(neuron_indices), figsize=(20, 10))
        for i, n in enumerate(neuron_indices):
            # sim_initial = np.tensordot(enc_item_initial[n, :], heatmap_vectors, axes=([0], [2]))
            # sim_final = np.tensordot(enc_item_final[n, :], heatmap_vectors, axes=([0], [2]))
            sim_initial = np.tensordot(enc_loc_initial[n, :]/np.linalg.norm(enc_loc_initial[n, :]), heatmap_vectors, axes=([0], [2]))
            sim_final = np.tensordot(enc_loc_final[n, :]/np.linalg.norm(enc_loc_final[n, :]), heatmap_vectors, axes=([0], [2]))

            ax[0, i].imshow(sim_initial, vmin=vmin, vmax=vmax)
            ax[1, i].imshow(sim_final, vmin=vmin, vmax=vmax)
            ax[0, i].set_title("N {}".format(n))

        plt.show()
