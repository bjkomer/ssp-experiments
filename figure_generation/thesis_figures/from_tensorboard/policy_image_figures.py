import numpy as np
import matplotlib.pyplot as plt

# TODO: make the RMSE label smaller in the title

# TODO: use a format loop to grab the required data
# fname = '/home/ctnuser/ssp-navigation/ssp_navigation/vis_images/test.npz'

base_folder = '/home/ctnuser/ssp-navigation/ssp_navigation/vis_images/med_dim_adam_exps/'
base_folder = '/media/ctnuser/53f2c4b3-4b3b-4768-ba69-f0a3da30c237/ctnuser/data/vis_images/med_dim_adam_exps/'

# nmazes, encoding, dim, seed
base_file_name = 'mazes{}_{}_d{}_seed{}.npz'

nmazes = [10, 25, 50]
encodings = [
    'hex-ssp',
    'ssp',
    'pc-gauss',
    'one-hot',
    'tile-coding',
    'learned',
    'random',
]

plot_names = {
    'hex-ssp': 'Hex SSP',
    'ssp': 'SSP',
    'pc-gauss': 'RBF',
    'tile-coding': 'Tile Code',
    'one-hot': 'One Hot',
    'learned': 'Learned',
    '2d': '2D',
    'random': 'Random',
    'legendre': 'Legendre',
}

seeds = [13, 14, 15, 16, 17]

n_rows = 2
n_cols = 4
fig, ax = plt.subplots(n_rows, n_cols, figsize=(8, 5), tight_layout=True)

# trim off the thick black border on the bottom right
trim_size = 5

maze_index = 0  # blocks
# maze_index = 2  # maze

nmaze = 25
dim = 256
seed = 24
for i, enc in enumerate(encodings):
    fname = base_folder + base_file_name.format(nmaze, enc, dim, seed)
    data = np.load(fname)

    ground_truth_images = data['ground_truth_images'][:, :-trim_size, :-trim_size]
    prediction_images = data['prediction_images'][:, :-trim_size, :-trim_size]
    overlay_images = data['overlay_images'][:, :-trim_size, :-trim_size, :]
    # shape is (n_images, 2), 0: rmse, 1:angle rmse
    rmses = data['rmses']

    indy = (i+1) % n_cols
    indx = (i+1) // n_cols

    ax[indx, indy].imshow(prediction_images[maze_index, :, :], cmap='hsv', interpolation=None)
    ax[indx, indy].imshow(overlay_images[maze_index, :, :, :])
    ax[indx, indy].set_title('{}\nRMSE = {:.3f}'.format(plot_names[enc], np.round(rmses[maze_index, 1], 3)))

    ax[indx, indy].set_axis_off()


ax[0, 0].imshow(ground_truth_images[maze_index, :, :], cmap='hsv', interpolation=None)
ax[0, 0].imshow(overlay_images[maze_index, :, :, :])
ax[0, 0].set_title('Ground Truth\nRMSE = 0.000')
ax[0, 0].set_axis_off()




# ground_truth_images = data['ground_truth_images'][:, :-trim_size, :-trim_size]
# prediction_images = data['prediction_images'][:, :-trim_size, :-trim_size]
# overlay_images = data['overlay_images'][:, :-trim_size, :-trim_size, :]
# # shape is (n_images, 2), 0: rmse, 1:angle rmse
# rmses = data['rmses']
#
#
# n_images = prediction_images.shape[0]
#
#
#
# ax[0, 0].imshow(ground_truth_images[0, :, :], cmap='hsv', interpolation=None)
# ax[0, 0].imshow(overlay_images[0, :, :, :])
# ax[0, 0].set_title('Ground Truth')
# ax[0, 1].imshow(prediction_images[0, :, :], cmap='hsv', interpolation=None)
# ax[0, 1].imshow(overlay_images[0, :, :, :])
# ax[0, 1].set_title('RMSE = {}'.format(rmses[0, 1]))
#
# ax[1, 0].imshow(ground_truth_images[2, :, :], cmap='hsv', interpolation=None)
# ax[1, 0].imshow(overlay_images[2, :, :, :])
# ax[0, 0].set_title('Ground Truth')
# ax[1, 1].imshow(prediction_images[2, :, :], cmap='hsv', interpolation=None)
# ax[1, 1].imshow(overlay_images[2, :, :, :])
# ax[1, 1].set_title('RMSE = {}'.format(rmses[2, 1]))
#
#
# for i in range(2):
#     for j in range(2):
#         ax[i, j].set_axis_off()

plt.show()
