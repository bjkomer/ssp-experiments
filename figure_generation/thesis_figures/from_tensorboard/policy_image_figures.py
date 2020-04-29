import numpy as np
import matplotlib.pyplot as plt

# TODO: use a format loop to grab the required data
fname = '/home/ctnuser/ssp-navigation/ssp_navigation/vis_images/test.npz'

data = np.load(fname)

# trim off the thick black border on the bottom right
trim_size = 5

ground_truth_images = data['ground_truth_images'][:, :-trim_size, :-trim_size]
prediction_images = data['prediction_images'][:, :-trim_size, :-trim_size]
overlay_images = data['overlay_images'][:, :-trim_size, :-trim_size, :]
# shape is (n_images, 2), 0: rmse, 1:angle rmse
rmses = data['rmses']


n_images = prediction_images.shape[0]

fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)

ax[0, 0].imshow(ground_truth_images[0, :, :], cmap='hsv', interpolation=None)
ax[0, 0].imshow(overlay_images[0, :, :, :])
ax[0, 0].set_title('Ground Truth')
ax[0, 1].imshow(prediction_images[0, :, :], cmap='hsv', interpolation=None)
ax[0, 1].imshow(overlay_images[0, :, :, :])
ax[0, 1].set_title('RMSE = {}'.format(rmses[0, 1]))

ax[1, 0].imshow(ground_truth_images[2, :, :], cmap='hsv', interpolation=None)
ax[1, 0].imshow(overlay_images[2, :, :, :])
ax[0, 0].set_title('Ground Truth')
ax[1, 1].imshow(prediction_images[2, :, :], cmap='hsv', interpolation=None)
ax[1, 1].imshow(overlay_images[2, :, :, :])
ax[1, 1].set_title('RMSE = {}'.format(rmses[2, 1]))


for i in range(2):
    for j in range(2):
        ax[i, j].set_axis_off()

plt.show()
