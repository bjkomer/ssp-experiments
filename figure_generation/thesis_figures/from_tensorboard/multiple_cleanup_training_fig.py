import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from tensorflow.python.summary.summary_iterator import summary_iterator
import glob
from matplotlib.lines import Line2D


tags = [
    'avg_mse_loss',
    'avg_cosine_loss',
]

summary_paths = {
    'mse': glob.glob('ssp_cleanup_thesis/epochs250/mse/*/*/events*'),
    'cosine': glob.glob('ssp_cleanup_thesis/epochs250/cosine/*/*/events*'),
}

mse_loss = np.zeros((2, 5, 250))
cosine_loss = np.zeros((2, 5, 250))

for i, loss_type in enumerate(['mse', 'cosine']):
    for j, summary_path in enumerate(summary_paths[loss_type]):
        mse_loss_list = []
        cosine_loss_list = []
        for e in summary_iterator(summary_path):
            for v in e.summary.value:
                if v.tag == 'avg_mse_loss':
                    mse_loss_list.append(v.simple_value)
                elif v.tag == 'avg_cosine_loss':
                    cosine_loss_list.append(v.simple_value)

        mse_loss[i, j, :] = np.array(mse_loss_list)
        cosine_loss[i, j, :] = np.array(cosine_loss_list)

fix, ax = plt.subplots(1, 2, figsize=(8.5, 4), tight_layout=True)

palette = sns.color_palette("hls", 2)

legend_lines = [Line2D([0], [0], color=palette[1], lw=4),
                Line2D([0], [0], color=palette[0], lw=4)]

ax[1].plot(mse_loss[0, :, :].T, color=palette[1], label="Trained with MSE Loss")
ax[1].plot(mse_loss[1, :, :].T, color=palette[0], label="Trained with Cosine Loss")
ax[1].set_title("MSE Loss")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Loss")
ax[1].legend(legend_lines, ["Trained with MSE Loss", "Trained with Cosine Loss"])
ax[0].plot(cosine_loss[0, :, :].T, color=palette[1], label="Trained with MSE Loss")
ax[0].plot(cosine_loss[1, :, :].T, color=palette[0], label="Trained with Cosine Loss")
ax[0].set_title("Cosine Loss")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
# ax[0].legend(legend_lines, ["Trained with MSE Loss", "Trained with Cosine Loss"])


sns.despine()

plt.show()
