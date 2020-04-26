import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from tensorflow.python.summary.summary_iterator import summary_iterator

if len(sys.argv) > 1:
    summary_path = sys.argv[1]
else:
    summary_path = '/home/ctnuser/metric-representation/metric_representation/pytorch/ssp_cleanup_cosine_15_items/May07_15-21-15/events.out.tfevents.1557256875.CTN15'

tags = [
    'avg_mse_loss',
    'avg_cosine_loss',
]

# n_epochs = 500
#
# mse_loss = np.zeros((n_epochs,))
# cosine_loss = np.zeros((n_epochs,))

mse_loss_list = []
cosine_loss_list = []

for e in summary_iterator(summary_path):
    for v in e.summary.value:
        if v.tag == 'avg_mse_loss':
            mse_loss_list.append(v.simple_value)
        elif v.tag == 'avg_cosine_loss':
            cosine_loss_list.append(v.simple_value)

mse_loss = np.array(mse_loss_list)
cosine_loss = np.array(cosine_loss_list)

fix, ax = plt.subplots(1, 2, figsize=(8.5, 4), tight_layout=True)

ax[0].plot(mse_loss)
ax[0].set_title("MSE Loss")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[1].plot(cosine_loss)
ax[1].set_title("Cosine Loss")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Loss")

sns.despine()

plt.show()
