import matplotlib.pyplot as plt
import numpy as np
import sys

fname = sys.argv[1]

data = np.load(fname)

true_phis = data['true_phis']
learned_phis = data['learned_phis']
losses = data['losses']
val_losses = data['val_losses']

plt.plot(losses.mean(axis=0))
plt.plot(val_losses.mean(axis=0))

plt.show()
