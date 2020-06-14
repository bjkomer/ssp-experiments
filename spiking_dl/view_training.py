import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

fname = sys.argv[1]


data = np.load(fname)

plt.plot(data['out_p_loss'])
plt.plot(data['val_out_p_loss'])


plt.show()
