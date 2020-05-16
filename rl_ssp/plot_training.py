import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

fname = sys.argv[1]

data = np.load(fname)['eval_data']

plt.plot(data[:, 0])

plt.show()
