import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nengo


def cyclic_1d_gaussian(xs, center, sigma):

    return np.exp(-((center - xs) ** 2) / (sigma**2)) +\
           np.exp(-((center - 360 - xs) ** 2) / (sigma**2)) +\
           np.exp(-((center + 360 - xs) ** 2) / (sigma**2))


res = 512
xs = np.linspace(0, 360, res)
center = 120
sigma = 45

# run the process for 1000 steps, sum a few together
raw_noise = np.zeros((res,))
for seed in [3, 1, 2]:
    # process = nengo.processes.WhiteSignal(1.0, high=15, seed=seed)
    process = nengo.processes.FilteredNoise(synapse=nengo.synapses.Alpha(0.05), seed=seed)
    raw_noise += process.run_steps(res)[:, 0]

# add a flipped version so the start and end are at the same point
noise = raw_noise + raw_noise[::-1] + 5


val = cyclic_1d_gaussian(xs, center, sigma)*75 + cyclic_1d_gaussian(xs, center, sigma*2)*noise*15 #+ noise*15

mx = np.max(val)
val = val / mx * 82

# polar plot
xs_rad = xs * np.pi/180.

fig = plt.figure(figsize=(6, 3), tight_layout=True)
ax = plt.subplot(121)
ax.plot(xs, val)
ax.set_xlim([0, 360])
ax.set_xticks([0, 90, 180, 270, 360])
ax.set_xlabel('Head Direction (degrees)')
ax.set_ylabel('Firing Rate (Hz)')
sns.despine()
ax2 = plt.subplot(122, projection='polar')
ax2.plot(xs_rad, val)
ax2.set_rticks([20, 40, 60])
# ax[1].plot(xs, noise)


plt.show()
