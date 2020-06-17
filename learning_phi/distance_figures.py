import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

dim = 7#7
dim = 5
dim = 32

rng = np.random.RandomState(seed=13)

n_phis = (dim-1)//2
phis = rng.uniform(-np.pi+0.001, np.pi-0.001, size=(2, n_phis))
# xf = np.ones((n_phis + 2,), dtype='complex64')
# xf[1:-1] = np.exp(1.j * phis[0, :])
# yf = np.ones((n_phis + 2,), dtype='complex64')
# yf[1:-1] = np.exp(1.j * phis[1, :])

xf = np.ones((dim,), dtype='complex64')
xf[1:(dim + 1) // 2] = np.exp(1.j * phis[0, :])
xf[-1:dim // 2:-1] = np.conj(xf[1:(dim + 1) // 2])
yf = np.ones((dim,), dtype='complex64')
yf[1:(dim + 1) // 2] = np.exp(1.j * phis[1, :])
yf[-1:dim // 2:-1] = np.conj(yf[1:(dim + 1) // 2])


# def encode_func(pos, xf, yf):
#     return np.fft.irfft((xf**pos[0])*(yf**pos[1]), n=dim)
#
#
# def encode_func_1d(pos, xf):
#     return np.fft.irfft((xf**pos[0]), n=dim)


def encode_func(pos, xf, yf):
    return np.fft.ifft((xf**pos[0])*(yf**pos[1])).real


def encode_func_1d(pos, xf):
    return np.fft.ifft((xf**pos[0])).real


# some position to encode in each case
pos = np.array([1.0, 1.2])
pos = np.array([3.0, 4.0])
pos2 = np.array([-3.4, 1.2])
pos2 = np.array([3.4, 1.2])
pos2 = np.array([1.4, 1.2])
pos2 = np.array([1.2, 1.0])
enc_pos = encode_func(pos, xf, yf)
enc_pos2 = encode_func(pos2, xf, yf)

# Random phis, one of which will be changed
r_phis = rng.uniform(-np.pi+0.001, np.pi-0.001, size=(2, n_phis))
# index to change each time
change_index = 1
res = 128
# change_phis = np.linspace(0, 2*np.pi, res)
change_phis = np.linspace(-np.pi, 2*np.pi, res)
# change_phis = np.linspace(np.pi-0.1, np.pi+.1, res)
# distance between points for every phi value
dists = np.zeros((res,))
dists2 = np.zeros((res,))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

enc_change_poses = np.zeros((res, dim))
enc_change_poses2 = np.zeros((res, dim))

for i, phi in enumerate(change_phis):
    # r_phis[0, change_index] = phi
    # r_phis[0, change_index+1] = phi
    # rxf = np.ones((n_phis + 2,), dtype='complex64')
    # rxf[1:-1] = np.exp(1.j * r_phis[0, :])
    # ryf = np.ones((n_phis + 2,), dtype='complex64')
    # ryf[1:-1] = np.exp(1.j * r_phis[1, :])
    # enc_change_pos = encode_func(pos, rxf, ryf)

    phis[0, change_index] = phi
    # phis[0, change_index+1] = 2
    # xf = np.ones((n_phis + 2,), dtype='complex64')
    # xf[1:-1] = np.exp(1.j * phis[0, :])
    # yf = np.ones((n_phis + 2,), dtype='complex64')
    # yf[1:-1] = np.exp(1.j * phis[1, :])

    xf = np.ones((dim,), dtype='complex64')
    xf[1:(dim + 1) // 2] = np.exp(1.j * phis[0, :])
    xf[-1:dim // 2:-1] = np.conj(xf[1:(dim + 1) // 2])
    yf = np.ones((dim,), dtype='complex64')
    yf[1:(dim + 1) // 2] = np.exp(1.j * phis[1, :])
    yf[-1:dim // 2:-1] = np.conj(yf[1:(dim + 1) // 2])

    enc_change_poses[i, :] = encode_func(pos, xf, yf)
    enc_change_poses2[i, :] = encode_func(pos2, xf, yf)

    dists[i] = np.linalg.norm(enc_pos - enc_change_poses[i, :]) #+ np.linalg.norm(enc_pos2 - enc_change_pos2)
    dists2[i] = np.linalg.norm(enc_pos2 - enc_change_poses2[i, :])

max_dist = np.max(dists)
min_dist = np.min(dists)

max_dist2 = np.max(dists2)
min_dist2 = np.min(dists2)

eps = 0.001

for i, phi in enumerate(change_phis):
    print(dists[i], min_dist, max_dist)
    print((dists[i]-min_dist)/(max_dist-min_dist))
    ax.scatter(
        enc_change_poses[i, 0], enc_change_poses[i, 1], enc_change_poses[i, 2], color='green',
        alpha=(dists[i]-min_dist)/(max_dist-min_dist)
    )
    ax.scatter(
        enc_change_poses2[i, 0], enc_change_poses2[i, 1], enc_change_poses2[i, 2], color='red',
        alpha=(dists2[i]-min_dist2)/(max_dist2-min_dist2)
    )

ax.scatter(enc_pos[0], enc_pos[1], enc_pos[2], color='blue', alpha=0.75)
ax.scatter(enc_pos2[0], enc_pos2[1], enc_pos2[2], color='purple', alpha=0.75)

# # looking at just the transition point
# change_phis = np.linspace(np.pi-0.1, np.pi+.1, res)
# for i, phi in enumerate(change_phis):
#     phis[0, change_index] = phi
#     xf = np.ones((dim,), dtype='complex64')
#     xf[1:(dim + 1) // 2] = np.exp(1.j * phis[0, :])
#     xf[-1:dim // 2:-1] = np.conj(xf[1:(dim + 1) // 2])
#     yf = np.ones((dim,), dtype='complex64')
#     yf[1:(dim + 1) // 2] = np.exp(1.j * phis[1, :])
#     yf[-1:dim // 2:-1] = np.conj(yf[1:(dim + 1) // 2])
#
#     enc_change_poses[i, :] = encode_func(pos, xf, yf)
#     enc_change_poses2[i, :] = encode_func(pos2, xf, yf)
#     ax.scatter(
#         enc_change_poses[i, 0], enc_change_poses[i, 1], enc_change_poses[i, 2], color='green',
#     )
#     ax.scatter(
#         enc_change_poses2[i, 0], enc_change_poses2[i, 1], enc_change_poses2[i, 2], color='red',
#     )


fig = plt.figure()
plt.plot(change_phis, dists)
plt.plot(change_phis, dists2)


# looking at the limit of large batch size
n_samples = 1000#1000
limit = .1
positions = rng.uniform(low=-limit, high=limit, size=(n_samples, 2))

enc_poses = np.zeros((n_samples, dim))

for n in range(n_samples):
    enc_poses[n, :] = encode_func(positions[n, :], xf, yf)

enc_change_poses = np.zeros((res, n_samples, dim))
dists = np.zeros((res,))

for i, phi in enumerate(change_phis):

    phis[0, change_index+2] = phi

    xf = np.ones((dim,), dtype='complex64')
    xf[1:(dim + 1) // 2] = np.exp(1.j * phis[0, :])
    xf[-1:dim // 2:-1] = np.conj(xf[1:(dim + 1) // 2])
    yf = np.ones((dim,), dtype='complex64')
    yf[1:(dim + 1) // 2] = np.exp(1.j * phis[1, :])
    yf[-1:dim // 2:-1] = np.conj(yf[1:(dim + 1) // 2])

    for n in range(n_samples):

        enc_change_poses[i, n, :] = encode_func(positions[n, :], xf, yf)

        dists[i] += np.linalg.norm(enc_poses[n, :] - enc_change_poses[i, n, :])

fig = plt.figure()
plt.plot(change_phis, dists)
plt.title("limit: {}".format(limit))

plt.show()
