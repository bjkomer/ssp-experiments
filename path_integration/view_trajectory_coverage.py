import numpy as np
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='../lab/reproducing/data/path_integration_trajectories_logits_200t_15s_seed13.npz')
parser.add_argument('--bin-res', type=int, default=64)
parser.add_argument('--limit-low', type=float, default=0.0)
parser.add_argument('--limit-high', type=float, default=2.2)
parser.add_argument('--n-samples', type=int, default=1000, help='subsampling the dataset')
parser.add_argument('--rollout-length', type=int, default=100, help='trajectory length for subsampling')

args = parser.parse_args()

data = np.load(args.dataset)

positions = data['positions']

cartesian_vels = data['cartesian_vels']

xs = np.linspace(args.limit_low, args.limit_high, args.bin_res)
ys = np.linspace(args.limit_low, args.limit_high, args.bin_res)

print(positions.shape)

# coverage scatter plot
plt.figure()
plt.scatter(positions[:, :, 0], positions[:, :, 1])

# place coverage heatmap
coverage, x, y = np.histogram2d(positions[:, :, 0].flatten(), positions[:, :, 1].flatten(), [args.bin_res, args.bin_res])

plt.figure()
plt.imshow(coverage.T, origin='lower')


# sample coverage
n_trajectories = positions.shape[0]
trajectory_length = positions.shape[1]

velocity_inputs = np.zeros((args.n_samples, args.rollout_length, 2))

# for the 2D encoding method
pos_outputs = np.zeros((args.n_samples, args.rollout_length, 2))

pos_inputs = np.zeros((args.n_samples, 2))


for i in range(args.n_samples):
    # choose random trajectory
    traj_ind = np.random.randint(low=0, high=n_trajectories)
    # choose random segment of trajectory
    step_ind = np.random.randint(low=0, high=trajectory_length - args.rollout_length - 1)

    # index of final step of the trajectory
    step_ind_final = step_ind + args.rollout_length

    velocity_inputs[i, :, :] = cartesian_vels[traj_ind, step_ind:step_ind_final, :]

    # for the 2D encoding method
    pos_outputs[i, :, :] = positions[traj_ind, step_ind + 1:step_ind_final + 1, :]
    pos_inputs[i, :] = positions[traj_ind, step_ind]


# plt.figure()
# plt.scatter(pos_outputs[:, :, 0], pos_outputs[:, :, 1])

coverage, x, y = np.histogram2d(pos_outputs[:, :, 0].flatten(), pos_outputs[:, :, 1].flatten(), [args.bin_res, args.bin_res])

plt.figure()
plt.imshow(coverage.T, origin='lower')


plt.show()
