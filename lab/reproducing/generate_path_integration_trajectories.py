import argparse
import numpy as np
from arguments import add_parameters
from scipy.misc import logsumexp

parser = argparse.ArgumentParser('Generate trajectories for 2D supervised path integration experiment')

parser = add_parameters(parser)

parser.add_argument('--n-trajectories', type=int, default=200, help='number of distinct full trajectories in the training set')
parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--include-softmax', action='store_true', help='compute the softmax for the saved data. Numerically less stable')

args = parser.parse_args()

np.random.seed(args.seed)

# Number of time steps in a full trajectory
trajectory_steps = int(args.duration / args.dt)

# # Number of state variables recorded
# # x,y,th, dx, dy, dth, place cell activations, head direction cell activations
# n_state_vars = 3 + 3 + args.n_place_cells + args.n_hd_cells
# data = np.zeros((args.n_trajectories, trajectory_steps, n_state_vars))
# print(data.shape)

# place cell centers
pc_centers = np.random.uniform(low=0, high=args.env_size, size=(args.n_place_cells, 2))

# head direction centers
hd_centers = np.random.uniform(low=-np.pi, high=np.pi, size=args.n_hd_cells)

positions = np.zeros((args.n_trajectories, trajectory_steps, 2))
angles = np.zeros((args.n_trajectories, trajectory_steps))
lin_vels = np.zeros((args.n_trajectories, trajectory_steps))
ang_vels = np.zeros((args.n_trajectories, trajectory_steps))
pc_activations = np.zeros((args.n_trajectories, trajectory_steps, args.n_place_cells))
hd_activations = np.zeros((args.n_trajectories, trajectory_steps, args.n_hd_cells))


def get_pc_activations(centers, pos, std, include_softmax=False):
    if include_softmax:
        num = np.zeros((centers.shape[0],))
        for ci in range(centers.shape[0]):
            num[ci] = np.exp(-np.linalg.norm(pos - pc_centers[ci, :]) / (2 * std ** 2))
        denom = np.sum(num)
        if denom == 0:
            print("0 in denominator for pc_activation, returning 0")
            return num * 0
        return num / denom
    else:
        # num = np.zeros((centers.shape[0],))
        # for ci in range(centers.shape[0]):
        #     num[ci] = -np.linalg.norm(pos - pc_centers[ci, :]) / (2 * std ** 2)
        # return num
        logp = np.zeros((centers.shape[0],))
        for ci in range(centers.shape[0]):
            logp[ci] = -np.linalg.norm(pos - pc_centers[ci, :]) / (2 * std ** 2)
        # log_posteriors = logp - np.log(np.sum(np.exp(logp)))
        log_posteriors = logp - logsumexp(logp)
        return log_posteriors


def get_hd_activations(centers, ang, conc, include_softmax=False):
    if include_softmax:
        num = np.zeros((centers.shape[0],))
        for hi in range(centers.shape[0]):
            num[hi] = np.exp(conc * np.cos(ang - hd_centers[hi]))
        denom = np.sum(num)
        if denom == 0:
            print("0 in denominator for hd_activation, returning 0")
            return num * 0
        return num / denom
    else:
        # num = np.zeros((centers.shape[0],))
        # for hi in range(centers.shape[0]):
        #     num[hi] = np.exp(conc * np.cos(ang - hd_centers[hi]))
        # return num
        logp = np.zeros((centers.shape[0],))
        for hi in range(centers.shape[0]):
            logp[hi] = -np.exp(conc * np.cos(ang - hd_centers[hi]))
        # log_posteriors = logp - np.log(np.sum(np.exp(logp)))
        log_posteriors = logp - logsumexp(logp)
        return log_posteriors


for n in range(args.n_trajectories):
    print("Generating Trajectory {} of {}".format(n+1, args.n_trajectories))
    # choose a random starting location and heading direction within the arena
    # NOTE: assuming square arena
    positions[n, 0, :] = np.random.uniform(low=0, high=args.env_size, size=2)
    angles[n, 0] = np.random.uniform(low=-np.pi, high=np.pi)
    # precompute linear and angular velocities used
    lin_vels[n, :] = np.random.rayleigh(scale=args.lin_vel_rayleigh_scale, size=trajectory_steps)
    ang_vels[n, :] = np.random.normal(loc=0, scale=args.rot_vel_std, size=trajectory_steps) * np.pi/180

    pc_activations[n, 0, :] = get_pc_activations(
        centers=pc_centers, pos=positions[n, 0, :], std=args.place_cell_std
    )
    hd_activations[n, 0, :] = get_hd_activations(
        centers=hd_centers, ang=angles[n, 0], conc=args.hd_concentration_param
    )

    for s in range(1, trajectory_steps):
        # TODO: make sure this can handle the case of moving into a corner
        dist_wall = min(positions[n, s-1, 0], positions[n, s-1, 1], args.env_size - positions[n, s-1, 0], args.env_size - positions[n, s-1, 1])
        # Find which of the four walls is closest, and calculate the angle based on that wall

        if dist_wall == positions[n, s-1, 0]:
            angle_wall = angles[n, s-1] - np.pi
        elif dist_wall == positions[n, s-1, 1]:
            angle_wall = angles[n, s-1] + np.pi/2
        elif dist_wall == args.env_size - positions[n, s-1, 0]:
            angle_wall = angles[n, s-1]
        elif dist_wall == args.env_size - positions[n, s-1, 1]:
            angle_wall = angles[n, s-1] - np.pi/2

        if dist_wall < args.perimeter_distance and abs(angle_wall) < np.pi/2:
            # modify angular velocity to turn away from the wall
            # ang_vels[n, s] = ang_vels[n, s] + angle_wall / abs(angle_wall) * (np.pi/2-abs(angle_wall))
            ang_vels[n, s] = ang_vels[n, s] + angle_wall / abs(angle_wall) * (np.pi / 2)
            # slow down linear velocity
            lin_vels[n, s] = lin_vels[n, s] * args.perimeter_vel_reduction
            # lin_vels[n, s] = 0

        positions[n, s, 0] = positions[n, s-1, 0] + np.cos(angles[n, s-1]) * lin_vels[n, s] * args.dt
        positions[n, s, 1] = positions[n, s-1, 1] + np.sin(angles[n, s-1]) * lin_vels[n, s] * args.dt
        angles[n, s] = angles[n, s-1] + ang_vels[n, s] * args.dt
        if angles[n, s] > np.pi:
            angles[n, s] -= 2*np.pi
        elif angles[n, s] < -np.pi:
            angles[n, s] += 2*np.pi

        # calculate PC activation
        # formula from paper in Methods section
        pc_activations[n, s, :] = get_pc_activations(
            centers=pc_centers, pos=positions[n, s, :], std=args.place_cell_std
        )
        # calculate HD activation
        # formula from paper in Methods section
        hd_activations[n, s, :] = get_hd_activations(
            centers=hd_centers, ang=angles[n, s], conc=args.hd_concentration_param
        )

if args.include_softmax:
    activation_type = 'softmax'
else:
    activation_type = 'logits'

np.savez(
    'data/path_integration_trajectories_{}_{}t_{}s.npz'.format(activation_type, args.n_trajectories, int(args.duration)),
    positions=positions,
    angles=angles,
    lin_vels=lin_vels,
    ang_vels=ang_vels,
    pc_activations=pc_activations,
    hd_activations=hd_activations,
    pc_centers=pc_centers,
    hd_centers=hd_centers,
)
