import argparse
import numpy as np
from arguments import add_parameters

parser = argparse.ArgumentParser('Generate trajectories for 2D supervised path integration experiment. This script generates lighter data without any encoding mechanism')

parser = add_parameters(parser)

parser.add_argument('--n-trajectories', type=int, default=1000, help='number of distinct full trajectories in the training set')
parser.add_argument('--seed', type=int, default=13)

args = parser.parse_args()

np.random.seed(args.seed)

# Number of time steps in a full trajectory (default is 15 / 0.02 = 750)
trajectory_steps = int(args.duration / args.dt)

positions = np.zeros((args.n_trajectories, trajectory_steps, 2))
angles = np.zeros((args.n_trajectories, trajectory_steps))
lin_vels = np.zeros((args.n_trajectories, trajectory_steps))
ang_vels = np.zeros((args.n_trajectories, trajectory_steps))

# velocity in x and y
cartesian_vels = np.zeros((args.n_trajectories, trajectory_steps, 2))

rng = np.random.RandomState(seed=args.seed)


for n in range(args.n_trajectories):
    print('\x1b[2K\r Generating Trajectory {} of {}'.format(n + 1, args.n_trajectories), end="\r")
    # choose a random starting location and heading direction within the arena
    # NOTE: assuming square arena
    positions[n, 0, :] = np.random.uniform(low=0, high=args.env_size, size=2)
    angles[n, 0] = np.random.uniform(low=-np.pi, high=np.pi)
    # precompute linear and angular velocities used
    lin_vels[n, :] = np.random.rayleigh(scale=args.lin_vel_rayleigh_scale, size=trajectory_steps)
    ang_vels[n, :] = np.random.normal(loc=0, scale=args.rot_vel_std, size=trajectory_steps) * np.pi/180


    for s in range(1, trajectory_steps):
        # # TODO: make sure this can handle the case of moving into a corner
        # dist_wall = min(positions[n, s-1, 0], positions[n, s-1, 1], args.env_size - positions[n, s-1, 0], args.env_size - positions[n, s-1, 1])
        # # Find which of the four walls is closest, and calculate the angle based on that wall
        #
        # if dist_wall == positions[n, s-1, 0]:
        #     angle_wall = angles[n, s-1] - np.pi
        # elif dist_wall == positions[n, s-1, 1]:
        #     angle_wall = angles[n, s-1] + np.pi/2
        # elif dist_wall == args.env_size - positions[n, s-1, 0]:
        #     angle_wall = angles[n, s-1]
        # elif dist_wall == args.env_size - positions[n, s-1, 1]:
        #     angle_wall = angles[n, s-1] - np.pi/2
        #
        # if dist_wall < args.perimeter_distance and abs(angle_wall) < np.pi/2:
        #     # modify angular velocity to turn away from the wall
        #     # ang_vels[n, s] = ang_vels[n, s] + angle_wall / abs(angle_wall) * (np.pi/2-abs(angle_wall))
        #     ang_vels[n, s] = ang_vels[n, s] + angle_wall / abs(angle_wall) * (np.pi / 2)
        #     # slow down linear velocity
        #     lin_vels[n, s] = lin_vels[n, s] * args.perimeter_vel_reduction
        #     # lin_vels[n, s] = 0

        # TODO: make sure this can handle the case of moving into a corner
        dist_wall_x = min(positions[n, s - 1, 0], args.env_size - positions[n, s - 1, 0])
        dist_wall_y = min(positions[n, s - 1, 1], args.env_size - positions[n, s - 1, 1])
        # Find which of the four walls is closest, and calculate the angle based on that wall

        if dist_wall_x == positions[n, s - 1, 0]:
            angle_wall_x = angles[n, s - 1] - np.pi
        elif dist_wall_x == args.env_size - positions[n, s - 1, 0]:
            angle_wall_x = angles[n, s - 1]

        if dist_wall_y == positions[n, s - 1, 1]:
            angle_wall_y = angles[n, s - 1] + np.pi/2
        elif dist_wall_y == args.env_size - positions[n, s - 1, 1]:
            angle_wall_y = angles[n, s - 1] - np.pi/2


        if angle_wall_x > np.pi:
            angle_wall_x -= 2*np.pi
        elif angle_wall_x < -np.pi:
            angle_wall_x += 2*np.pi

        if angle_wall_y > np.pi:
            angle_wall_y -= 2*np.pi
        elif angle_wall_y < -np.pi:
            angle_wall_y += 2*np.pi


        if dist_wall_x < args.perimeter_distance and abs(angle_wall_x) < np.pi/2:
            # modify angular velocity to turn away from the wall
            # ang_vels[n, s] = ang_vels[n, s] + angle_wall / abs(angle_wall) * (np.pi/2-abs(angle_wall))
            ang_vels[n, s] = ang_vels[n, s] + angle_wall_x / abs(angle_wall_x) * (np.pi / 2)
            # slow down linear velocity
            lin_vels[n, s] = lin_vels[n, s] * args.perimeter_vel_reduction
            # lin_vels[n, s] = 0

        if dist_wall_y < args.perimeter_distance and abs(angle_wall_y) < np.pi/2:
            # modify angular velocity to turn away from the wall
            # ang_vels[n, s] = ang_vels[n, s] + angle_wall / abs(angle_wall) * (np.pi/2-abs(angle_wall))
            ang_vels[n, s] = ang_vels[n, s] + angle_wall_y / abs(angle_wall_y) * (np.pi / 2)
            # slow down linear velocity
            lin_vels[n, s] = lin_vels[n, s] * args.perimeter_vel_reduction
            # lin_vels[n, s] = 0

        cartesian_vels[n, s, 0] = np.cos(angles[n, s-1]) * lin_vels[n, s]
        cartesian_vels[n, s, 1] = np.sin(angles[n, s-1]) * lin_vels[n, s]
        positions[n, s, 0] = positions[n, s-1, 0] + cartesian_vels[n, s, 0] * args.dt
        positions[n, s, 1] = positions[n, s-1, 1] + cartesian_vels[n, s, 1] * args.dt
        angles[n, s] = angles[n, s-1] + ang_vels[n, s] * args.dt
        if angles[n, s] > np.pi:
            angles[n, s] -= 2*np.pi
        elif angles[n, s] < -np.pi:
            angles[n, s] += 2*np.pi

fname = 'data/path_integration_raw_trajectories_{}t_{}s_seed{}.npz'.format(
    args.n_trajectories, int(args.duration), args.seed
)

np.savez(
    fname,
    positions=positions,
    angles=angles,
    lin_vels=lin_vels,
    ang_vels=ang_vels,
    cartesian_vels=cartesian_vels,
    env_size=args.env_size,
)

print("data saved to: {}".format(fname))
