import argparse
import numpy as np
from arguments import add_parameters

parser = argparse.ArgumentParser('Generate trajectories for 2D supervised path integration experiment')

parser = add_parameters(parser)

parser.add_argument('--n-trajectories', default=200, help='number of distinct full trajectories in the training set')
parser.add_argument('--seed', type=int, default=13)

args = parser.parse_args()

np.random.seed(args.seed)

# Number of time steps in a full trajectory
trajectory_steps = int(args.duration / args.dt)

# # Number of state variables recorded
# # x,y,th, dx, dy, dth, place cell activations, head direction cell activations
# n_state_vars = 3 + 3 + args.n_place_cells + args.n_hd_cells
# data = np.zeros((args.n_trajectories, trajectory_steps, n_state_vars))
# print(data.shape)

positions = np.zeros((args.n_trajectories, trajectory_steps, 2))
angles = np.zeros((args.n_trajectories, trajectory_steps))
lin_vels = np.zeros((args.n_trajectories, trajectory_steps))
ang_vels = np.zeros((args.n_trajectories, trajectory_steps))
pc_activations = np.zeros((args.n_trajectories, trajectory_steps, args.n_place_cells))
hd_activations = np.zeros((args.n_trajectories, trajectory_steps, args.n_hd_cells))

for n in range(args.n_trajectories):
    # choose a random starting location and heading direction within the arena
    # NOTE: assuming square arena
    positions[n, 0, :] = np.random.uniform(low=0, high=args.env_size, size=2)
    angles[n, 0] = np.random.uniform(low=-np.pi, high=np.pi)
    # precompute linear and angular velocities used
    lin_vels[n, :] = np.random.rayleigh(scale=args.lin_vel_rayleigh_scale, size=trajectory_steps)
    ang_vels[n, :] = np.random.normal(loc=0, scale=args.rot_vel_std, size=trajectory_steps) * np.pi/180

    for s in range(1, trajectory_steps):
        # TODO: make sure this can handle the case of moving into a corner
        dist_wall = min(positions[n, s-1, 0], positions[n, s-1, 1], args.env_size - positions[n, s-1, 0], args.env_size - positions[n, s-1, 1])
        # Find which of the four walls is closest, and calculate the angle based on that wall
        # if dist_wall == positions[n, s-1, 0]:
        #     angle_wall = angles[n, s-1]#np.pi - angles[n, s-1]
        # elif dist_wall == positions[n, s-1, 1]:
        #     angle_wall = angles[n, s-1]#np.pi/2 - angles[n, s-1]
        # elif dist_wall == args.env_size - positions[n, s-1, 0]:
        #     angle_wall = angles[n, s-1]#0 - angles[n, s-1]
        # elif dist_wall == args.env_size - positions[n, s-1, 1]:
        #     angle_wall = angles[n, s-1]#-np.pi/2 - angles[n, s-1]

        if dist_wall == positions[n, s-1, 0]:
            angle_wall = angles[n, s-1] - np.pi
        elif dist_wall == positions[n, s-1, 1]:
            angle_wall = angles[n, s-1] + np.pi/2
        elif dist_wall == args.env_size - positions[n, s-1, 0]:
            angle_wall = angles[n, s-1]
        elif dist_wall == args.env_size - positions[n, s-1, 1]:
            angle_wall = angles[n, s-1] - np.pi/2

        # Check if close to the wall and facing the wall
        # if dist_wall < args.perimeter_distance:
        #     print("close to wall detected")
        # print(angle_wall)
        # print(angles[n, s-1])
        # assert False
        # print(abs(angle_wall))
        # if abs(angle_wall) < np.pi/2:
        #     print("close angle detected")
        if dist_wall < args.perimeter_distance and abs(angle_wall) < np.pi/2:
            # print("close to wall and angle detected")
            # print("position", positions[n, s-1, :])
            # print("angle", angles[n, s-1])
            # print("angle_deg", angles[n, s - 1]*180/np.pi)
            # print("next_step", [np.cos(angles[n, s-1]), np.sin(angles[n, s-1])])
            # print("dist_wall", dist_wall)
            # print("angle_wall", angle_wall)
            # print("angle_wall_deg", angle_wall*180/np.pi)
            # print("")
            # if positions[n, s-1, 0] > args.perimeter_distance and 2.2 - positions[n, s-1, 1] > args.perimeter_distance:
            #     assert False
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

        # TODO: calculate PC activation
        # TODO: calculate HD activation

np.savez(
    'data/path_integration_trajectories_{}.npz'.format(args.n_trajectories),
    positions=positions,
    angles=angles,
    lin_vels=lin_vels,
    ang_vels=ang_vels,
    pc_activation=pc_activations,
    hd_activations=hd_activations,
)
