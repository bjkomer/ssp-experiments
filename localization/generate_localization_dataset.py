import argparse
import numpy as np
# from arguments import add_parameters
from scipy.misc import logsumexp
from spatial_semantic_pointers.utils import make_good_unitary, encode_point

# environment simulation imports
from gridworlds.envs import GridWorldEnv, generate_obs_dict
from gridworlds.maze_generation import generate_maze
from agents import RandomTrajectoryAgent

parser = argparse.ArgumentParser('Generate trajectories with sensor data for 2D supervised path integration experiment')

parser.add_argument('--n-trajectories', type=int, default=200, help='number of distinct full trajectories per map in the training set')
parser.add_argument('--trajectory-steps', type=int, default=750, help='number of steps in each trajectory')
parser.add_argument('--seed', type=int, default=13)
# parser.add_argument('--include-softmax', action='store_true', help='compute the softmax for the saved data. Numerically less stable')
parser.add_argument('--sp-dim', type=int, default=512, help='dimensionality of semantic pointers')
parser.add_argument('--n-sensors', type=int, default=36, help='number of distance sensors around the agent')
parser.add_argument('--fov', type=float, default=360, help='field of view of distance sensors, in degrees')
parser.add_argument('--n-maps', type=int, default=10, help='number of different map layouts to use')
parser.add_argument('--map-style', type=str, default='blocks', choices=['blocks', 'maze'], help='type of map layout')
parser.add_argument('--map-size', type=int, default=10, help='height and width of the maze, in cells')
parser.add_argument('--ssp-scaling', type=float, default=1.0, help='amount to multiply coordinates by before converting to SSP')
parser.add_argument('--ssp-offset', type=float, default=0.0)
parser.add_argument('--maze-dataset', type=str, default='', help='if given, use the maze layouts from the dataset instead of random')

args = parser.parse_args()

# Number of time steps in a full trajectory
# trajectory_steps = int(args.duration / args.dt)
trajectory_steps = args.trajectory_steps

np.random.seed(args.seed)

rng = np.random.RandomState(seed=args.seed)

x_axis_sp = make_good_unitary(dim=args.sp_dim, rng=rng)
y_axis_sp = make_good_unitary(dim=args.sp_dim, rng=rng)

x_axis_vec = x_axis_sp.v
y_axis_vec = y_axis_sp.v

params = {
    'continuous': True,
    'fov': 360,
    'n_sensors': 36,
    'max_sensor_dist': 10,
    'normalize_dist_sensors': False,
    'movement_type': 'holonomic',
    'seed': 13,
    'map_style': args.map_style,
    'map_size': 10,
    'fixed_episode_length': False,  # Setting to false so location resets are not automatic
    'episode_length': trajectory_steps,
    'max_lin_vel': 5,
    'max_ang_vel': 5,
    'dt': 0.1,
    'full_map_obs': False,
    'pob': 0,
    'n_grid_cells': 0,
    'heading': 'none',
    'location': 'none',
    'goal_loc': 'none',
    'goal_vec': 'none',
    'bc_n_ring': 0,
    'hd_n_cells': 0,
    'goal_csp': False,
    'goal_distance': 0,  # 0 means completely random, goal isn't used anyway
    'agent_csp': True,
    'csp_dim': 512,
    'x_axis_vec': x_axis_sp,
    'y_axis_vec': y_axis_sp,
}

obs_dict = generate_obs_dict(params)



# # Number of state variables recorded
# # x,y,th, dx, dy, dth, place cell activations, head direction cell activations
# n_state_vars = 3 + 3 + args.n_place_cells + args.n_hd_cells
# data = np.zeros((args.n_trajectories, trajectory_steps, n_state_vars))
# print(data.shape)

# # place cell centers
# pc_centers = np.random.uniform(low=0, high=args.env_size, size=(args.n_place_cells, 2))
#
# # head direction centers
# hd_centers = np.random.uniform(low=-np.pi, high=np.pi, size=args.n_hd_cells)

positions = np.zeros((args.n_maps, args.n_trajectories, trajectory_steps, 2))
# angles = np.zeros((args.n_maps, args.n_trajectories, trajectory_steps))
# lin_vels = np.zeros((args.n_maps, args.n_trajectories, trajectory_steps))
# ang_vels = np.zeros((args.n_maps, args.n_trajectories, trajectory_steps))
dist_sensors = np.zeros((args.n_maps, args.n_trajectories, trajectory_steps, args.n_sensors))
# pc_activations = np.zeros((args.n_maps, args.n_trajectories, trajectory_steps, args.n_place_cells))
# hd_activations = np.zeros((args.n_maps, args.n_trajectories, trajectory_steps, args.n_hd_cells))

coarse_maps = np.zeros((args.n_maps, args.map_size, args.map_size))

# spatial semantic pointers for position
ssps = np.zeros((args.n_maps, args.n_trajectories, trajectory_steps, args.sp_dim))
# velocity in x and y
cartesian_vels = np.zeros((args.n_maps, args.n_trajectories, trajectory_steps, 2))


# def get_pc_activations(centers, pos, std, include_softmax=False):
#     if include_softmax:
#         num = np.zeros((centers.shape[0],))
#         for ci in range(centers.shape[0]):
#             num[ci] = np.exp(-np.linalg.norm(pos - pc_centers[ci, :]) / (2 * std ** 2))
#         denom = np.sum(num)
#         if denom == 0:
#             print("0 in denominator for pc_activation, returning 0")
#             return num * 0
#         return num / denom
#     else:
#         # num = np.zeros((centers.shape[0],))
#         # for ci in range(centers.shape[0]):
#         #     num[ci] = -np.linalg.norm(pos - pc_centers[ci, :]) / (2 * std ** 2)
#         # return num
#         logp = np.zeros((centers.shape[0],))
#         for ci in range(centers.shape[0]):
#             logp[ci] = -np.linalg.norm(pos - pc_centers[ci, :]) / (2 * std ** 2)
#         # log_posteriors = logp - np.log(np.sum(np.exp(logp)))
#         log_posteriors = logp - logsumexp(logp)
#         return log_posteriors
#
#
# def get_hd_activations(centers, ang, conc, include_softmax=False):
#     if include_softmax:
#         num = np.zeros((centers.shape[0],))
#         for hi in range(centers.shape[0]):
#             num[hi] = np.exp(conc * np.cos(ang - hd_centers[hi]))
#         denom = np.sum(num)
#         if denom == 0:
#             print("0 in denominator for hd_activation, returning 0")
#             return num * 0
#         return num / denom
#     else:
#         # num = np.zeros((centers.shape[0],))
#         # for hi in range(centers.shape[0]):
#         #     num[hi] = np.exp(conc * np.cos(ang - hd_centers[hi]))
#         # return num
#         logp = np.zeros((centers.shape[0],))
#         for hi in range(centers.shape[0]):
#             logp[hi] = -np.exp(conc * np.cos(ang - hd_centers[hi]))
#         # log_posteriors = logp - np.log(np.sum(np.exp(logp)))
#         log_posteriors = logp - logsumexp(logp)
#         return log_posteriors


def get_ssp_activation(pos):

    return encode_point(pos[0]*args.ssp_scaling, pos[1]*args.ssp_scaling, x_axis_sp, y_axis_sp).v


if args.maze_dataset:
    coarse_maps = np.load(args.maze_dataset)['coarse_mazes']

for mi in range(args.n_maps):
    print("Map {} of {}".format(mi + 1, args.n_maps))

    if args.maze_dataset == '':
        # Generate maps if not given
        coarse_maps[mi, :, :] = generate_maze(map_style=params['map_style'], side_len=params['map_size'])

    env = GridWorldEnv(
        map_array=coarse_maps[mi, :, :],
        observations=obs_dict,
        movement_type=params['movement_type'],
        max_lin_vel=params['max_lin_vel'],
        max_ang_vel=params['max_ang_vel'],
        continuous=params['continuous'],
        max_steps=params['episode_length'],
        fixed_episode_length=params['fixed_episode_length'],
        dt=params['dt'],
        screen_width=300,
        screen_height=300,
        # TODO: use these parameters appropriately, and save them with the dataset
        csp_scaling=args.ssp_scaling,  # multiply state by this value before creating a csp
        csp_offset=args.ssp_offset,  # subtract this value from state before creating a csp
    )

    agent = RandomTrajectoryAgent(obs_index_dict=env.obs_index_dict)

    obs_index_dict = env.obs_index_dict

    for n in range(args.n_trajectories):
        print("Generating Trajectory {} of {}".format(n+1, args.n_trajectories))

        obs = env.reset(goal_distance=params['goal_distance'])

        dist_sensors[mi, n, 0, :] = obs[env.obs_index_dict['dist_sensors']]
        ssps[mi, n, 0, :] = obs[env.obs_index_dict['agent_csp']]

        # TODO: should this be converted from env coordinates to SSP coordinates?
        positions[mi, n, 0, 0] = env.state[0]
        positions[mi, n, 0, 1] = env.state[1]

        # NOTE: velocities exist in between state measurements, so there will be one less velocity measurement

        # pc_activations[n, 0, :] = get_pc_activations(
        #     centers=pc_centers, pos=positions[n, 0, :], std=args.place_cell_std
        # )
        # hd_activations[n, 0, :] = get_hd_activations(
        #     centers=hd_centers, ang=angles[n, 0], conc=args.hd_concentration_param
        # )

        for s in range(1, trajectory_steps):

            action = agent.act(obs)

            cartesian_vels[mi, n, s-1, 0] = action[0]
            cartesian_vels[mi, n, s-1, 1] = action[1]

            obs, reward, done, info = env.step(action)

            dist_sensors[mi, n, s, :] = obs[env.obs_index_dict['dist_sensors']]
            ssps[mi, n, s, :] = obs[env.obs_index_dict['agent_csp']]  # TODO: make sure this observation is correct

            positions[mi, n, s, 0] = env.state[0]
            positions[mi, n, s, 1] = env.state[1]

            # # calculate PC activation
            # # formula from paper in Methods section
            # pc_activations[n, s, :] = get_pc_activations(
            #     centers=pc_centers, pos=positions[n, s, :], std=args.place_cell_std
            # )
            # # calculate HD activation
            # # formula from paper in Methods section
            # hd_activations[n, s, :] = get_hd_activations(
            #     centers=hd_centers, ang=angles[n, s], conc=args.hd_concentration_param
            # )
            #
            # ssps[n, s, :] = get_ssp_activation(pos=positions[n, s, :])

        # Purely for consistency, this action is not used
        action = agent.act(obs)

        cartesian_vels[mi, n, -1, 0] = action[0]
        cartesian_vels[mi, n, -1, 1] = action[1]

# if args.include_softmax:
#     activation_type = 'softmax'
# else:
#     activation_type = 'logits'

np.savez(
    'data/localization_trajectories_{}m_{}t_{}s_seed{}.npz'.format(
        args.n_maps,
        args.n_trajectories,
        args.trajectory_steps,
        args.seed
    ),
    positions=positions,
    # angles=angles,
    # lin_vels=lin_vels,
    # ang_vels=ang_vels,
    dist_sensors=dist_sensors,
    # pc_activations=pc_activations,
    # hd_activations=hd_activations,
    # pc_centers=pc_centers,
    # hd_centers=hd_centers,
    ssps=ssps,
    cartesian_vels=cartesian_vels,
    x_axis_vec=x_axis_vec,
    y_axis_vec=y_axis_vec,
    # ssp_scaling=args.ssp_scaling,
    # env_size=args.env_size,
    coarse_maps=coarse_maps,
    ssp_scaling=args.ssp_scaling,
    ssp_offset=args.ssp_offset,
)
