"""
- semantic goal to goal ssp through deconvolution of known memory and cleanup
- sensory info and velocity commands to agent ssp through LSTM/LMU
- ssp goal and agent ssp to velocity command through policy function
"""

import numpy as np
import argparse
import torch
from gridworlds.envs import GridWorldEnv, generate_obs_dict
from agent import GoalFindingAgent

# softlinked from ../pytorch/models.py
from models import FeedForward

# softlinked from ../localization/localization_training_utils.py
from localization_training_utils import LocalizationModel

parser = argparse.ArgumentParser('Demo full system on a maze task')

parser.add_argument('--cleanup-network', type=str,
                    default='networks/cleanup_network.pt',
                    help='SSP cleanup network')
parser.add_argument('--localization-network', type=str,
                    default='networks/localization_network.pt',
                    help='localization from sensors and velocity')
parser.add_argument('--policy-network', type=str,
                    default='networks/policy_network.pt',
                    help='goal navigation network')
parser.add_argument('--dataset', type=str,
                    default='../pytorch/maze_datasets/maze_dataset_maze_style_10mazes_25goals_64res_13size_13seed.npz',
                    # default='../pytorch/maze_datasets/maze_dataset_maze_style_50mazes_25goals_64res_13size_13seed_modified.npz',
                    help='dataset to get the maze layout from')
parser.add_argument('--maze-index', type=int, default=0, help='index within the dataset for the maze to use')
parser.add_argument('--seed', type=int, default=13)

args = parser.parse_args()

n_sensors = 36
n_maps = 10

# Size of map IDs. Equal to n_maps if using one-hot encoding
id_size = n_maps

params = {
    'continuous': True,
    'fov': 360,
    'n_sensors': n_sensors,
    'max_sensor_dist': 10,
    'normalize_dist_sensors': False,
    'movement_type': 'holonomic',
    'seed': 13,
    # 'map_style': args.map_style,
    'map_size': 10,
    'fixed_episode_length': False,  # Setting to false so location resets are not automatic
    'episode_length': 200, #1000,
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
    'csp_dim': 0,
    'goal_csp': False,
    'agent_csp': False,

    'goal_distance': 0,#args.goal_distance  # 0 means completely random
}

obs_dict = generate_obs_dict(params)

np.random.seed(params['seed'])

data = np.load(args.dataset)

# n_mazes by size by size
coarse_mazes = data['coarse_mazes']

map_array = coarse_mazes[args.maze_index, :, :]

env = GridWorldEnv(
    map_array=map_array,
    # object_locations=object_locations,
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
)

ssp_dim = 512

cleanup_network = FeedForward(input_size=ssp_dim, hidden_size=512, output_size=ssp_dim)
cleanup_network.load_state_dict(torch.load(args.cleanup_network), strict=False)

# Input is x and y velocity plus the distance sensor measurements, plus map ID
localization_network = LocalizationModel(
    input_size=2 + n_sensors + n_maps,
    unroll_length=1, #rollout_length,
    sp_dim=ssp_dim
)
localization_network.load_state_dict(torch.load(args.localization_network), strict=False)

policy_network = FeedForward(input_size=id_size + ssp_dim * 2, output_size=2)
policy_network.load_state_dict(torch.load(args.policy_network), strict=False)

agent = GoalFindingAgent(
    cleanup_network=cleanup_network,
    localization_network=localization_network,
    policy_network=policy_network,
)


num_episodes = 10
time_steps = 10000#100
returns = np.zeros((num_episodes,))
for e in range(num_episodes):
    obs = env.reset(goal_distance=params['goal_distance'])
    for s in range(params['episode_length']):
        env.render()
        env._render_extras()

        # TODO: need to give the semantic goal and the maze_id to the agent
        action = agent.act(obs)

        obs, reward, done, info = env.step(action)
        # print(obs)
        returns[e] += reward
        # if reward != 0:
        #    print(reward)
        # time.sleep(dt)
        # ignoring done flag and not using fixed episodes, which effectively means there is no goal
        # if done:
        #     break

print(returns)
