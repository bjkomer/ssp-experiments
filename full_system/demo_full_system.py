"""
- semantic goal to goal ssp through deconvolution of known memory and cleanup
- sensory info and velocity commands to agent ssp through LSTM/LMU
- ssp goal and agent ssp to velocity command through policy function
"""

import numpy as np
import argparse
import torch
from gridworlds.envs import GridWorldEnv, generate_obs_dict
from gridworlds.constants import possible_objects
from agent import GoalFindingAgent
import nengo.spa as spa
from collections import OrderedDict
from spatial_semantic_pointers.utils import encode_point

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
parser.add_argument('--snapshot-localization-network', type=str,
                    default='networks/snapshot_localization_network.pt',
                    help='localization from sensors without velocity, for initialization')
parser.add_argument('--dataset', type=str,
                    default='../pytorch/maze_datasets/maze_dataset_maze_style_10mazes_25goals_64res_13size_13seed.npz',
                    # default='../pytorch/maze_datasets/maze_dataset_maze_style_50mazes_25goals_64res_13size_13seed_modified.npz',
                    help='dataset to get the maze layout from')
parser.add_argument('--maze-index', type=int, default=0, help='index within the dataset for the maze to use')
parser.add_argument('--noise', type=float, default=0.75, help='Magnitude of gaussian noise to add to the actions')
parser.add_argument('--seed', type=int, default=13)

args = parser.parse_args()

np.random.seed(args.seed)

ssp_dim = 512
n_sensors = 36
n_maps = 10

# Size of map IDs. Equal to n_maps if using one-hot encoding
id_size = n_maps

map_id = np.zeros((n_maps,))
map_id[args.maze_index] = 1
map_id = torch.Tensor(map_id).unsqueeze(0)

params = {
    'continuous': True,
    'fov': 360,
    'n_sensors': n_sensors,
    'max_sensor_dist': 10,
    'normalize_dist_sensors': False,
    'movement_type': 'holonomic',
    'seed': args.seed,
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

# n_mazes by res by res
fine_mazes = data['fine_mazes']
xs = data['xs']
ys = data['ys']
res = fine_mazes.shape[1]

print(xs)

map_array = coarse_mazes[args.maze_index, :, :]

x_axis_sp = spa.SemanticPointer(data=data['x_axis_sp'])
y_axis_sp = spa.SemanticPointer(data=data['y_axis_sp'])

# TODO: choose a random set of locations in free space to place goals
# FIXME: TEMP placeholders
semantic_goal = spa.SemanticPointer(ssp_dim)
item_memory = spa.SemanticPointer(data=np.zeros((ssp_dim,)))
n_goals = 10  # TODO: make this a parameter
object_locations = OrderedDict()
vocab = {}
for i in range(n_goals):
    sp_name = possible_objects[i]  #'GOAL{}'.format(i + 1)
    object_locations[sp_name] = None
    # Choose random indices within the fine maze
    indx = np.random.randint(low=0, high=res)
    indy = np.random.randint(low=0, high=res)
    while fine_mazes[args.maze_index, indx, indy] == 1:
        # Keep trying different locations until one is not in a wall
        indx = np.random.randint(low=0, high=res)
        indy = np.random.randint(low=0, high=res)
    x = xs[indx]
    y = ys[indy]
    object_locations[sp_name] = np.array([x, y])
    vocab[sp_name] = spa.SemanticPointer(ssp_dim)
    item_memory += vocab[sp_name] * encode_point(x, y, x_axis_sp, y_axis_sp)
item_memory.normalize()


env = GridWorldEnv(
    map_array=map_array,
    object_locations=object_locations,  # object locations explicitly chosen so a fixed SSP memory can be given
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

# for i in range(n_goals):
#     sp_name = possible_objects[i]
#     object_locations[sp_name] = np.array([x, y])
#     vocab[sp_name] = spa.SemanticPointer(ssp_dim)
#     item_memory += vocab[sp_name] * encode_point(x, y, x_axis_sp, y_axis_sp)
# item_memory.normalize()

cleanup_network = FeedForward(input_size=ssp_dim, hidden_size=512, output_size=ssp_dim)
cleanup_network.load_state_dict(torch.load(args.cleanup_network), strict=False)
cleanup_network.eval()

# Input is x and y velocity plus the distance sensor measurements, plus map ID
localization_network = LocalizationModel(
    input_size=2 + n_sensors + n_maps,
    unroll_length=1, #rollout_length,
    sp_dim=ssp_dim
)
localization_network.load_state_dict(torch.load(args.localization_network), strict=False)
localization_network.eval()

policy_network = FeedForward(input_size=id_size + ssp_dim * 2, output_size=2)
policy_network.load_state_dict(torch.load(args.policy_network), strict=False)
policy_network.eval()

snapshot_localization_network = FeedForward(
    input_size=n_sensors + n_maps,
    hidden_size=512,
    output_size=ssp_dim,
)
snapshot_localization_network.load_state_dict(torch.load(args.snapshot_localization_network), strict=False)
snapshot_localization_network.eval()

agent = GoalFindingAgent(
    cleanup_network=cleanup_network,
    localization_network=localization_network,
    policy_network=policy_network,
    snapshot_localization_network=snapshot_localization_network,
)

num_episodes = 10
time_steps = 10000#100
returns = np.zeros((num_episodes,))
for e in range(num_episodes):
    obs = env.reset(goal_distance=params['goal_distance'])

    # env.goal_object is the string name for the goal
    # env.goal_state[[0, 1]] is the 2D location for that goal
    semantic_goal = vocab[env.goal_object]

    distances = torch.Tensor(obs).unsqueeze(0)
    agent.snapshot_localize(distances, map_id)
    action = np.zeros((2,))
    for s in range(params['episode_length']):
        env.render()
        env._render_extras()

        distances = torch.Tensor(obs).unsqueeze(0)
        velocity = torch.Tensor(action).unsqueeze(0)

        # TODO: need to give the semantic goal and the maze_id to the agent
        # action = agent.act(obs)
        action = agent.act(
            distances=distances,
            velocity=velocity,  # TODO: use real velocity rather than action, because of hitting walls
            semantic_goal=semantic_goal,
            map_id=map_id,
            item_memory=item_memory,
        )

        # Add small amount of noise to the action
        action += np.random.normal(size=2) * args.noise

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
