import numpy as np
import argparse
import torch
from gridworlds.envs import GridWorldEnv, generate_obs_dict
from gridworlds.constants import possible_objects

import nengo_spa as spa
from collections import OrderedDict
from spatial_semantic_pointers.utils import encode_point, ssp_to_loc, get_heatmap_vectors

import matplotlib.pyplot as plt
import seaborn as sns

seed=13
np.random.seed(seed)
maze_index = 0

ssp_dim = 512
n_sensors = 36

dataset = '/home/ctnuser/ssp-navigation/ssp_navigation/datasets/mixed_style_100mazes_100goals_64res_13size_13seed/maze_dataset.npz'

params = {
    'continuous': True,
    'fov': 360,
    'n_sensors': n_sensors,
    'max_sensor_dist': 10,
    'normalize_dist_sensors': False,
    'movement_type': 'holonomic',
    'seed': seed,
    # 'map_style': args.map_style,
    'map_size': 10,
    'fixed_episode_length': False,  # Setting to false so location resets are not automatic
    'episode_length': 1000,  #200,
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

data = np.load(dataset)

# n_mazes by size by size
coarse_mazes = data['coarse_mazes']
coarse_size = coarse_mazes.shape[1]
n_maps = coarse_mazes.shape[0]

# Size of map IDs. Equal to n_maps if using one-hot encoding
id_size = n_maps

map_id = np.zeros((n_maps,))
map_id[maze_index] = 1
map_id = torch.Tensor(map_id).unsqueeze(0)

# n_mazes by res by res
fine_mazes = data['fine_mazes']
xs = data['xs']
ys = data['ys']
res = fine_mazes.shape[1]

coarse_xs = np.linspace(xs[0], xs[-1], coarse_size)
coarse_ys = np.linspace(ys[0], ys[-1], coarse_size)

map_array = coarse_mazes[maze_index, :, :]

x_axis_sp = spa.SemanticPointer(data=data['x_axis_sp'])
y_axis_sp = spa.SemanticPointer(data=data['y_axis_sp'])
heatmap_vectors = get_heatmap_vectors(xs, ys, x_axis_sp, y_axis_sp)
coarse_heatmap_vectors = get_heatmap_vectors(coarse_xs, coarse_ys, x_axis_sp, y_axis_sp)

# fixed random set of locations for the goals
limit_range = xs[-1] - xs[0]

goal_sps = data['goal_sps']
goals = data['goals']
# print(np.min(goals))
# print(np.max(goals))
goals_scaled = ((goals - xs[0]) / limit_range) * coarse_size
# print(np.min(goals_scaled))
# print(np.max(goals_scaled))

n_goals = 0#10  # TODO: make this a parameter
object_locations = OrderedDict()
vocab = {}
use_dataset_goals = False
for i in range(n_goals):
    sp_name = possible_objects[i]
    if use_dataset_goals:
        object_locations[sp_name] = goals_scaled[maze_index, i]  # using goal locations from the dataset
    else:
        # If set to None, the environment will choose a random free space on init
        object_locations[sp_name] = None
    # vocab[sp_name] = spa.SemanticPointer(ssp_dim)
    vocab[sp_name] = spa.SemanticPointer(data=np.random.uniform(-1, 1, size=ssp_dim)).normalized()

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
    debug_ghost=True,
)

# obs = env.reset(goal_distance=params['goal_distance'])
# env.set_agent_state(np.array([6, 9, 0]))
env.set_agent_state(np.array([3, 7, 0]))
env.step(np.array([0, 0]))
env.render()
env._render_extras()

sensors = env.get_dist_sensor_readings(
    state=env.state,
    n_sensors=params['n_sensors'],
    fov_rad=params['fov']*np.pi/180.,
    max_dist=params['max_sensor_dist'],
    normalize=params['normalize_dist_sensors'],
)

fig, ax = plt.subplots(1, 1, figsize=(3, 3), tight_layout=True)
ax.bar(np.arange(len(sensors)), sensors)
ax.set_ylabel('Distance')
ax.set_xlabel('Sensor Index')
sns.despine()
plt.show()
