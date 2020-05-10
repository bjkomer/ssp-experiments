from gridworlds.envs import GridWorldEnv, generate_obs_dict
from spatial_semantic_pointers.utils import make_good_unitary
import numpy as np

env_sizes = {
    'miniscule': np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]),
    'tiny': np.array([
            [1, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 1],
            [1, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1],
        ]),
    'small': np.array([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 1, 0, 1],
            [1, 1, 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ]),
    'medium': np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
            [1, 1, 0, 0, 0, 0, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]),
    'large': np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]),
}


def create_env(args, goal_distance=0):

    map_array = env_sizes[args.env_size]

    base_params = {
        'full_map_obs': False,
        'pob': 0,
        'max_sensor_dist': 10,
        'n_sensors': 10,
        'fov': 180,
        'normalize_dist_sensors': True,
        'n_grid_cells': 0,
        'bc_n_ring': 0,
        'bc_n_rad': 0,
        'bc_dist_rad': 0,
        'bc_receptive_field_min': 0,
        'bc_receptive_field_max': 0,
        'hd_n_cells': 0,
        'hd_receptive_field_min': 0,
        'hd_receptive_field_max': 0,
        'csp_dim': 0,
        'goal_csp': False,
        'agent_csp': False,
        'goal_csp_egocentric': False,
        'location': 'none',
        'goal_vec': 'none',
        'goal_loc': 'none',
        'heading': 'none',
    }

    rng = np.random.RandomState(seed=args.seed)
    X = make_good_unitary(args.ssp_dim, rng=rng)
    Y = make_good_unitary(args.ssp_dim, rng=rng)

    specific_params = {
        'csp_dim': args.ssp_dim,
        'goal_csp': True,
        'agent_csp': True,
        'n_sensors': args.n_sensors,
        'fov': 360,
        'x_axis_vec':X,
        'y_axis_vec':Y,
    }

    if args.regular_coordinates:
        specific_params['location'] = 'normalized'
        specific_params['goal_loc'] = 'normalized'


    # Merge dictionaries, replacing base params with specific params
    params = {**base_params, **specific_params}

    obs_dict = generate_obs_dict(params)

    config = {
        'map_array': map_array,
        'observations': obs_dict,
        'continuous': args.continuous,
        'movement_type': args.movement_type,
        'dt': 0.1,
        'max_steps': 100,#1000,
        'wall_penalty':-.01,#-1.,
        'movement_cost':-.01,
        'goal_reward':10.,
        'goal_distance': goal_distance,
    }

    env = GridWorldEnv(**config)

    return env


def get_max_dist(env_size):
    map_array = env_sizes[env_size]
    return map_array.shape[0] - 2
