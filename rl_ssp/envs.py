from gridworlds.envs import GridWorldEnv, generate_obs_dict
from spatial_semantic_pointers.utils import make_good_unitary
import numpy as np
from gym.core import Wrapper
import gym


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


def create_env(args, goal_distance=0, eval_mode=False, max_steps=1000):

    map_array = env_sizes[args.env_size]

    if args.use_open_env:
        map_array[1:-1, 1:-1] = 0

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
        specific_params['goal_csp'] = False
        specific_params['agent_csp'] = False
        specific_params['csp_dim'] = 0


    # Merge dictionaries, replacing base params with specific params
    params = {**base_params, **specific_params}

    obs_dict = generate_obs_dict(params)

    # make sure pseudorewards are not reflected in evaluations
    if eval_mode:
        pseudoreward_mag = 0
        pseudoreward_std = 1
    else:
        pseudoreward_mag = args.pseudoreward_mag
        pseudoreward_std = args.pseudoreward_std

    config = {
        'map_array': map_array,
        'observations': obs_dict,
        'continuous': args.continuous,
        'movement_type': args.movement_type,
        'dt': 0.1,
        'max_steps': max_steps, #100,#1000,
        'wall_penalty':-.02, #-.01,#-1.,
        'movement_cost':-.01,
        'goal_reward':10.,
        'goal_distance': goal_distance,
        'normalize_actions': args.continuous,
        'pseudoreward_mag': pseudoreward_mag,
        'pseudoreward_std': pseudoreward_std,
    }

    if args.discrete_actions > 0:
        env = DiscreteWrapper(env=GridWorldEnv(**config), n_actions=args.discrete_actions)
    else:
        env = GridWorldEnv(**config)

    return env


def get_max_dist(env_size):
    map_array = env_sizes[env_size]
    return map_array.shape[0] - 2


class DiscreteWrapper(Wrapper):

    def __init__(self, env, n_actions=8, mag=1):
        Wrapper.__init__(self, env)

        self._wrapped_env = env
        self.n_actions = n_actions

        # self._discrete_action_space = gym.spaces.Discrete(self.n_actions)
        self.action_space = gym.spaces.Discrete(self.n_actions)

        # evenly choose angles based on the number of actions
        angs = np.linspace(0, 2*np.pi, self.n_actions+1)[:-1]

        # look-up table for the x and y component of the action
        self.cont_x = np.cos(angs) * mag
        self.cont_y = np.sin(angs) * mag

    def _disc_to_cont(self, action):

        return np.array([self.cont_x[action], self.cont_y[action]])

    def step(self, action):

        # convert given discrete action to one the continuous environment will understand
        cont_action = self._disc_to_cont(action)

        return self._wrapped_env.step(cont_action)

    # @property
    # def action_space(self):
    #     return self._discrete_action_space

    def reset(self):

        return self._wrapped_env.reset()

    def __str__(self):
        return "Discrete-{}: {}".format(self.n_actions, self.env)
