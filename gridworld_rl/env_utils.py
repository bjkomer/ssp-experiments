# This file contains a definition of the environment to use
from gridworlds.envs import GridWorldEnv, generate_obs_dict
from spatial_semantic_pointers.utils import make_good_unitary
import numpy as np
from gym.core import Wrapper
from collections import OrderedDict
import gym.spaces


class DiscretizedEnv(Wrapper):
    def __init__(self, seed=13, dim=512, holonomic=False):
        self.holonomic = holonomic

        # csp_offset = coarse_mazes.shape[1] / 2.
        # csp_scaling = xs[-1] / (coarse_mazes.shape[1] / 2.)

        self._wrapped_env = get_env(seed=seed, dim=dim)
        self.env = self._wrapped_env

        if self.holonomic:
            # Up, down, left, right
            self.action_space = gym.spaces.Discrete(4)
            # For compatibility with DeepRL code
            self.action_dim = 4
        else:
            # Forward, turn left, and turn right
            self.action_space = gym.spaces.Discrete(3)
            # For compatibility with DeepRL code
            self.action_dim = 3

        self.observation_space = self._wrapped_env.observation_space

        self.name = 'discretized_gridworld_ssp_env'

    def step(self, action):
        if self.holonomic:
            # TODO: allow more types of actions
            if action == 0:  # up?
                wrapped_action = np.array([0, 1.0])
            elif action == 1:  # down?
                wrapped_action = np.array([0, -1.0])
            elif action == 2:  # left?
                wrapped_action = np.array([-1.0, 0])
            elif action == 3:  # right?
                wrapped_action = np.array([1.0, 0])
            else:
                print("Warning, invalid discrete action chosen ({}). Performing no-op".format(action))
                wrapped_action = np.array([0, 0])
        else:
            # TODO: allow more types of actions
            if action == 0:  # move forward
                # wrapped_action = np.array([0.75, 0])
                wrapped_action = np.array([1.0, 0])
            elif action == 1:  # turn left?
                # wrapped_action = np.array([0, .25])
                wrapped_action = np.array([0, 1.0])
            elif action == 2:  # turn right?
                # wrapped_action = np.array([0, -.25])
                wrapped_action = np.array([0, -1.0])
            else:
                print("Warning, invalid discrete action chosen ({}). Performing no-op".format(action))
                wrapped_action = np.array([0, 0])
        obs, reward, done, info = self._wrapped_env.step(wrapped_action)

        return obs, reward, done, info

    def reset(self):
        return self._wrapped_env.reset()


def get_env(seed=13, dim=512):

    rstate = np.random.RandomState(seed=seed)
    x_axis_sp = make_good_unitary(dim=dim, rng=rstate)
    y_axis_sp = make_good_unitary(dim=dim, rng=rstate)

    map_array = np.array([
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
    ])

    # Parameters to define the environment to use
    params = {
        'x_axis_vec': x_axis_sp,
        'y_axis_vec': y_axis_sp,
        'full_map_obs': False,
        'pob': 0,
        'max_sensor_dist': 10,
        'n_sensors': 36,
        'fov': 360,
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
        'heading': 'circular',
        'location': 'none',
        'goal_loc': 'none',
        'goal_vec': 'none',
        'goal_csp': True,
        'agent_csp': True,
        'goal_csp_egocentric': True,
        'csp_dim': dim,
    }

    obs_dict = generate_obs_dict(params)

    csp_offset = map_array.shape[0] / 2
    csp_scaling = 5 / (map_array.shape[0] / 2)

    env = GridWorldEnv(
        map_array=map_array,
        # object_locations=object_locations,
        observations=obs_dict,
        continuous=True,
        movement_type='holonomic',
        dt=0.1,
        max_steps=1000,
        fixed_episode_length=False,
        max_lin_vel=5,
        max_ang_vel=5,
        screen_width=300,
        screen_height=300,
        csp_scaling=csp_offset,
        csp_offset=csp_scaling,
    )
    return env
