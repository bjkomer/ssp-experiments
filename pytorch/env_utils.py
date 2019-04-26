from gridworlds.envs import GridWorldEnv, generate_obs_dict
from gridworlds.constants import possible_objects
from collections import OrderedDict
import nengo.spa as spa
import numpy as np
import gym.spaces
from gym.core import Wrapper


def make_multigoal_ssp_env(map_array, csp_scaling, csp_offset, object_locations, x_axis_vec, y_axis_vec, dim=512,
                           continuous=True):
    params = {
        'x_axis_vec': spa.SemanticPointer(data=x_axis_vec),
        'y_axis_vec': spa.SemanticPointer(data=y_axis_vec),
        'goal_csp': True,
        'agent_csp': True,
        'csp_dim': dim,
        'goal_csp_egocentric': False,

        # other arguments for bio sensors
        "full_map_obs": False,
        "pob": 0,
        "max_sensor_dist": 10,
        "n_sensors": 10,
        "fov": 180,
        "normalize_dist_sensors": True,
        "n_grid_cells": 0,
        "heading": "none",
        "location": "none",
        "goal_loc": "none",
        "bc_n_ring": 12,
        "bc_n_rad": 3,
        "bc_dist_rad": 0.75,
        "bc_receptive_field_min": 1,
        "bc_receptive_field_max": 1.5,
        "hd_n_cells": 8,
        "hd_receptive_field_min": 0.78539816339,
        "hd_receptive_field_max": 0.78539816339,
        "goal_vec": "normalized",
    }
    obs_dict = generate_obs_dict(params)

    env = GridWorldEnv(
        map_array=map_array,
        object_locations=object_locations,
        observations=obs_dict,
        movement_type='holonomic',
        max_lin_vel=5,
        max_ang_vel=5,
        continuous=continuous,
        max_steps=1000,
        fixed_episode_length=False,  # True,
        dt=0.1,
        screen_width=300,
        screen_height=300,
        csp_scaling=csp_scaling,
        csp_offset=csp_offset,
    )

    return env


# TODO: format this an an actual wrapper env
class WrappedSSPEnv(Wrapper):

    def __init__(self, data, map_index, max_n_goals=10, map_encoding='ssp'):
        # n_mazes by size by size
        coarse_mazes = data['coarse_mazes']

        # n_mazes by res by res
        fine_mazes = data['fine_mazes']

        # n_mazes by res by res by 2
        solved_mazes = data['solved_mazes']

        # n_mazes by dim
        maze_sps = data['maze_sps']

        # n_mazes by n_goals by dim
        goal_sps = data['goal_sps']

        # n_mazes by n_goals by 2
        goals = data['goals']

        n_goals = goals.shape[1]
        n_mazes = fine_mazes.shape[0]

        x_axis_vec = data['x_axis_sp']
        y_axis_vec = data['y_axis_sp']
        dim = x_axis_vec.shape[0]

        xs = data['xs']

        csp_offset = coarse_mazes.shape[1] / 2.
        csp_scaling = xs[-1] / (coarse_mazes.shape[1] / 2.)

        object_locations = OrderedDict()
        # NOTE: currently only 13 possible objects
        # Only using up to 10 goals
        for gi in range(min(max_n_goals, n_goals)):
            object_locations[possible_objects[gi]] = (goals[map_index, gi, :] / csp_scaling) + csp_offset

        self._wrapped_env = make_multigoal_ssp_env(
            map_array=coarse_mazes[map_index, :, :],
            csp_scaling=csp_scaling,
            csp_offset=csp_offset,
            object_locations=object_locations,
            x_axis_vec=x_axis_vec,
            y_axis_vec=y_axis_vec,
            dim=dim,
        )
        self.env = self._wrapped_env

        self.observation_space = gym.spaces.Box(
            low=-np.ones(dim*3),
            high=np.ones(dim*3),
        )
        self.action_space = self._wrapped_env.action_space

        if map_encoding == 'ssp':
            # This will remain constant for the current map
            self.map_ssp = maze_sps[map_index, :]
        else:
            self.map_ssp = np.zeros((n_mazes,))
            self.map_ssp[map_index] = 1

        # For compatibility with DeepRL code
        self.state_dim = dim * 3
        self.action_dim = 2
        self.name = 'ssp_env'

    def modify_obs(self, obs):

        goal_ssp = obs[self._wrapped_env.obs_index_dict['goal_csp']]
        agent_ssp = obs[self._wrapped_env.obs_index_dict['agent_csp']]

        return np.concatenate([self.map_ssp, agent_ssp, goal_ssp])

    def step(self, action):
        obs, reward, done, info = self._wrapped_env.step(action)

        return self.modify_obs(obs), reward, done, info

    def reset(self):
        obs = self._wrapped_env.reset()

        return self.modify_obs(obs)
