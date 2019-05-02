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

    def __init__(self, data, map_index, max_n_goals=10, map_encoding='ssp',
                 random_object_locations=False,
                 discretize_actions=False,
                 ):
        """
        :param data: maze dataset to get parameters from
        :param map_index: the map to use from the maze dataset
        :param max_n_goals: maximum number of goal locations to switch between
        :param map_encoding: type of map ID encoding, either ssp or one-hot
        :param random_object_locations: if true, randomly choose goal locations rather than use those in the dataset
        :param discretize_actions: if true, the environment interface will accept discrete actions and then convert them
                                   to the appropriate continuous actions in the Gridworld environment
        """
        self.discretize_actions = discretize_actions

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
        ys = data['ys']

        csp_offset = coarse_mazes.shape[1] / 2.
        csp_scaling = xs[-1] / (coarse_mazes.shape[1] / 2.)

        object_locations = OrderedDict()
        if random_object_locations:
            # use random locations rather than locations from the dataset
            for gi in range(min(max_n_goals, n_goals)):
                # Choose a random location that is not a wall
                x_ind = np.random.randint(low=0, high=len(xs))
                y_ind = np.random.randint(low=0, high=len(xs))
                while fine_mazes[map_index, x_ind, y_ind] == 1:
                    x_ind = np.random.randint(low=0, high=len(xs))
                    y_ind = np.random.randint(low=0, high=len(xs))
                object_locations[possible_objects[gi]] = (np.array([xs[x_ind], ys[y_ind]]) / csp_scaling) + csp_offset
        else:
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

        if self.discretize_actions:
            # Forward, turn left, and turn right
            self.action_space = gym.spaces.Discrete(3)
        else:
            self.action_space = self._wrapped_env.action_space

        if map_encoding == 'ssp':
            # This will remain constant for the current map
            self.map_ssp = maze_sps[map_index, :]

            # For compatibility with DeepRL code
            self.state_dim = dim * 3
        else:
            self.map_ssp = np.zeros((n_mazes,))
            self.map_ssp[map_index] = 1

            # For compatibility with DeepRL code
            self.state_dim = dim * 2 + n_mazes

        self.observation_space = gym.spaces.Box(
            low=-np.ones(self.state_dim),
            high=np.ones(self.state_dim),
        )

        # For compatibility with DeepRL code
        self.action_dim = 2
        self.name = 'ssp_env'

    def modify_obs(self, obs):

        goal_ssp = obs[self._wrapped_env.obs_index_dict['goal_csp']]
        agent_ssp = obs[self._wrapped_env.obs_index_dict['agent_csp']]

        return np.concatenate([self.map_ssp, agent_ssp, goal_ssp])

    def step(self, action):
        if self.discretize_actions:
            # TODO: allow more types of actions
            if action == 0:  # move forward
                wrapped_action = np.array([0.75, 0])
            elif action == 1:  # turn left?
                wrapped_action = np.array([0, .25])
            elif action == 2:  # turn right?
                wrapped_action = np.array([0, -.25])
            else:
                print("Warning, invalid discrete action chosen ({}). Performing no-op".format(action))
                wrapped_action = np.array([0, 0])
        else:
            wrapped_action = action
        obs, reward, done, info = self._wrapped_env.step(wrapped_action)

        return self.modify_obs(obs), reward, done, info

    def reset(self):
        obs = self._wrapped_env.reset()

        return self.modify_obs(obs)
