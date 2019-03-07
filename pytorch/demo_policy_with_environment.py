from gridworlds.envs import GridWorldEnv, generate_obs_dict
from gridworlds.constants import possible_objects
from collections import OrderedDict
import numpy as np
import nengo.spa as spa
import argparse


def make_multigoal_ssp_env(map_array, csp_scaling, csp_offset, object_locations, x_axis_vec, y_axis_vec, dim=512, continuous=True):

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('View a policy running on an enviromnent')

    parser.add_argument('--dataset', type=str, default='maze_datasets/maze_dataset_10mazes_25goals_64res_13seed.npz')
    parser.add_argument('--model', type=str, default='', help='Saved model to load from')

    args = parser.parse_args()

    # Load a dataset to get the mazes/goals and axes the policy was trained on
    data = np.load(args.dataset)

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
    for gi in range(min(10, n_goals)):

        object_locations[possible_objects[gi]] = (goals[0, gi, :] / csp_scaling) + csp_offset

    # # debugging
    # import matplotlib.pyplot as plt
    # print(coarse_mazes[0, :, :])
    # print(fine_mazes[0, :, :])
    # print(solved_mazes[0, 0, :, :])
    # print(np.max(coarse_mazes))
    # print(np.max(fine_mazes))
    # print(np.max(solved_mazes))
    # # plt.imshow(coarse_mazes[0, :, :])
    # plt.imshow(solved_mazes[0, 0, :, :, 0])
    # plt.show()

    env = make_multigoal_ssp_env(
        map_array=coarse_mazes[0, :, :],
        csp_scaling=csp_scaling,
        csp_offset=csp_offset,
        object_locations=object_locations,
        x_axis_vec=x_axis_vec,
        y_axis_vec=y_axis_vec,
        dim=dim,
    )
