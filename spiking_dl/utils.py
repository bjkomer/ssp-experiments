import numpy as np
import torch
import nengo
import nengo.spa as spa
import nengo_spa
import tensorflow as tf
import os
from collections import OrderedDict
from gridworlds.envs import GridWorldEnv, generate_obs_dict
from gridworlds.constants import possible_objects
from spatial_semantic_pointers.utils import encode_point


def create_policy_train_test_sets(
        data, n_train_samples, n_test_samples,
        input_noise,
        shift_noise,
        args,
        n_mazes,
        encoding_func,
        tile_mazes=False,
        connected_tiles=False,
        split_seed=13):
    """

    :param data:
    :param n_train_samples:
    :param n_test_samples:
    :param n_mazes: number of mazes to allow training/testing on
    :param maze_sps:
    :param args:
    :param encoding_func: function for encoding 2D points into a higher dimensional space
    :param train_split: train/test split of the core data to generate from
    :param split_seed: the seed used for splitting the train and test sets
    :param pin_memory: set to True if using gpu, it will make things faster
    :return:
    """

    rng = np.random.RandomState(seed=args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    id_size = args.maze_id_dim
    if id_size > 0:
        maze_sps = np.zeros((args.n_mazes, args.maze_id_dim))
        # overwrite data
        for mi in range(args.n_mazes):
            maze_sps[mi, :] = spa.SemanticPointer(args.maze_id_dim).v
    else:
        maze_sps = None


    rng = np.random.RandomState(seed=split_seed)

    # n_mazes by res by res
    fine_mazes = data['fine_mazes'][:n_mazes, :, :]

    # n_mazes by res by res by 2
    solved_mazes = data['solved_mazes'][:n_mazes, :, :, :]

    # n_mazes by n_goals by 2
    goals = data['goals'][:n_mazes, :, :]

    n_goals = goals.shape[1]
    # n_mazes = fine_mazes.shape[0]

    # spatial offsets used when tiling mazes
    offsets = np.zeros((n_mazes, 2))

    if tile_mazes:
        length = int(np.ceil(np.sqrt(n_mazes)))
        size = data['coarse_mazes'].shape[1]
        for x in range(length):
            for y in range(length):
                ind = int(x * length + y)
                if ind >= n_mazes:
                    continue
                else:
                    offsets[int(x * length + y), 0] = x * size
                    offsets[int(x * length + y), 1] = y * size

    goal_sps = np.zeros((n_mazes, n_goals, args.dim))
    for ni in range(n_mazes):
        for gi in range(n_goals):
            goal_sps[ni, gi, :] = encoding_func(
                x=goals[ni, gi, 0] + offsets[ni, 0],
                y=goals[ni, gi, 1] + offsets[ni, 1]
            )
    xs = data['xs']
    ys = data['ys']

    free_spaces = np.argwhere(fine_mazes == 0)
    n_free_spaces = free_spaces.shape[0]

    # The first 75% of the goals can be trained on
    r_train_goal_split = .75
    n_train_goal_split = int(n_goals*r_train_goal_split)
    # The last 75% of the goals can be tested on
    r_test_goal_split = .75
    n_test_goal_split = int(n_goals * r_test_goal_split)
    # This means that 50% of the goals can appear in both

    # The first 75% of the starts can be trained on
    r_train_start_split = .75
    n_train_start_split = int(n_free_spaces * r_train_start_split)
    # The last 75% of the starts can be tested on
    r_test_start_split = .75
    n_test_start_split = int(n_free_spaces * r_test_start_split)
    # This means that 50% of the starts can appear in both

    start_indices = np.arange(n_free_spaces)
    rng.shuffle(start_indices)

    # NOTE: goal indices probably don't need to be shuffled, as they are already randomly located
    goal_indices = np.arange(n_goals)
    rng.shuffle(goal_indices)

    # extra noise on the training data
    gauss_noise_loc = rng.standard_normal((n_train_samples, args.dim)) * input_noise
    gauss_noise_goal = rng.standard_normal((n_train_samples, args.dim)) * input_noise
    offset_noise_loc = rng.uniform(low=-shift_noise, high=shift_noise, size=(n_train_samples, 2))
    offset_noise_goal = rng.uniform(low=-shift_noise, high=shift_noise, size=(n_train_samples, 2))

    for test_set, n_samples in enumerate([n_train_samples, n_test_samples]):

        if test_set == 0:
            sample_indices = rng.randint(low=0, high=n_train_start_split, size=n_samples)
            sample_goal_indices = rng.randint(low=0, high=n_train_goal_split, size=n_samples)
        elif test_set == 1:
            sample_indices = rng.randint(low=n_test_start_split, high=n_free_spaces, size=n_samples)
            sample_goal_indices = rng.randint(low=n_test_goal_split, high=n_goals, size=n_samples)

        sample_locs = np.zeros((n_samples, 2))
        sample_goals = np.zeros((n_samples, 2))
        sample_loc_sps = np.zeros((n_samples, goal_sps.shape[2]))
        sample_goal_sps = np.zeros((n_samples, goal_sps.shape[2]))
        sample_output_dirs = np.zeros((n_samples, 2))
        if maze_sps is None:
            sample_maze_sps = None
        else:
            sample_maze_sps = np.zeros((n_samples, maze_sps.shape[1]))

        for n in range(n_samples):

            # n_mazes by res by res
            indices = free_spaces[start_indices[sample_indices[n]], :]
            maze_index = indices[0]
            x_index = indices[1]
            y_index = indices[2]
            goal_index = goal_indices[sample_goal_indices[n]]

            # 2D coordinate of the agent's current location
            loc_x = xs[x_index] + offsets[maze_index, 0]
            loc_y = ys[y_index] + offsets[maze_index, 1]

            if test_set == 0:
                loc_x += offset_noise_loc[n, 0]
                loc_y += offset_noise_loc[n, 1]

            sample_locs[n, 0] = loc_x
            sample_locs[n, 1] = loc_y

            if connected_tiles and rng.choice([0, 1]) == 1:
                # 50% chance to pick outside of the current tile if using connected tiles
                # overwrite the goal chosen with a new one in any continuous location not in this tile
                tile_len = int(np.ceil(np.sqrt(n_mazes)))
                # max_ind = data['full_maze'].shape[0]
                max_loc = xs[-1]*tile_len

                goal_maze_index = maze_index  # just an initialization to get the loop to run at least once
                while goal_maze_index == maze_index:
                    goal_loc = rng.uniform(0, max_loc, size=(2,))
                    xi = int(np.floor(goal_loc[0] / xs[-1]))
                    yi = int(np.floor(goal_loc[1] / xs[-1]))
                    goal_maze_index = xi * tile_len + yi

                sample_goals[n, :] = goal_loc

                sample_loc_sps[n, :] = encoding_func(x=loc_x, y=loc_y)

                sample_goal_sps[n, :] = encoding_func(x=goal_loc[0], y=goal_loc[1])

                sample_output_dirs[n, :] = data['{}_{}'.format(maze_index, goal_maze_index)][x_index, y_index, :]
            else:
                # Regular way of doing things

                sample_goals[n, :] = goals[maze_index, goal_index, :] + offsets[maze_index, :]

                sample_loc_sps[n, :] = encoding_func(x=loc_x, y=loc_y)

                if test_set == 0:
                    sample_goals[n, :] += offset_noise_goal[n, :]
                    sample_goal_sps[n, :] = encoding_func(x=sample_goals[n, 0], y=sample_goals[n, 1])
                else:
                    sample_goal_sps[n, :] = goal_sps[maze_index, goal_index, :]

                sample_output_dirs[n, :] = solved_mazes[maze_index, goal_index, x_index, y_index, :]

                if test_set == 0:
                    sample_loc_sps[n, :] += gauss_noise_loc[n, :]
                    sample_goal_sps[n, :] += gauss_noise_goal[n, :]



            if maze_sps is not None:
                sample_maze_sps[n, :] = maze_sps[maze_index]

        if test_set == 0:
            if maze_sps is None:
                train_input = np.hstack([sample_loc_sps, sample_goal_sps])
            else:
                train_input = np.hstack([sample_maze_sps, sample_loc_sps, sample_goal_sps])
            train_output = sample_output_dirs
        elif test_set == 1:
            if maze_sps is None:
                test_input = np.hstack([sample_loc_sps, sample_goal_sps])
            else:
                test_input = np.hstack([sample_maze_sps, sample_loc_sps, sample_goal_sps])
            test_output = sample_output_dirs

    return train_input, train_output, test_input, test_output


def create_policy_vis_set(
        data,
        args,
        n_mazes,
        encoding_func,
        maze_indices,
        goal_indices,
        tile_mazes=False,
        connected_tiles=False,
        subsample=1,
        x_offset=0,  # optional offsets for checking overfitting
        y_offset=0,
        split_seed=13):

    rng = np.random.RandomState(seed=args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    id_size = args.maze_id_dim
    if id_size > 0:
        maze_sps = np.zeros((args.n_mazes, args.maze_id_dim))
        # overwrite data
        for mi in range(args.n_mazes):
            maze_sps[mi, :] = spa.SemanticPointer(args.maze_id_dim).v
    else:
        maze_sps = None


    # n_mazes by res by res
    fine_mazes = data['fine_mazes']

    # n_mazes by n_goals by res by res by 2
    solved_mazes = data['solved_mazes']

    # NOTE: this can be modified from the original dataset, so it is explicitly passed in
    # n_mazes by dim
    # maze_sps = data['maze_sps']

    # n_mazes by n_goals by 2
    goals = data['goals']

    n_mazes = goals.shape[0]
    n_goals = goals.shape[1]
    # dim = data['goal_sps'].shape[2]

    # NOTE: this code is assuming xs as ys are the same
    assert (np.all(data['xs'] == data['ys']))
    limit_low = data['xs'][0]
    limit_high = data['xs'][1]

    # # NOTE: only used for one-hot encoded location representation case
    # xso = np.linspace(limit_low, limit_high, int(np.sqrt(dim)))
    # yso = np.linspace(limit_low, limit_high, int(np.sqrt(dim)))

    xs = data['xs']
    ys = data['ys']

    maze_indices = maze_indices
    goal_indices = goal_indices
    n_mazes = len(maze_indices)
    n_goals = len(goal_indices)

    res = fine_mazes.shape[1]

    # spatial offsets used when tiling mazes
    offsets = np.zeros((n_mazes, 2))

    if tile_mazes:
        length = int(np.ceil(np.sqrt(n_mazes)))
        size = data['coarse_mazes'].shape[1]
        for x in range(length):
            for y in range(length):
                ind = int(x * length + y)
                if ind >= n_mazes:
                    continue
                else:
                    offsets[int(x * length + y), 0] = x * size
                    offsets[int(x * length + y), 1] = y * size


    goal_sps = np.zeros((n_mazes, n_goals, args.dim))
    for ni in range(goal_sps.shape[0]):
        for gi, goal_index in enumerate(goal_indices):
            goal_sps[ni, gi, :] = encoding_func(
                x=goals[ni, goal_index, 0] + offsets[ni, 0],
                y=goals[ni, goal_index, 1] + offsets[ni, 1]
            )

    n_samples = int(res / subsample) * int(res / subsample) * n_mazes * n_goals

    # Visualization
    viz_locs = np.zeros((n_samples, 2))
    viz_goals = np.zeros((n_samples, 2))
    viz_loc_sps = np.zeros((n_samples, goal_sps.shape[2]))
    viz_goal_sps = np.zeros((n_samples, goal_sps.shape[2]))
    viz_output_dirs = np.zeros((n_samples, 2))
    if maze_sps is None:
        viz_maze_sps = None
    else:
        viz_maze_sps = np.zeros((n_samples, maze_sps.shape[1]))

    # Generate data so each batch contains a single maze and goal
    si = 0  # sample index, increments each time
    for mi in maze_indices:
        for gi, goal_index in enumerate(goal_indices):
            for xi in range(0, res, subsample):
                for yi in range(0, res, subsample):
                    loc_x = xs[xi] + offsets[mi, 0]
                    loc_y = ys[yi] + offsets[mi, 1]

                    viz_locs[si, 0] = loc_x
                    viz_locs[si, 1] = loc_y
                    viz_goals[si, :] = goals[mi, gi, :] + offsets[mi, :]

                    viz_loc_sps[si, :] = encoding_func(x=loc_x + x_offset, y=loc_y + y_offset)

                    viz_goal_sps[si, :] = goal_sps[mi, gi, :]

                    viz_output_dirs[si, :] = solved_mazes[mi, goal_index, xi, yi, :]

                    if maze_sps is not None:
                        viz_maze_sps[si, :] = maze_sps[mi]

                    si += 1

    batch_size = int(si / (n_mazes * n_goals))

    print("Visualization Data Generated")
    print("Total Samples: {}".format(si))
    print("Mazes: {}".format(n_mazes))
    print("Goals: {}".format(n_goals))
    print("Batch Size: {}".format(batch_size))
    print("Batches: {}".format(n_mazes * n_goals))

    if maze_sps is None:
        viz_input = np.hstack([viz_loc_sps, viz_goal_sps])
    else:
        viz_input = np.hstack([viz_maze_sps, viz_loc_sps, viz_goal_sps])
    viz_output = viz_output_dirs

    return viz_input, viz_output, batch_size




# def compute_angular_rmse(directions_pred, directions_true):
#     """ Computes just the RMSE, without generating a figure """
#
#     angles_flat_pred = np.arctan2(directions_pred[:, 1], directions_pred[:, 0])
#     angles_flat_true = np.arctan2(directions_true[:, 1], directions_true[:, 0])
#
#     # Create 3 possible offsets to cover all cases
#     angles_offset_true = np.zeros((len(angles_flat_true), 3))
#     angles_offset_true[:, 0] = angles_flat_true - 2 * np.pi
#     angles_offset_true[:, 1] = angles_flat_true
#     angles_offset_true[:, 2] = angles_flat_true + 2 * np.pi
#
#     angles_offset_true -= angles_flat_pred.reshape(len(angles_flat_pred), 1)
#     angles_offset_true = np.abs(angles_offset_true)
#
#     angle_error = np.min(angles_offset_true, axis=1)
#
#     angle_squared_error = angle_error**2
#
#     angle_rmse = np.sqrt(angle_squared_error.mean())
#
#     return angle_rmse

def compute_angular_rmse(directions_pred, directions_true):
    """ Computes just the RMSE, with tensorflow compatible code """

    angles_flat_pred = tf.math.atan2(directions_pred[:, 1], directions_pred[:, 0])
    angles_flat_true = tf.math.atan2(directions_true[:, 1], directions_true[:, 0])

    # Create 3 possible offsets to cover all cases
    # angles_offset_true = np.zeros((len(angles_flat_true), 3))
    # angles_offset_true = tf.zeros((angles_flat_true.shape[0], 3))
    # angles_offset_true[:, 0] = angles_flat_true - 2 * np.pi
    # angles_offset_true[:, 1] = angles_flat_true
    # angles_offset_true[:, 2] = angles_flat_true + 2 * np.pi

    angles_offset_true = tf.stack(
        [
            angles_flat_true - 2 * np.pi,
            angles_flat_true,
            angles_flat_true + 2 * np.pi,
        ]
    )

    # angles_offset_true -= angles_flat_pred.reshape(angles_flat_pred.shape[0], 1)
    angles_offset_true -= angles_flat_pred

    # angles_offset_true = np.abs(angles_offset_true)
    angles_offset_true = tf.math.abs(angles_offset_true)

    # angle_error = np.min(angles_offset_true, axis=1)
    angle_error = tf.math.reduce_min(angles_offset_true, axis=1)

    angle_squared_error = angle_error**2

    # angle_rmse = np.sqrt(angle_squared_error.mean())
    angle_rmse = tf.math.sqrt(tf.math.reduce_mean(angle_squared_error))

    return angle_rmse


def create_localization_train_test_sets(
        data, n_train_samples, n_test_samples,
        args,
        n_mazes_to_use,
        encoding_func,
        tile_mazes=False,
        connected_tiles=False,
        rng=np.random,
        split_seed=13):

    np.random.seed(args.seed)
    id_size = args.maze_id_dim
    if id_size > 0:
        maze_sps = np.zeros((args.n_mazes, args.maze_id_dim))
        # overwrite data
        for mi in range(args.n_mazes):
            maze_sps[mi, :] = spa.SemanticPointer(args.maze_id_dim).v
    else:
        maze_sps = None


    # shape is (n_mazes, n_samples, n_sensors, 4)
    dist_sensors = data['dist_sensors']

    total_dataset_samples = dist_sensors.shape[1]

    # shape is (n_mazes, n_samples, 2)
    locations = data['locations']

    n_sensors = dist_sensors.shape[2]

    n_mazes = dist_sensors.shape[0]

    for test_set, n_samples in enumerate([n_train_samples, n_test_samples]):

        sensor_inputs = np.zeros((n_samples, n_sensors*4))

        encoding_outputs = np.zeros((n_samples, args.dim))

        # for the 2D encoding method
        # pos_outputs = np.zeros((n_samples, 2))

        if maze_sps is not None:
            maze_ids = np.zeros((n_samples, maze_sps.shape[1]))
        else:
            maze_ids = None

        if n_train_samples + n_test_samples < total_dataset_samples:
            if test_set:
                sample_indices = rng.randint(low=n_train_samples, high=n_train_samples+n_test_samples, size=(n_test_samples,))
            else:
                sample_indices = rng.randint(low=0, high=n_train_samples, size=(n_train_samples,))
        else:
            if test_set:
                sample_indices = rng.randint(low=0, high=total_dataset_samples, size=(n_train_samples,))
            else:
                sample_indices = rng.randint(low=0, high=total_dataset_samples, size=(n_train_samples,))

        for i in range(n_samples):
            # choose random maze and position in maze

            maze_ind = np.random.randint(low=0, high=n_mazes_to_use)

            sensor_inputs[i, :] = dist_sensors[maze_ind, sample_indices[i], :, :].flatten()

            loc = locations[maze_ind, sample_indices[i], :]
            encoding_outputs[i, :] = encoding_func(loc[0], loc[1])

            # # one-hot maze ID
            # maze_ids[i, maze_ind] = 1

            if maze_sps is not None:
                # supports both one-hot and random-sp
                maze_ids[i, :] = maze_sps[maze_ind, :]

        if test_set == 0:
            train_input = np.hstack([sensor_inputs, maze_ids])
            train_output = encoding_outputs
        elif test_set == 1:
            test_input = np.hstack([sensor_inputs, maze_ids])
            test_output = encoding_outputs

    return train_input, train_output, test_input, test_output


def create_localization_viz_set(
        args,
        encoding_func,
        n_mazes_to_use=10,
        tile_mazes=False,
        connected_tiles=False,
        rng=np.random,
        split_seed=13):

    np.random.seed(args.seed)
    id_size = args.maze_id_dim
    if id_size > 0:
        maze_sps = np.zeros((args.n_mazes, args.maze_id_dim))
        # overwrite data
        for mi in range(args.n_mazes):
            maze_sps[mi, :] = spa.SemanticPointer(args.maze_id_dim).v
    else:
        maze_sps = None

    colour_centers = np.array([
        [3, 3],
        [10, 4],
        [7, 7],
    ])

    def colour_func(x, y, sigma=7):
        ret = np.zeros((3,))

        for c in range(3):
            ret[c] = np.exp(-((colour_centers[c, 0] - x) ** 2 + (colour_centers[c, 1] - y) ** 2) / (sigma ** 2))

        return ret

    home = os.path.expanduser("~")
    dataset_file = os.path.join(
        home,
        'ssp-navigation/ssp_navigation/datasets/mixed_style_100mazes_100goals_64res_13size_13seed/maze_dataset.npz'
    )
    base_data = np.load(dataset_file)

    coarse_mazes = base_data['coarse_mazes']
    fine_mazes = base_data['fine_mazes']
    xs = base_data['xs']
    ys = base_data['ys']

    res = fine_mazes.shape[2]
    coarse_maze_size = coarse_mazes.shape[1]

    limit_low = xs[0]
    limit_high = xs[-1]

    sensor_scaling = (limit_high - limit_low) / coarse_maze_size

    # Scale to the coordinates of the coarse maze, for getting distance measurements
    xs_scaled = ((xs - limit_low) / (limit_high - limit_low)) * coarse_maze_size
    ys_scaled = ((ys - limit_low) / (limit_high - limit_low)) * coarse_maze_size

    n_sensors = args.n_sensors
    fov_rad = args.fov * np.pi / 180

    # R, G, B, distance
    dist_sensors = np.zeros((n_mazes_to_use, res*res, n_sensors * 4))

    # output SSP
    target_outputs = np.zeros((n_mazes_to_use, res*res, args.dim))

    maze_ids = np.zeros((n_mazes_to_use, res*res, args.maze_id_dim))

    # Generate sensor readings for every location in xs and ys in each maze
    for mi in range(n_mazes_to_use):
        # Print that replaces the line each time
        print('\x1b[2K\r Map {0} of {1}'.format(mi + 1, n_mazes_to_use), end="\r")

        for xi, x in enumerate(xs_scaled):
            for yi, y in enumerate(ys_scaled):

                # Only compute measurements if not in a wall
                if fine_mazes[mi, xi, yi] == 0:
                    # Compute sensor measurements and scale them based on xs and ys
                    dist_sensors[mi, xi * res + yi, :] = generate_colour_sensor_readings(
                        map_arr=coarse_mazes[mi, :, :],
                        colour_func=colour_func,
                        n_sensors=n_sensors,
                        fov_rad=fov_rad,
                        x=x,
                        y=y,
                        th=0,
                        max_sensor_dist=args.max_dist,
                    ).flatten() * sensor_scaling

                    target_outputs[mi, xi * res + yi, :] = encoding_func(x=x, y=y)

                    maze_ids[mi, xi * res + yi, :] = maze_sps[mi, :]

    viz_input = np.concatenate([dist_sensors, maze_ids], axis=2)
    viz_output = target_outputs

    return viz_input, viz_output


def generate_colour_sensor_readings(map_arr,
                                    colour_func,
                                    n_sensors=30,
                                    fov_rad=np.pi,
                                    x=0,
                                    y=0,
                                    th=0,
                                    max_sensor_dist=10,
                                    ):
    """
    Given a map, agent location in the map, number of sensors, field of view
    calculate the distance readings of each sensor to the nearest obstacle
    uses supersampling to find the approximate collision points
    """
    dists = np.zeros((n_sensors, 4))

    angs = np.linspace(-fov_rad / 2. + th, fov_rad / 2. + th, n_sensors)

    for i, ang in enumerate(angs):
        dists[i, :] = get_collision_coord(map_arr, x, y, ang, colour_func, max_sensor_dist)

    return dists


def get_collision_coord(map_array, x, y, th, colour_func,
                        max_sensor_dist=10,
                        dr=.05,
                        ):
    """
    Find the first occupied space given a start point and direction
    """
    # Define the step sizes
    dx = np.cos(th)*dr
    dy = np.sin(th)*dr

    # Initialize to starting point
    cx = x
    cy = y

    # R, G, B, dist
    ret = np.zeros((4, ))

    for i in range(int(max_sensor_dist/dr)):
        # Move one unit in the direction of the sensor
        cx += dx
        cy += dy

        if map_array[int(round(cx)), int(round(cy))] == 1:
            ret[:3] = colour_func(cx, cy)
            ret[3] = (i - 1)*dr
            return ret

    return max_sensor_dist


def free_space(pos, map_array, width, height):
    """
    Returns True if the position corresponds to a free space in the map
    :param pos: 2D floating point x-y coordinates
    """
    # TODO: doublecheck that rounding is the correct thing to do here
    x = np.clip(int(np.round(pos[0])), 0, width - 1)
    y = np.clip(int(np.round(pos[1])), 0, height - 1)
    return map_array[x, y] == 0


def generate_cleanup_dataset(
        encoding_func,
        n_samples,
        dim,
        n_items,
        item_set=None,
        allow_duplicate_items=False,
        limits=(-1, 1, -1, 1),
        seed=13,
        normalize_memory=True):
    """
    TODO: fix this description
    Create a dataset of memories that contain items bound to coordinates

    :param n_samples: number of memories to create
    :param dim: dimensionality of the memories
    :param n_items: number of items in each memory
    :param item_set: optional list of possible item vectors. If not supplied they will be generated randomly
    :param allow_duplicate_items: if an item set is given, this will allow the same item to be at multiple places
    # :param x_axis_sp: optional x_axis semantic pointer. If not supplied, will be generated as a unitary vector
    # :param y_axis_sp: optional y_axis semantic pointer. If not supplied, will be generated as a unitary vector
    :param encoding_func: function for generating the encoding
    :param limits: limits of the 2D space (x_low, x_high, y_low, y_high)
    :param seed: random seed for the memories and axis vectors if not supplied
    :param normalize_memory: if true, call normalize() on the memory semantic pointer after construction
    :return: memory, items, coords, x_axis_sp, y_axis_sp, z_axis_sp
    """
    # This seed must match the one that was used to generate the model
    np.random.seed(seed)

    # Memory containing n_items of items bound to coordinates
    memory = np.zeros((n_samples, dim))

    # SP for the item of interest
    items = np.zeros((n_samples, n_items, dim))

    # Coordinate for the item of interest
    coords = np.zeros((n_samples * n_items, 2))

    # Clean ground truth SSP
    clean_ssps = np.zeros((n_samples * n_items, dim))

    # Noisy output SSP
    noisy_ssps = np.zeros((n_samples * n_items, dim))

    for i in range(n_samples):
        memory_sp = nengo.spa.SemanticPointer(data=np.zeros((dim,)))

        # If a set of items is given, choose a subset to use now
        if item_set is not None:
            items_used = np.random.choice(item_set, size=n_items, replace=allow_duplicate_items)
        else:
            items_used = None

        for j in range(n_items):

            x = np.random.uniform(low=limits[0], high=limits[1])
            y = np.random.uniform(low=limits[2], high=limits[3])

            # pos = encode_point(x, y, x_axis_sp=x_axis_sp, y_axis_sp=y_axis_sp)
            pos = nengo.spa.SemanticPointer(data=encoding_func(x, y))

            if items_used is None:
                item = nengo.spa.SemanticPointer(dim)
            else:
                item = nengo.spa.SemanticPointer(data=items_used[j])

            items[i, j, :] = item.v
            coords[i * n_items + j, 0] = x
            coords[i * n_items + j, 1] = y
            clean_ssps[i * n_items + j, :] = pos.v
            memory_sp += (pos * item)

        if normalize_memory:
            memory_sp.normalize()

        memory[i, :] = memory_sp.v

        # Query for each item to get the noisy SSPs
        for j in range(n_items):
            noisy_ssps[i * n_items + j, :] = (memory_sp * ~nengo.spa.SemanticPointer(data=items[i, j, :])).v

    return clean_ssps, noisy_ssps, coords


class NengoGridEnv(object):

    def __init__(
            self,
            maze_sps,
            x_axis_sp,
            y_axis_sp,
            dim=256,
            n_sensors=36,
            n_goals=7,
            use_dataset_goals=False,
            sim_dt=0.01,
            nengo_dt=0.001,
            maze_index=0,
            normalize_action=True,
            noise=0.1,
            env_seed=13,
    ):

        self.dim = dim
        self.n_sensors = n_sensors
        self.sim_dt = sim_dt
        self.nengo_dt = nengo_dt
        self.dt_ratio = int(self.sim_dt / self.nengo_dt)
        self.normalize_action = normalize_action
        self.noise = noise

        # Output vector periodically updated based on sim_dt
        # last 4 dimensions are for debug (agent x,y and goal x,y)
        self.env_output = np.zeros((self.dim*2 + self.n_sensors*4 + 4))

        self.steps = 0

        home = os.path.expanduser("~")
        dataset_file = os.path.join(
            home,
            'ssp-navigation/ssp_navigation/datasets/mixed_style_100mazes_100goals_64res_13size_13seed/maze_dataset.npz'
        )
        data = np.load(dataset_file)

        xs = data['xs']
        ys = data['ys']
        # fixed random set of locations for the goals
        limit_range = xs[-1] - xs[0]

        # n_mazes by size by size
        coarse_mazes = data['coarse_mazes']
        coarse_size = coarse_mazes.shape[1]

        goals = data['goals']
        goals_scaled = ((goals - xs[0]) / limit_range) * coarse_size

        self.map_array = coarse_mazes[maze_index, :, :]
        object_locations = OrderedDict()
        self.vocab = {}
        for i in range(n_goals):
            sp_name = possible_objects[i]
            if use_dataset_goals:
                object_locations[sp_name] = goals_scaled[maze_index, i]  # using goal locations from the dataset
            else:
                # If set to None, the environment will choose a random free space on init
                object_locations[sp_name] = None
            # vocab[sp_name] = spa.SemanticPointer(ssp_dim)
            self.vocab[sp_name] = nengo_spa.SemanticPointer(data=np.random.uniform(-1, 1, size=dim)).normalized()

        colour_centers = np.array([
            [3, 3],
            [10, 4],
            [7, 7],
        ])

        def colour_func(x, y, sigma=7):
            ret = np.zeros((3,))

            for c in range(3):
                ret[c] = np.exp(-((colour_centers[c, 0] - x) ** 2 + (colour_centers[c, 1] - y) ** 2) / (sigma ** 2))

            return ret

        params = {
            'continuous': True,
            'fov': 360,
            'n_sensors': n_sensors,
            'colour_func': colour_func,
            'max_sensor_dist': 10,
            'normalize_dist_sensors': False,
            'movement_type': 'holonomic',
            'seed': env_seed,
            # 'map_style': args.map_style,
            'map_size': 10,
            'fixed_episode_length': True,
            'episode_length': 1000,  # 500, 1000,  #200,
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

            # set up rewards so minimum score is -1 and maximum score is +1 (based on 1000 steps max)
            'wall_penalty': -.001,
            'movement_cost': -.001,
            'goal_reward': 1.,

            'goal_distance': 0,  # args.goal_distance  # 0 means completely random
        }
        obs_dict = generate_obs_dict(params)

        self.env = GridWorldEnv(
            map_array=self.map_array,
            object_locations=object_locations,  # object locations explicitly chosen so a fixed SSP memory can be given
            observations=obs_dict,
            movement_type=params['movement_type'],
            max_lin_vel=params['max_lin_vel'],
            max_ang_vel=params['max_ang_vel'],
            continuous=params['continuous'],
            max_steps=params['episode_length'],
            fixed_episode_length=params['fixed_episode_length'],
            wall_penalty=params['wall_penalty'],
            movement_cost=params['movement_cost'],
            goal_reward=params['goal_reward'],
            dt=params['dt'],
            screen_width=300,
            screen_height=300,
            debug_ghost=True,
            seed=env_seed + maze_index,
        )

        # Fill the item memory with the correct SSP for remembering the goal locations
        self.item_memory = nengo_spa.SemanticPointer(data=np.zeros((dim,)))

        for i in range(n_goals):
            sp_name = possible_objects[i]
            x_env, y_env = self.env.object_locations[sp_name][[0, 1]]

            # Need to scale to SSP coordinates
            # Env is 0 to 13, SSP is -5 to 5
            x = ((x_env - 0) / coarse_size) * limit_range + xs[0]
            y = ((y_env - 0) / coarse_size) * limit_range + ys[0]

            self.item_memory += self.vocab[sp_name] * encode_point(x, y, x_axis_sp, y_axis_sp)
        # item_memory.normalize()
        self.item_memory = self.item_memory.normalized()

        # the unsqueeze is to add the batch dimension
        # map_id = torch.Tensor(maze_sps[maze_index, :]).unsqueeze(0)


        obs = self.env.reset()
        # get the cue
        goal_object_index = self.env.goal_object_index
        cue_sp = self.vocab[possible_objects[goal_object_index]]

        self.env_output[self.dim * 2:self.dim * 2 + self.n_sensors*4] = obs

        # Load in the static outputs for this environment
        self.env_output[:self.dim] = maze_sps[maze_index, :]

        # TODO: temporarily just returning the deconvolved noisy goal
        # self.env_output[self.dim:2*self.dim] = item_memory.v
        self.env_output[self.dim:2 * self.dim] = (self.item_memory *~ cue_sp).v

        self.env_output[-4] = self.env.state[0]
        self.env_output[-3] = self.env.state[1]
        self.env_output[-2] = self.env.goal_state[0]
        self.env_output[-1] = self.env.goal_state[1]

        self.th = 0
        # TODO: set scales appropriately
        self.scale_x = 1
        self.scale_y = 1
        self.build_html_string()

        self.update_html()

        self._nengo_html_ = self.base_html.format(
            self.env.state[0],
            self.env.state[1],
            self.env.goal_state[0],
            self.env.goal_state[1]
        )

    def build_html_string(self):

        self.tile_template = '<rect x={0} y={1} width=1 height=1 style="fill:black"/>'
        self.goal_template = '<circle cx="{0}" cy="{1}" r="0.5" style="fill:green"/>'
        self.agent_template = '<polygon points="0.25,0.25 -0.25,0.25 0,-0.5" style="fill:blue" transform="translate({0},{1}) rotate({2})"/>'
        self.sensor_template = '<line x1="{0}" y1="{1}" x2="{2}" y2="{3}" style="stroke:rgb(128,128,128);stroke-width:.1"/>'
        self.height = 13
        self.width = 13

        # Used to display HTML plot
        self.base_html = '''<svg width="100%" height="100%" viewbox="0 0 {0} {1}">'''.format(self.height, self.width)

        # Draw the outer rectangle
        # self.base_html += '<rect width="100" height="100" stroke-width="2.0" stroke="black" fill="white" />'

        # Draw the obstacles
        # TODO: make sure coordinates are correct (e.g. inverted y axis)
        tiles = []
        for i in range(self.height):
            for j in range(self.width):
                if self.map_array[i, j] == 1:
                    # tiles.append(self.tile_template.format(i, j))
                    tiles.append(self.tile_template.format(i, self.height - (j + 1)))
        self.base_html += ''.join(tiles)

        # # draw distance sensors
        # lines = []
        # self.sensor_dists = generate_sensor_readings(
        #     map_arr=self.map,
        #     zoom_level=8,
        #     n_sensors=self.n_sensors,
        #     fov_rad=self.fov_rad,
        #     x=x,
        #     y=y,
        #     th=th,
        #     max_sensor_dist=self.max_sensor_dist,
        # )
        # ang_interval = self.fov_rad / self.n_sensors
        # start_ang = -self.fov_rad/2. + th
        #
        # for i, dist in enumerate(self.sensor_dists):
        #     sx = dist*np.cos(start_ang + i*ang_interval) + self.x
        #     sy = dist*np.sin(start_ang + i*ang_interval) + self.y
        #     lines.append(self.sensor_template.format(self.x, self.y, sx, sy))
        # self.base_html += ''.join(lines)

        # Set up the goal to be filled in later with 'format()'
        self.base_html += '<circle {0} r=".25" stroke-width="0.01" stroke="green" fill="green" />'

        # Set up the agent to be filled in later with 'format()'
        # self.base_html += '<polygon points="{1}" stroke="black" fill="black" />'
        self.base_html += '<circle {1} r=".25" stroke-width="0.01" stroke="blue" fill="blue" />'

        # Close the svg
        self.base_html += '</svg>'

    def update_html(self, body_scale=0.5):
        # # Define points of the triangular agent based on x, y, and th
        # x1 = (self.env.state[0] + body_scale * 0.5 * np.cos(self.th - 2 * np.pi / 3)) * self.scale_x
        # y1 = 100 - (self.env.state[1] + body_scale * 0.5 * np.sin(self.th - 2 * np.pi / 3)) * self.scale_y
        #
        # x2 = (self.env.state[0] + body_scale * np.cos(self.th)) * self.scale_x
        # y2 = 100 - (self.env.state[1] + body_scale * np.sin(self.th)) * self.scale_y
        #
        # x3 = (self.env.state[0] + body_scale * 0.5 * np.cos(self.th + 2 * np.pi / 3)) * self.scale_x
        # y3 = 100 - (self.env.state[1] + body_scale * 0.5 * np.sin(self.th + 2 * np.pi / 3)) * self.scale_y

        # points = "{0},{1} {2},{3} {4},{5}".format(x1, y1, x2, y2, x3, y3)
        loc = 'cx="{0}" cy="{1}"'.format(
            self.env.state[0] * self.scale_x+.5,
            self.height - (self.env.state[1] * self.scale_y+.5)
        )
        goal = 'cx="{0}" cy="{1}"'.format(
            self.env.goal_state[0] * self.scale_x+.5,
            self.height - (self.env.goal_state[1] * self.scale_y+.5)
        )

        # Update the html plot
        self._nengo_html_ = self.base_html.format(goal, loc)

    def __call__(self, t, x):

        if self.steps % self.dt_ratio == 0:
            # influence the environment and update the return value

            action = np.array(x)
            if self.normalize_action:
                mag = np.linalg.norm(action)
                if mag > 0.001:
                    action = action / np.linalg.norm(action)

            if self.noise > 0:
                # Add small amount of noise to the action
                action += np.random.normal(size=2) * self.noise

            obs, reward, done, info = self.env.step(action)

            if done:
                obs = self.env.reset()

                # get the new cue
                goal_object_index = self.env.goal_object_index
                cue_sp = self.vocab[possible_objects[goal_object_index]]

                # TODO: temporarily just doing the deconvolving here
                self.env_output[self.dim:2 * self.dim] = (self.item_memory * ~ cue_sp).v

            self.env_output[self.dim*2:self.dim*2+self.n_sensors*4] = obs

            # Debug values
            self.env_output[-4] = self.env.state[0]
            self.env_output[-3] = self.env.state[1]
            self.env_output[-2] = self.env.goal_state[0]
            self.env_output[-1] = self.env.goal_state[1]

            self.update_html()

        self.steps += 1
        return self.env_output


class PolicyGT(object):

    def __init__(self, heatmap_vectors, maze_index=0, dim=256):
        """ground truth policy"""
        home = os.path.expanduser("~")
        dataset_file = os.path.join(
            home,
            'ssp-navigation/ssp_navigation/datasets/mixed_style_100mazes_100goals_64res_13size_13seed/maze_dataset.npz'
        )
        data = np.load(dataset_file)

        self.dim = dim

        self.xs = data['xs']
        self.ys = data['ys']

        self.heatmap_vectors = heatmap_vectors

        # fixed random set of locations for the goals
        self.limit_range = self.xs[-1] - self.xs[0]

        # n_mazes by size by size
        # coarse_mazes = data['coarse_mazes']
        # n_mazes by res by res
        self.fine_mazes = data['fine_mazes'][maze_index, :, :]

        # n_mazes by n_goals, res by res by 2
        self.solved_mazes = data['solved_mazes'][maze_index, :, :, :, :]

        # n_mazes by n_goals by 2
        self.goals = data['goals'][maze_index, :, :]

    def get_closest_goal_index(self, goal_ssp):
        vs = np.tensordot(goal_ssp, self.heatmap_vectors, axes=([0], [2]))

        xy = np.unravel_index(vs.argmax(), vs.shape)
        goal_pos = np.array(
            [[self.xs[xy[0]], self.ys[xy[1]]]]
        )
        goal_dists = np.linalg.norm(self.goals - goal_pos , axis=1)
        goal_index = np.argmin(goal_dists)

        return goal_index

    def get_closest_grid_point(self, loc_ssp):
        vs = np.tensordot(loc_ssp, self.heatmap_vectors, axes=([0], [2]))

        xy = np.unravel_index(vs.argmax(), vs.shape)

        return xy[0], xy[1]


    def __call__(self, t, v):

        loc_ssp = v[self.dim:self.dim*2]
        goal_ssp = v[self.dim*2:self.dim*3]

        gi = self.get_closest_goal_index(goal_ssp)

        xi, yi = self.get_closest_grid_point(loc_ssp)

        return self.solved_mazes[gi, xi, yi]
