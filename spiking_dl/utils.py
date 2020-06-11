import numpy as np
import torch
import nengo.spa as spa
import tensorflow as tf


def create_policy_train_test_sets(
        data, n_train_samples, n_test_samples,
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

    for test_set, n_samples in enumerate([n_train_samples, n_test_samples]):

        if test_set == 0:
            sample_indices = np.random.randint(low=0, high=n_train_start_split, size=n_samples)
            sample_goal_indices = np.random.randint(low=0, high=n_train_goal_split, size=n_samples)
        elif test_set == 1:
            sample_indices = np.random.randint(low=n_test_start_split, high=n_free_spaces, size=n_samples)
            sample_goal_indices = np.random.randint(low=n_test_goal_split, high=n_goals, size=n_samples)

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

            sample_locs[n, 0] = loc_x
            sample_locs[n, 1] = loc_y

            if connected_tiles and np.random.choice([0, 1]) == 1:
                # 50% chance to pick outside of the current tile if using connected tiles
                # overwrite the goal chosen with a new one in any continuous location not in this tile
                tile_len = int(np.ceil(np.sqrt(n_mazes)))
                # max_ind = data['full_maze'].shape[0]
                max_loc = xs[-1]*tile_len

                goal_maze_index = maze_index  # just an initialization to get the loop to run at least once
                while goal_maze_index == maze_index:
                    goal_loc = np.random.uniform(0, max_loc, size=(2,))
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

                sample_goal_sps[n, :] = goal_sps[maze_index, goal_index, :]

                sample_output_dirs[n, :] = solved_mazes[maze_index, goal_index, x_index, y_index, :]

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
        for gi in range(goal_sps.shape[1]):
            goal_sps[ni, gi, :] = encoding_func(
                x=goals[ni, gi, 0] + offsets[ni, 0],
                y=goals[ni, gi, 1] + offsets[ni, 1]
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
        for gi in goal_indices:
            for xi in range(0, res, subsample):
                for yi in range(0, res, subsample):
                    loc_x = xs[xi] + offsets[mi, 0]
                    loc_y = ys[yi] + offsets[mi, 1]

                    viz_locs[si, 0] = loc_x
                    viz_locs[si, 1] = loc_y
                    viz_goals[si, :] = goals[mi, gi, :] + offsets[mi, :]

                    viz_loc_sps[si, :] = encoding_func(x=loc_x, y=loc_y)

                    viz_goal_sps[si, :] = goal_sps[mi, gi, :]

                    viz_output_dirs[si, :] = solved_mazes[mi, gi, xi, yi, :]

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
