import nengo.spa as spa
import numpy as np
from spatial_semantic_pointers.utils import encode_point, encode_random
from path_utils import plot_path_predictions
import torch
import torch.nn as nn
from datasets import MazeDataset


class ValidationSet(object):

    def __init__(self, data, maze_sps, maze_indices, goal_indices, subsample=2, spatial_encoding='ssp'):
        x_axis_sp = spa.SemanticPointer(data=data['x_axis_sp'])
        y_axis_sp = spa.SemanticPointer(data=data['y_axis_sp'])

        # n_mazes by res by res
        fine_mazes = data['fine_mazes']

        # n_mazes by n_goals by res by res by 2
        solved_mazes = data['solved_mazes']

        # NOTE: this can be modified from the original dataset, so it is explicitly passed in
        # n_mazes by dim
        # maze_sps = data['maze_sps']

        # n_mazes by n_goals by 2
        goals = data['goals']

        # n_mazes by n_goals by dim
        if spatial_encoding == 'ssp':
            goal_sps = data['goal_sps']
        elif spatial_encoding == 'random':
            goal_sps = np.zeros_like(data['goal_sps'])
            for ni in range(goal_sps.shape[0]):
                for gi in range(goal_sps.shape[1]):
                    goal_sps[ni, gi, :] = encode_random(x=goals[ni, gi, 0], y=goals[ni, gi, 1], dim=goal_sps.shape[2])
        else:
            raise NotImplementedError

        self.xs = data['xs']
        self.ys = data['ys']

        # n_mazes = goals.shape[0]
        # n_goals = goals.shape[1]

        self.maze_indices = maze_indices
        self.goal_indices = goal_indices
        self.n_mazes = len(maze_indices)
        self.n_goals = len(goal_indices)

        res = fine_mazes.shape[1]
        dim = goal_sps.shape[2]
        n_samples = int(res/subsample) * int(res/subsample) * self.n_mazes * self.n_goals

        # Visualization
        viz_locs = np.zeros((n_samples, 2))
        viz_goals = np.zeros((n_samples, 2))
        viz_loc_sps = np.zeros((n_samples, dim))
        viz_goal_sps = np.zeros((n_samples, dim))
        viz_output_dirs = np.zeros((n_samples, 2))
        viz_maze_sps = np.zeros((n_samples, maze_sps.shape[1]))

        # Generate data so each batch contains a single maze and goal
        si = 0  # sample index, increments each time
        for mi in maze_indices:
            for gi in goal_indices:
                for xi in range(0, res, subsample):
                    for yi in range(0, res, subsample):
                        loc_x = self.xs[xi]
                        loc_y = self.ys[yi]

                        viz_locs[si, 0] = loc_x
                        viz_locs[si, 1] = loc_y
                        viz_goals[si, :] = goals[mi, gi, :]
                        if spatial_encoding == 'ssp':
                            viz_loc_sps[si, :] = encode_point(loc_x, loc_y, x_axis_sp, y_axis_sp).v
                        else:
                            viz_loc_sps[si, :] = encode_random(loc_x, loc_y, dim)
                        viz_goal_sps[si, :] = goal_sps[mi, gi, :]

                        viz_output_dirs[si, :] = solved_mazes[mi, gi, xi, yi, :]

                        viz_maze_sps[si, :] = maze_sps[mi]

                        si += 1

        self.batch_size = int(si / (self.n_mazes * self.n_goals))

        print("Visualization Data Generated")
        print("Total Samples: {}".format(si))
        print("Mazes: {}".format(self.n_mazes))
        print("Goals: {}".format(self.n_goals))
        print("Batch Size: {}".format(self.batch_size))
        print("Batches: {}".format(self.n_mazes * self.n_goals))

        dataset_viz = MazeDataset(
            maze_ssp=viz_maze_sps,
            loc_ssps=viz_loc_sps,
            goal_ssps=viz_goal_sps,
            locs=viz_locs,
            goals=viz_goals,
            direction_outputs=viz_output_dirs,
        )

        # Each batch will contain the samples for one maze. Must not be shuffled
        self.vizloader = torch.utils.data.DataLoader(
            dataset_viz, batch_size=self.batch_size, shuffle=False, num_workers=0,
        )

    def run_ground_truth(self, writer):

        with torch.no_grad():
            # Each maze is in one batch
            for i, data in enumerate(self.vizloader):
                print("Ground Truth Viz batch {} of {}".format(i + 1, self.n_mazes * self.n_goals))
                maze_loc_goal_ssps, directions, locs, goals = data

                fig_truth = plot_path_predictions(
                    directions=directions, coords=locs, type='colour'
                )
                writer.add_figure('v{}/ground truth'.format(i), fig_truth)

                fig_truth_quiver = plot_path_predictions(
                    directions=directions, coords=locs, dcell=self.xs[1] - self.xs[0]
                )
                writer.add_figure('v{}/ground truth quiver'.format(i), fig_truth_quiver)

    def run_validation(self, model, writer, epoch):
        criterion = nn.MSELoss()

        with torch.no_grad():
            # Each maze is in one batch
            for i, data in enumerate(self.vizloader):
                print("Viz batch {} of {}".format(i + 1, self.n_mazes * self.n_goals))
                maze_loc_goal_ssps, directions, locs, goals = data

                outputs = model(maze_loc_goal_ssps)

                loss = criterion(outputs, directions)

                fig_pred = plot_path_predictions(
                    directions=outputs, coords=locs, type='colour'
                )
                writer.add_figure('v{}/viz set predictions'.format(i), fig_pred, epoch)

                fig_pred_quiver = plot_path_predictions(
                    directions=outputs, coords=locs, dcell=self.xs[1] - self.xs[0]
                )
                writer.add_figure('v{}/viz set predictions quiver'.format(i), fig_pred_quiver, epoch)

                writer.add_scalar(tag='viz_loss/{}'.format(i), scalar_value=loss.data.item(), global_step=epoch)


def create_dataloader(data, n_samples, maze_sps, args):
    x_axis_sp = spa.SemanticPointer(data=data['x_axis_sp'])
    y_axis_sp = spa.SemanticPointer(data=data['y_axis_sp'])

    # n_mazes by size by size
    coarse_mazes = data['coarse_mazes']

    # n_mazes by res by res
    fine_mazes = data['fine_mazes']

    # n_mazes by res by res by 2
    solved_mazes = data['solved_mazes']

    # NOTE: this can be modified from the original dataset, so it is explicitly passed in
    # n_mazes by dim
    # maze_sps = data['maze_sps']

    # n_mazes by n_goals by 2
    goals = data['goals']

    n_goals = goals.shape[1]
    n_mazes = fine_mazes.shape[0]

    # n_mazes by n_goals by dim
    if args.spatial_encoding == 'ssp':
        goal_sps = data['goal_sps']
    elif args.spatial_encoding == 'random':
        goal_sps = np.zeros((n_mazes, n_goals, args.dim))
        for ni in range(n_mazes):
            for gi in range(n_goals):
                goal_sps[ni, gi, :] = encode_random(x=goals[ni, gi, 0], y=goals[ni, gi, 1], dim=args.dim)
    else:
        raise NotImplementedError

    if 'xs' in data.keys():
        xs = data['xs']
        ys = data['ys']
    else:
        # backwards compatibility
        xs = np.linspace(args.limit_low, args.limit_high, args.res)
        ys = np.linspace(args.limit_low, args.limit_high, args.res)

    free_spaces = np.argwhere(fine_mazes == 0)
    n_free_spaces = free_spaces.shape[0]

    # Training
    train_locs = np.zeros((n_samples, 2))
    train_goals = np.zeros((n_samples, 2))
    train_loc_sps = np.zeros((n_samples, args.dim))
    train_goal_sps = np.zeros((n_samples, args.dim))
    train_output_dirs = np.zeros((n_samples, 2))
    train_maze_sps = np.zeros((n_samples, maze_sps.shape[1]))

    train_indices = np.random.randint(low=0, high=n_free_spaces, size=n_samples)

    for n in range(n_samples):
        # print("Sample {} of {}".format(n + 1, n_samples))

        # n_mazes by res by res
        indices = free_spaces[train_indices[n], :]
        maze_index = indices[0]
        x_index = indices[1]
        y_index = indices[2]
        goal_index = np.random.randint(low=0, high=n_goals)

        # 2D coordinate of the agent's current location
        loc_x = xs[x_index]
        loc_y = ys[y_index]

        train_locs[n, 0] = loc_x
        train_locs[n, 1] = loc_y
        train_goals[n, :] = goals[maze_index, goal_index, :]

        if args.spatial_encoding == 'ssp':
            train_loc_sps[n, :] = encode_point(loc_x, loc_y, x_axis_sp, y_axis_sp).v
        elif args.spatial_encoding == 'random':
            train_loc_sps[n, :] = encode_random(loc_x, loc_y, args.dim)
        train_goal_sps[n, :] = goal_sps[maze_index, goal_index, :]

        train_output_dirs[n, :] = solved_mazes[maze_index, goal_index, x_index, y_index, :]

        train_maze_sps[n, :] = maze_sps[maze_index]

    dataset_train = MazeDataset(
        maze_ssp=train_maze_sps,
        loc_ssps=train_loc_sps,
        goal_ssps=train_goal_sps,
        locs=train_locs,
        goals=train_goals,
        direction_outputs=train_output_dirs,
    )

    trainloader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0,
    )

    return trainloader
