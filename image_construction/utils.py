import os
import numpy as np
import torch
from ssp_navigation.utils.datasets import MazeDataset


def create_train_test_image_dataloaders(
        data, n_train_samples, n_test_samples,
        n_images,
        id_vecs, args, encoding_func,
        split_seed=13,
        pin_memory=False):
    """

    :param data:
    :param n_train_samples:
    :param n_test_samples:
    :param n_images: number of images to allow training/testing on
    :param id_vecs:
    :param args:
    :param encoding_func: function for encoding 2D points into a higher dimensional space
    :param split_seed: the seed used for generating the train and test sets
    :param pin_memory: set to True if using gpu, it will make things faster
    :return:
    """

    rng = np.random.RandomState(seed=split_seed)

    images = data['images'][:n_images, :, :, :]

    image_size = images.shape[1]

    limit_low = 0
    limit_high = image_size / 2

    encoding_dim = args.dim
    id_dim = args.id_dim

    for test_set, n_samples in enumerate([n_train_samples, n_test_samples]):

        sample_locs = rng.randint(low=limit_low, high=limit_high, size=(n_samples, 2))
        sample_goals = rng.randint(low=limit_low, high=limit_high, size=(n_samples, 2))
        sample_loc_sps = np.zeros((n_samples, encoding_dim))
        sample_goal_sps = np.zeros((n_samples, encoding_dim))
        sample_output_dirs = np.zeros((n_samples, 3))
        sample_id_vecs = np.zeros((n_samples, id_dim))

        image_indices = rng.randint(low=0, high=n_images, size=n_samples)

        for n in range(n_samples):

            sample_loc_sps[n, :] = encoding_func(x=sample_locs[n, 0], y=sample_locs[n, 1])
            sample_goal_sps[n, :] = encoding_func(x=sample_goals[n, 0], y=sample_goals[n, 1])
            # x and y coordinates in the image for the output
            x = sample_locs[n, 0] + sample_goals[n, 0]
            y = sample_locs[n, 0] + sample_goals[n, 0]
            sample_output_dirs[n, :] = images[image_indices[n], x, y, :]
            sample_id_vecs[n, :] = id_vecs[image_indices[n]]

        # MazeDataset re-purposed for images (same format)
        dataset = MazeDataset(
            maze_ssp=sample_id_vecs,
            loc_ssps=sample_loc_sps,
            goal_ssps=sample_goal_sps,
            locs=sample_locs,
            goals=sample_goals,
            direction_outputs=sample_output_dirs,
        )

        if test_set == 0:
            trainloader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory
            )
        elif test_set == 1:
            testloader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory
            )

    return trainloader, testloader


class PolicyValidationSet(object):

    def __init__(self, data, dim, id_vecs, maze_indices, n_goals, subsample=1,
                 encoding_func=None, device=None, cache_fname='', seed=13
                 ):

        rng = np.random.RandomState(seed=seed)

        # Either cpu or cuda
        self.device = device

        # n_images by size by size by channel
        images = data['images']

        image_size = images.shape[1]

        limit_low = 0
        limit_high = image_size / 2

        res = int(limit_high / subsample)

        self.n_images = len(maze_indices)
        self.n_goals = n_goals

        if os.path.exists(cache_fname) or False:  # TODO: implement cache
            print("Loading visualization data from cache")

            cache_data = np.load(cache_fname)

            viz_maze_sps = cache_data['maze_ssp']
            viz_loc_sps = cache_data['loc_ssps']
            viz_goal_sps = cache_data['goal_ssps']
            viz_locs = cache_data['locs']
            viz_goals = cache_data['goals']
            viz_output_dirs = cache_data['direction_outputs']

            self.batch_size = res * res

        else:

            goal_sps = np.zeros((self.n_images, self.n_goals, dim))
            for ni in range(self.n_images):
                for gi in range(self.n_goals):
                    x = rng.randint(0, limit_high)
                    y = rng.randint(0, limit_high)
                    goal_sps[ni, gi, :] = encoding_func(x=x, y=y)

            n_samples = res * res * self.n_images * self.n_goals

            # Visualization
            viz_locs = np.zeros((n_samples, 2))
            viz_goals = np.zeros((n_samples, 2))
            viz_loc_sps = np.zeros((n_samples, goal_sps.shape[2]))
            viz_goal_sps = np.zeros((n_samples, goal_sps.shape[2]))
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

                            viz_loc_sps[si, :] = encoding_func(x=loc_x, y=loc_y)

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

            if cache_fname != '':
                print("Saving generated data to cache")

                np.savez(
                    cache_fname,
                    maze_ssp=viz_maze_sps,
                    loc_ssps=viz_loc_sps,
                    goal_ssps=viz_goal_sps,
                    locs=viz_locs,
                    goals=viz_goals,
                    direction_outputs=viz_output_dirs,
                )

        dataset_viz = MazeDataset(
            maze_ssp=viz_maze_sps,
            loc_ssps=viz_loc_sps,
            goal_ssps=viz_goal_sps,
            locs=viz_locs,
            goals=viz_goals,
            direction_outputs=viz_output_dirs,
        )

        # Each batch will contain the samples for one image. Must not be shuffled
        self.vizloader = torch.utils.data.DataLoader(
            dataset_viz, batch_size=self.batch_size, shuffle=False, num_workers=0,
        )

    def run_ground_truth(self, writer):

        with torch.no_grad():
            # Each maze is in one batch
            for i, data in enumerate(self.vizloader):
                print("Ground Truth Viz batch {} of {}".format(i + 1, self.n_mazes * self.n_goals))
                maze_loc_goal_ssps, directions, locs, goals = data

                wall_overlay = (directions.detach().numpy()[:, 0] == 0) & (directions.detach().numpy()[:, 1] == 0)

                # TODO: new function for this
                fig_truth, rmse = plot_path_predictions_image(
                    directions_pred=directions.detach().cpu().numpy(),
                    directions_true=directions.detach().cpu().numpy(),
                    wall_overlay=wall_overlay
                )

                # fig_truth_quiver = plot_path_predictions(
                #     directions=directions.detach().numpy(), coords=locs.detach().numpy(), dcell=self.xs[1] - self.xs[0]
                # )

                # Record figures to tensorboard
                writer.add_figure('v{}/ground truth'.format(i), fig_truth)
                # writer.add_figure('v{}/ground truth quiver'.format(i), fig_truth_quiver)

    # # Note that this must be a separate function because the previous cannot contain yields
    # def run_ground_truth_generator(self):
    #
    #     with torch.no_grad():
    #         # Each maze is in one batch
    #         for i, data in enumerate(self.vizloader):
    #             print("Ground Truth Viz batch {} of {}".format(i + 1, self.n_mazes * self.n_goals))
    #             maze_loc_goal_ssps, directions, locs, goals = data
    #
    #             wall_overlay = (directions.detach().numpy()[:, 0] == 0) & (directions.detach().numpy()[:, 1] == 0)
    #
    #             fig_truth, rmse = plot_path_predictions_image(
    #                 directions_pred=directions.detach().numpy(),
    #                 directions_true=directions.detach().numpy(),
    #                 wall_overlay=wall_overlay
    #             )
    #
    #             fig_truth_quiver = plot_path_predictions(
    #                 directions=directions.detach().numpy(), coords=locs.detach().numpy(), dcell=self.xs[1] - self.xs[0]
    #             )
    #
    #             yield fig_truth, fig_truth_quiver

    def run_validation(self, model, writer, epoch, use_wall_overlay=True):
        criterion = nn.MSELoss()

        with torch.no_grad():
            # Each maze is in one batch
            for i, data in enumerate(self.vizloader):
                print("Viz batch {} of {}".format(i + 1, self.n_images * self.n_goals))
                maze_loc_goal_ssps, directions, locs, goals = data

                outputs = model(maze_loc_goal_ssps.to(self.device))

                loss = criterion(outputs, directions.to(self.device))

                if use_wall_overlay:

                    wall_overlay = (directions.detach().numpy()[:, 0] == 0) & (directions.detach().numpy()[:, 1] == 0)

                    fig_pred, rmse = plot_path_predictions_image(
                        directions_pred=outputs.detach().cpu().numpy(),
                        directions_true=directions.detach().cpu().numpy(),
                        wall_overlay=wall_overlay
                    )

                    # fig_pred_quiver = plot_path_predictions(
                    #     directions=outputs.detach().numpy(), coords=locs.detach().numpy(), dcell=self.xs[1] - self.xs[0], wall_overlay=wall_overlay
                    # )

                    writer.add_scalar(tag='viz_rmse/{}'.format(i), scalar_value=rmse, global_step=epoch)
                else:

                    fig_pred = plot_path_predictions(
                        directions=outputs.detach().cpu().numpy(), coords=locs.detach().cpu().numpy(), type='colour'
                    )

                    # fig_pred_quiver = plot_path_predictions(
                    #     directions=outputs.detach().numpy(), coords=locs.detach().numpy(), dcell=self.xs[1] - self.xs[0]
                    # )

                # Record figures to tensorboard
                writer.add_figure('v{}/viz set predictions'.format(i), fig_pred, epoch)
                # writer.add_figure('v{}/viz set predictions quiver'.format(i), fig_pred_quiver, epoch)
                writer.add_scalar(tag='viz_loss/{}'.format(i), scalar_value=loss.data.item(), global_step=epoch)

    # # Note that this must be a separate function because the previous cannot contain yields
    # def run_validation_generator(self, model, epoch, use_wall_overlay=True):
    #     criterion = nn.MSELoss()
    #
    #     with torch.no_grad():
    #         # Each maze is in one batch
    #         for i, data in enumerate(self.vizloader):
    #             print("Viz batch {} of {}".format(i + 1, self.n_images * self.n_goals))
    #             maze_loc_goal_ssps, directions, locs, goals = data
    #
    #             outputs = model(maze_loc_goal_ssps.to(self.device))
    #
    #             loss = criterion(outputs, directions)
    #
    #             if use_wall_overlay:
    #
    #                 wall_overlay = (directions.detach().numpy()[:, 0] == 0) & (directions.detach().numpy()[:, 1] == 0)
    #
    #                 fig_pred, rmse = plot_path_predictions_image(
    #                     directions_pred=outputs.detach().cpu().numpy(),
    #                     directions_true=directions.detach().cpu().numpy(),
    #                     wall_overlay=wall_overlay
    #                 )
    #
    #                 fig_pred_quiver = plot_path_predictions(
    #                     directions=outputs.detach().numpy(), coords=locs.detach().numpy(), dcell=self.xs[1] - self.xs[0], wall_overlay=wall_overlay
    #                 )
    #             else:
    #
    #                 fig_pred = plot_path_predictions(
    #                     directions=outputs.detach().numpy(), coords=locs.detach().numpy(), type='colour'
    #                 )
    #
    #                 fig_pred_quiver = plot_path_predictions(
    #                     directions=outputs.detach().numpy(), coords=locs.detach().numpy(), dcell=self.xs[1] - self.xs[0]
    #                 )
    #
    #             yield fig_pred, fig_pred_quiver

    def get_rmse(self, model):

        ret = np.zeros((self.n_images * self.n_goals, 2))

        with torch.no_grad():
            # Each maze is in one batch
            for i, data in enumerate(self.vizloader):
                print("Viz batch {} of {}".format(i + 1, self.n_images * self.n_goals))
                maze_loc_goal_ssps, directions, locs, goals = data

                outputs = model(maze_loc_goal_ssps.to(self.device))

                wall_overlay = (directions.detach().numpy()[:, 0] == 0) & (directions.detach().numpy()[:, 1] == 0)

                rmse, angle_rmse = compute_rmse(
                    directions_pred=outputs.detach().cpu().numpy(),
                    directions_true=directions.detach().cpu().numpy(),
                    wall_overlay=wall_overlay
                )

                ret[i, 0] = rmse
                ret[i, 1] = angle_rmse

        return ret


def compute_rmse(directions_pred, directions_true, wall_overlay=None):
    """ Computes just the RMSE, without generating a figure """

    angles_flat_pred = np.arctan2(directions_pred[:, 1], directions_pred[:, 0])
    angles_flat_true = np.arctan2(directions_true[:, 1], directions_true[:, 0])

    # Create 3 possible offsets to cover all cases
    angles_offset_true = np.zeros((len(angles_flat_true), 3))
    angles_offset_true[:, 0] = angles_flat_true - 2 * np.pi
    angles_offset_true[:, 1] = angles_flat_true
    angles_offset_true[:, 2] = angles_flat_true + 2 * np.pi

    angles_offset_true -= angles_flat_pred.reshape(len(angles_flat_pred), 1)
    angles_offset_true = np.abs(angles_offset_true)

    angle_error = np.min(angles_offset_true, axis=1)

    angle_squared_error = angle_error**2
    if wall_overlay is not None:
        angle_rmse = np.sqrt(angle_squared_error[np.where(wall_overlay == 0)].mean())
    else:
        angle_rmse = np.sqrt(angle_squared_error.mean())

    sin = np.sin(angles_flat_pred)
    cos = np.cos(angles_flat_pred)

    pred_dir_normalized = np.vstack([cos, sin]).T

    squared_error = (pred_dir_normalized - directions_true)**2

    # only calculate mean across the non-wall elements
    # mse = np.mean(squared_error[np.where(wall_overlay == 0)])
    if wall_overlay is not None:
        mse = squared_error[np.where(wall_overlay == 0)].mean()
    else:
        mse = squared_error.mean()

    rmse = np.sqrt(mse)

    return rmse, angle_rmse
